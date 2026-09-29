#!/usr/bin/env python3
"""
CIFAR-10 training: SGD vs AK-SGD, and AdamW vs AK-AdamW.

These are the paper's generality checks -- that the gain is not tied to Adam's
diagonal preconditioner, nor to language modelling.

    # MLP, the optimizer-family check
    python reproduce/vision/train.py --model mlp --optimizer sgd    --lr 0.1 --momentum 0.9
    python reproduce/vision/train.py --model mlp --optimizer ak_sgd --lr 1.7 --momentum 0.99 \
        --eta 0.99 --factored

    # ResNet-18 / ViT-Tiny, the architecture check
    python reproduce/vision/train.py --model resnet18 --optimizer adamw    --lr 5e-3 --momentum 0.95
    python reproduce/vision/train.py --model resnet18 --optimizer ak_adamw --lr 1e-4 --momentum 0.99 --eta 0.37

Everything else is held at the values in `reproduce/README.md`.
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ak_momentum import AKSGD, ActivationCapture, categorize_params, configure_ak_adamw  # noqa: E402
from reproduce.vision.data import build_dataloaders  # noqa: E402
from reproduce.vision.models import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="resnet18", choices=["mlp", "resnet18", "vit_tiny"])
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw", "ak_sgd", "ak_adamw"])
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--momentum", type=float, default=0.9, help="SGD momentum, or Adam beta1.")
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--eta", type=float, default=0.37, help="Delta-rule coefficient.")
    p.add_argument("--aux-lr", type=float, default=3e-4,
                   help="AK-SGD only: AdamW rate for the head and 1-D parameters.")
    p.add_argument("--factored", action="store_true",
                   help="Use dY^T X_hat as the gradient term instead of param.grad.")
    p.add_argument("--nesterov", action="store_true", default=True)
    p.add_argument("--no-nesterov", dest="nesterov", action="store_false")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", default="./data/cifar10")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out", default="./runs")
    return p.parse_args()


class MultiOptimizer:
    """Presents several optimizers as one, for the AK-SGD + aux AdamW split."""

    def __init__(self, optimizers: List[torch.optim.Optimizer]):
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list:
        return [g for o in self.optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for o in self.optimizers:
            o.step()


def build_optimizer(model: nn.Module, args: argparse.Namespace):
    """Return ``(optimizer, schedulable_optimizers)``.

    ``sgd`` / ``adamw`` / ``ak_adamw`` put every parameter in one group, so the
    two arms of each comparison differ only in the momentum rule and the tuned
    learning rate.  ``ak_sgd`` follows the paper's MLP run: AK-SGD drives the
    hidden matrices while the head and the 1-D parameters go to an auxiliary
    AdamW, since a plain SGD rate tuned for the corrected buffer does not suit
    them.
    """
    name = args.optimizer
    if name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay, nesterov=args.nesterov)
        return opt, [opt]
    if name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                betas=(args.momentum, args.beta2),
                                weight_decay=args.weight_decay)
        return opt, [opt]
    if name == "ak_adamw":
        opt, _ = configure_ak_adamw(
            model, lr=args.lr, betas=(args.momentum, args.beta2),
            weight_decay=args.weight_decay, eta=args.eta, factored=args.factored,
        )
        return opt, [opt]

    hidden, head, other = categorize_params(model)
    capture = ActivationCapture(factored=args.factored)
    capture.register(model)
    ak = AKSGD(hidden, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum,
               eta=args.eta, nesterov=args.nesterov, activation_capture=capture)
    ak.register_param_names(model)
    aux_groups = []
    if head:
        aux_groups.append({"params": head, "lr": args.aux_lr,
                           "weight_decay": args.weight_decay})
    if other:
        aux_groups.append({"params": other, "lr": args.aux_lr, "weight_decay": 0.0})
    aux = torch.optim.AdamW(aux_groups, lr=args.aux_lr, betas=(0.9, 0.95))
    return MultiOptimizer([ak, aux]), [ak, aux]


class WarmupCosine(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup over epochs, then cosine decay to an absolute `min_lr`."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            scale = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base * scale for base in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.min_lr + (base - self.min_lr) * cosine for base in self.base_lrs]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def run_epoch(model, loader, criterion, optimizer, device) -> dict:
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        seen += targets.size(0)
    return {"train_loss": total_loss / seen, "train_acc": 100.0 * correct / seen}


@torch.no_grad()
def validate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(inputs)
        total_loss += criterion(outputs, targets).item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        seen += targets.size(0)
    return {"val_loss": total_loss / seen, "val_acc": 100.0 * correct / seen}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.model).to(device)
    train_loader, val_loader = build_dataloaders(args.data_root, args.batch_size, args.num_workers)
    optimizer, schedulable = build_optimizer(model, args)
    schedulers = [WarmupCosine(o, args.warmup_epochs, args.epochs, args.min_lr)
                  for o in schedulable]
    criterion = nn.CrossEntropyLoss()

    tag = f"{args.model}_{args.optimizer}_lr{args.lr:g}_seed{args.seed}"
    out_dir = Path(args.out) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"{tag}: {sum(p.numel() for p in model.parameters()):,} parameters on {device}")

    history, best, start = [], 0.0, time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        for s in schedulers:
            s.step()
        best = max(best, val_metrics["val_acc"])
        row = {"epoch": epoch, **train_metrics, **val_metrics,
               "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        (out_dir / "metrics.json").write_text(
            json.dumps({"best_acc": best, "history": history}, indent=2))
        print(f"epoch {epoch:>3}/{args.epochs}  train {row['train_loss']:.4f} "
              f"({row['train_acc']:.2f}%)  val {row['val_loss']:.4f} "
              f"({row['val_acc']:.2f}%)  best {best:.2f}%", flush=True)

    print(f"best val acc {best:.2f}% in {(time.perf_counter() - start) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
