#!/usr/bin/env python3
"""
FineWeb-Edu pre-training: AdamW vs AK-AdamW.

This is the headline comparison of the paper. The two arms differ in exactly
one place -- how the first moment is formed -- so every other component
(architecture, muP parameterization, data order, schedule, clipping, bf16) is
shared code here.

    accelerate launch --num_processes 1 --mixed_precision bf16 \
        reproduce/lm/train.py --config reproduce/lm/configs/67m_ak_adamw.yaml \
        --data_dir ./data/fineweb-edu

Pass ``--synthetic`` to run the whole pipeline on random tokens, which needs no
corpus and is the quickest way to check an installation.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ak_momentum import build_mup_param_groups, configure_ak_adamw  # noqa: E402
from reproduce.lm.data import SyntheticDataset, build_datasets  # noqa: E402
from reproduce.lm.model import LanguageModel, ModelConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="YAML config (see configs/).")
    p.add_argument("--data_dir", default=None, help="Overrides data.data_dir.")
    p.add_argument("--dataset_name", default=None, help="Overrides data.dataset_name.")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Overrides training.max_steps, which is also the schedule horizon.")
    p.add_argument("--stop_after", type=int, default=None,
                   help="Stop after this many optimizer steps while the cosine schedule "
                        "still spans training.max_steps, so a short run follows the "
                        "opening segment of the full one.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default="./runs", help="Where metrics land.")
    p.add_argument("--synthetic", action="store_true",
                   help="Train on random tokens (smoke test; no corpus needed).")
    return p.parse_args()


def load_config(path: str, args: argparse.Namespace) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if args.data_dir is not None:
        cfg["data"]["data_dir"] = args.data_dir
    if args.dataset_name is not None:
        cfg["data"]["dataset_name"] = args.dataset_name
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    return cfg


def build_optimizer(model, cfg: Dict[str, Any]):
    """Build AdamW or AK-AdamW over the three muP groups.

    Both arms use the same grouping: hidden 2-D weights and the readout carry
    the O(1/n) width rule, embeddings and 1-D parameters stay at the O(1) rate
    with no weight decay.  The baseline drives all three from one rate (AdamW's
    own muP rule); AK-AdamW additionally tunes the auxiliary rate, because
    tying it to its hidden rate, ~40x below the tuned auxiliary one, would
    starve those parameters.
    """
    o, m = cfg["optimizer"], cfg["model"]
    groups, stats = build_mup_param_groups(
        model,
        lr=o["lr"],
        d_model=m["d_model"],
        base_width=m["mup_base_width"],
        weight_decay=o["weight_decay"],
        family="adamw",
        aux_lr=o.get("aux_lr"),
        aux_weight_decay=0.0,
    )
    name = o["name"].lower()
    if name == "adamw":
        opt = torch.optim.AdamW(
            groups, lr=stats["hidden_lr"], betas=tuple(o["betas"]), eps=o.get("eps", 1e-8),
        )
        capture = None
    elif name in ("ak_adamw", "akadamw"):
        if o.get("bf16_gram", False):
            # Mirrors the paper runs: the correction GEMMs run on tensor cores,
            # so any remaining fp32 matmul is allowed the same treatment.
            torch.set_float32_matmul_precision("high")
        opt, capture = configure_ak_adamw(
            model,
            lr=stats["hidden_lr"], betas=tuple(o["betas"]), eps=o.get("eps", 1e-8),
            weight_decay=o["weight_decay"], eta=o["eta"],
            factored=o.get("factored", False),
            bf16_gram=o.get("bf16_gram", False),
            param_groups=groups,
        )
    else:
        raise ValueError(f"Unknown optimizer '{o['name']}'. Use 'adamw' or 'ak_adamw'.")
    return opt, capture, stats


def build_scheduler(optimizer, cfg: Dict[str, Any]):
    """Linear warmup, then cosine decay to ``min_lr_ratio`` of each group's rate."""
    t = cfg["training"]
    warmup, total = t["warmup_steps"], t["max_steps"]
    floor = t.get("min_lr_ratio", 0.1)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, accelerator, max_batches: int) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        ids = batch["input_ids"]
        loss = model(input_ids=ids, labels=ids)["loss"]
        total += accelerator.reduce(loss, reduction="mean").item()
        n += 1
        if n >= max_batches:
            break
    model.train()
    return total / max(1, n)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args)
    t, d = cfg["training"], cfg["data"]
    stop = t["max_steps"] if args.stop_after is None else min(args.stop_after, t["max_steps"])

    accelerator = Accelerator(
        mixed_precision="bf16" if t.get("bf16", True) else "no",
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
    )
    # Seeded explicitly, so initialization and data order are reproducible and
    # identical across the two arms of a comparison.
    set_seed(t["seed"])

    model = LanguageModel(ModelConfig(**cfg["model"]))
    optimizer, capture, stats = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    if args.synthetic:
        train_set = SyntheticDataset(cfg["model"]["vocab_size"], t["seq_length"], 512)
        eval_set = SyntheticDataset(cfg["model"]["vocab_size"], t["seq_length"], 64, seed=1)
    else:
        train_set, eval_set = build_datasets(
            d["data_dir"], d["dataset_name"], t["seq_length"], d.get("test_size", 5.71e-5),
        )

    train_loader = DataLoader(
        train_set, batch_size=t["batch_size"], shuffle=True, drop_last=True,
        num_workers=t.get("num_workers", 2),
    )
    eval_loader = DataLoader(eval_set, batch_size=t["batch_size"], shuffle=False)
    model, optimizer, train_loader, eval_loader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader,
    )

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tokens = (t["batch_size"] * t["seq_length"] * t["gradient_accumulation_steps"]
                  * accelerator.num_processes)
        print(f"optimizer   : {cfg['optimizer']['name']}")
        print(f"parameters  : {n_params:,}")
        print(f"hidden lr   : {stats['hidden_lr']:.4e}  (base {cfg['optimizer']['lr']:.4e} "
              f"x {stats['lr_scale']:.4f})")
        print(f"head lr     : {stats['head_lr']:.4e}   other lr: {stats['other_lr']:.4e}")
        print(f"tokens/step : {tokens:,}   steps: {t['max_steps']:,}"
              + (f"   (stopping after {stop:,})" if stop < t["max_steps"] else ""), flush=True)

    out_dir = Path(args.out) / Path(args.config).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    history, step, start = [], 0, time.perf_counter()

    model.train()
    while step < stop:
        for batch in train_loader:
            if step >= stop:
                break
            ids = batch["input_ids"]
            with accelerator.accumulate(model):
                loss = model(input_ids=ids, labels=ids)["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # Clip first: on the non-factored path the clipped gradient
                    # is the gradient term the delta rule consumes.
                    accelerator.clip_grad_norm_(model.parameters(), t.get("grad_clip", 1.0))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step += 1

                    if step % t.get("log_steps", 50) == 0 and accelerator.is_main_process:
                        print(f"step {step:>6}  loss {loss.item():.4f}  "
                              f"lr {scheduler.get_last_lr()[0]:.3e}  "
                              f"{time.perf_counter() - start:.0f}s", flush=True)
                    if step % t["eval_steps"] == 0:
                        ev = evaluate(model, eval_loader, accelerator,
                                      t.get("eval_max_batches", 100))
                        if accelerator.is_main_process:
                            print(f"step {step:>6}  eval_loss {ev:.4f}  ppl {math.exp(ev):.2f}",
                                  flush=True)
                            history.append({"step": step, "eval_loss": ev})
                            (out_dir / "metrics.json").write_text(json.dumps(history, indent=2))

    if accelerator.is_main_process:
        print(f"done in {(time.perf_counter() - start) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
