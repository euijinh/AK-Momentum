"""
Parameter grouping and muP learning-rate rules.

AK-MOMENTUM adds exactly one hyperparameter (eta) to the base optimizer, and
that one does not scale with width: the normalized key makes
``|| x_hat x_hat^T ||_op = 1`` at every width, so eta = Theta(1) is the
feature-learning choice (Theorem Q.1 of the paper).  The base optimizer's muP
learning-rate rule is therefore inherited unchanged:

    hidden weights      AK-AdamW  lr ~ Theta(1/n)      AK-SGD  lr ~ Theta(1)
    readout (lm_head)   Adam-updated head: lr ~ Theta(1/n)
    embeddings, 1-D     unchanged, Theta(1)

`n` is the current width and `base_width` the proxy width the hyperparameters
were tuned at, so every multiplier is 1.0 when they coincide.
"""

import math
from typing import List, Optional, Tuple

import torch.nn as nn

__all__ = [
    "categorize_params",
    "mup_hidden_lr_scale",
    "mup_readout_lr",
    "build_mup_param_groups",
]

_READOUT_NAMES = ("lm_head", "head", "output_projection", "fc")


def _is_readout(name: str) -> bool:
    n = name.lower()
    return n in _READOUT_NAMES or any(n.endswith("." + r) for r in _READOUT_NAMES)


def categorize_params(model: nn.Module):
    """Split parameters into ``(hidden, head, other)``.

    - **hidden**: 2-D+ weights of interior layers; these carry the muP width rule
      and are the ones AK-MOMENTUM corrects.
    - **head**: the readout weight, when it is a separate tensor.
    - **other**: embeddings, biases, norm gains, and a tied readout.

    A tied readout stays in `other`, since its learning rate cannot then be
    scaled independently of the embedding.
    """
    embed_ids, head_ids = set(), set()
    for name, module in model.named_modules():
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
            for p in module.parameters(recurse=False):
                embed_ids.add(id(p))
        if _is_readout(name):
            for pname, p in module.named_parameters(recurse=False):
                if pname == "weight":
                    head_ids.add(id(p))

    tied = head_ids & embed_ids
    hidden: List[nn.Parameter] = []
    head: List[nn.Parameter] = []
    other: List[nn.Parameter] = []
    seen = set()
    for _, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if id(p) in embed_ids or id(p) in tied:
            other.append(p)
        elif id(p) in head_ids:
            head.append(p)
        elif p.ndim >= 2:
            hidden.append(p)
        else:
            other.append(p)
    return hidden, head, other


def mup_hidden_lr_scale(d_model: int, base_width: int, family: str) -> float:
    """muP multiplier for hidden-weight learning rates.

    Args:
        family: ``'adamw'`` (Theta(1/n), so ``base_width / d_model``) or
            ``'sgd'`` (Theta(1), so 1.0).  AK-AdamW and AdamW share the AdamW
            rule; AK-SGD and SGD-momentum share the SGD rule.
    """
    f = family.lower()
    if f in ("adamw", "adam", "ak_adamw", "akadamw"):
        return base_width / d_model
    if f in ("sgd", "ak_sgd", "aksgd", "momentum"):
        return 1.0
    raise ValueError(f"Unknown optimizer family '{family}'; expected 'adamw' or 'sgd'.")


def mup_readout_lr(aux_lr: float, d_model: int, base_width: int) -> float:
    """muP learning rate for an Adam-updated readout: ``aux_lr * base/n``."""
    return aux_lr * (base_width / d_model)


def build_mup_param_groups(
    model: nn.Module,
    lr: float,
    d_model: int,
    base_width: int,
    weight_decay: float,
    family: str = "adamw",
    aux_lr: Optional[float] = None,
    aux_weight_decay: float = 0.0,
) -> Tuple[List[dict], dict]:
    """Build the three muP groups used by the paper's language-model runs.

    hidden -> ``lr * scale``, readout -> ``aux_lr * base/n``, everything else ->
    ``aux_lr`` with `aux_weight_decay`.  When `aux_lr` is None the base `lr` is
    used for the auxiliary groups, which reproduces the single-rate AdamW
    baseline.

    Returns:
        ``(param_groups, stats)`` where `stats` records the resolved rates, for
        logging or assertions.
    """
    hidden, head, other = categorize_params(model)
    scale = mup_hidden_lr_scale(d_model, base_width, family)
    hidden_lr = lr * scale
    base_aux = lr if aux_lr is None else aux_lr
    head_lr = mup_readout_lr(base_aux, d_model, base_width)

    groups: List[dict] = []
    if hidden:
        groups.append({"params": hidden, "lr": hidden_lr, "weight_decay": weight_decay})
    if head:
        groups.append({"params": head, "lr": head_lr, "weight_decay": weight_decay})
    if other:
        groups.append({"params": other, "lr": base_aux, "weight_decay": aux_weight_decay})

    stats = {
        "d_model": d_model,
        "base_width": base_width,
        "lr_scale": scale,
        "hidden_lr": hidden_lr,
        "head_lr": head_lr,
        "other_lr": base_aux,
        "hidden_tensors": len(hidden),
        "head_tensors": len(head),
        "other_tensors": len(other),
        "hidden_numel": sum(p.numel() for p in hidden),
        "head_numel": sum(p.numel() for p in head),
        "other_numel": sum(p.numel() for p in other),
    }
    return groups, stats
