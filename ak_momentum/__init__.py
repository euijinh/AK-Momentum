"""
AK-MOMENTUM — activation-keyed momentum via the delta rule.

The gradient of a linear layer factors into an input activation that acts as a
key and an output-side error that acts as a value.  AK-MOMENTUM keys the
momentum buffer on that activation and updates it by the classical delta rule,
so each direction is forgotten at a rate set by how often it appears, instead of
the single fixed rate an exponential moving average applies everywhere.

    M_t = beta M_{t-1} + eta (G_t - M_{t-1} Sigma_hat_t),
    Sigma_hat = X_hat^T X_hat / N,   x_hat_i = x_i / ||x_i||

It is a drop-in replacement for the first-moment buffer of any optimizer; the
second moment, weight decay and any preconditioning are untouched.

Quick start::

    from ak_momentum import configure_ak_adamw

    optimizer, capture = configure_ak_adamw(
        model, lr=3e-4, betas=(0.99, 0.99), weight_decay=0.1, eta=0.37,
    )

    for batch in loader:
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()          # consumes the captured keys
        optimizer.zero_grad()

Recommended recipe: ``beta = 0.99``, ``eta = 0.4``, second moment and weight
decay at their AdamW values, and a learning rate near one tenth of the tuned
AdamW one -- then tune only the learning rate.
"""

from .ak_adamw import AKAdamW, configure_ak_adamw
from .ak_sgd import AKSGD, ak_momentum_update, configure_ak_sgd
from .capture import (
    ActivationCapture,
    normalize_rows,
    normalize_rows_eager,
    set_fused_norm,
)
from .groups import (
    build_mup_param_groups,
    categorize_params,
    mup_hidden_lr_scale,
    mup_readout_lr,
)

__version__ = "0.1.0"

__all__ = [
    "AKAdamW",
    "AKSGD",
    "ActivationCapture",
    "ak_momentum_update",
    "build_mup_param_groups",
    "categorize_params",
    "configure_ak_adamw",
    "configure_ak_sgd",
    "mup_hidden_lr_scale",
    "mup_readout_lr",
    "normalize_rows",
    "normalize_rows_eager",
    "set_fused_norm",
    "__version__",
]
