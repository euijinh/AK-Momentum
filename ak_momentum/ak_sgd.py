"""
AK-SGD — SGD-with-momentum whose buffer is updated by the delta rule.

    SGD-momentum      m_t = beta m_{t-1} + (1 - beta) g_t
    AK-SGD            m_t = beta m_{t-1} + eta (G_t - m_{t-1} Sigma_hat_t)

    update            theta_{t+1} = theta_t - lr u_t
                      u_t = m_t                       (nesterov=False)
                      u_t = lerp(g_t, m_t, beta)      (nesterov=True)

Weight decay is decoupled (``theta <- (1 - lr wd) theta``).  Parameters with no
captured activation -- 1-D gains and biases, or anything uncaptured -- fall back
to the plain EMA buffer, so those behave exactly like SGD-momentum.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .capture import ActivationCapture


def ak_momentum_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    eta: float = 0.05,
    nesterov: bool = True,
    delta_update: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Advance one buffer in place and return the descent direction.

    Args:
        delta_update: ``G - M Sigma_hat`` when the layer was captured; None
            falls back to the EMA buffer.
    """
    g, m = grad, momentum
    if g.ndim <= 1 or delta_update is None:
        m.lerp_(g, 1 - beta)
    else:
        m.mul_(beta).add_(delta_update.view(m.shape), alpha=eta)
    return torch.lerp(g, m, beta) if nesterov else m.clone()


class AKSGD(torch.optim.Optimizer):
    """SGD-momentum with an activation-keyed (delta-rule) buffer.

    Works unchanged under DDP: state stays replicated and `ActivationCapture`
    reduces the correction inputs, so every rank computes the same update.

    Args:
        params: parameters or parameter groups.
        lr: learning rate.  The delta-corrected buffer is smaller in magnitude
            than an EMA buffer, so the optimum sits above the SGD one.
        weight_decay: decoupled weight decay.
        momentum: the uniform decay beta of the delta rule.
        eta: delta-rule coefficient (width-invariant under muP).
        nesterov: use the lookahead direction ``lerp(g, m, beta)``.
        activation_capture: capture supplying the correction; None gives plain
            SGD-momentum.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        eta: float = 0.05,
        nesterov: bool = True,
        activation_capture: Optional[ActivationCapture] = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if eta < 0.0:
            raise ValueError(f"Invalid eta: {eta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum,
            eta=eta, nesterov=nesterov,
        )
        super().__init__(params, defaults)
        self.activation_capture = activation_capture
        self._param_to_name: Dict[int, str] = {}

    def register_param_names(self, model: nn.Module) -> None:
        """Map weights to module names and bind the capture to this optimizer."""
        for name, module in model.named_modules():
            for pname, p in module.named_parameters(recurse=False):
                if pname == "weight":
                    self._param_to_name[id(p)] = name
        if self.activation_capture is not None:
            self.activation_capture.bind_optimizer(self, momentum_key="momentum_buffer")

    def _buffer(self, p: torch.Tensor) -> torch.Tensor:
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p)
        return state["momentum_buffer"]

    @torch.no_grad()
    def flush_microbatch(self) -> None:
        """Fold any live captures into the accumulators and free them."""
        if self.activation_capture is None:
            return
        name_to_mom = {}
        for group in self.param_groups:
            for p in group["params"]:
                name = self._param_to_name.get(id(p))
                if name is not None:
                    name_to_mom[name] = self._buffer(p)
        self.activation_capture.flush_microbatch(name_to_mom)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        cap = self.activation_capture
        if cap is not None:
            if cap._inputs:
                self.flush_microbatch()
            cap.sync_accumulators()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta = group["momentum"]
            eta = group["eta"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                buf = self._buffer(p)

                delta_update = None
                if cap is not None and grad.ndim >= 2:
                    name = self._param_to_name.get(id(p))
                    if name is not None:
                        delta_update = cap.compute_delta_update(name, buf, grad=grad)

                update = ak_momentum_update(
                    grad, buf, beta=beta, eta=eta,
                    nesterov=nesterov, delta_update=delta_update,
                )
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(update.reshape(p.shape), alpha=-lr)

        if cap is not None:
            cap.clear()
        return loss


def configure_ak_sgd(
    model: nn.Module,
    lr: float = 0.02,
    weight_decay: float = 0.0,
    momentum: float = 0.95,
    eta: float = 0.05,
    nesterov: bool = True,
    factored: bool = False,
    chunk_size: int = 4096,
    chunk_budget_frac: float = 0.02,
    shape_adaptive: bool = True,
    bf16_gram: bool = False,
    share_sigma: bool = True,
    param_groups: Optional[List[dict]] = None,
) -> Tuple[AKSGD, ActivationCapture]:
    """Build AK-SGD over `model`, with the capture wired up.

    Returns:
        ``(optimizer, capture)``.
    """
    capture = ActivationCapture(
        factored=factored, chunk_size=chunk_size,
        chunk_budget_frac=chunk_budget_frac, shape_adaptive=shape_adaptive,
        bf16_gram=bf16_gram, share_sigma=share_sigma,
    )
    capture.register(model)
    optimizer = AKSGD(
        param_groups if param_groups is not None else model.parameters(),
        lr=lr, weight_decay=weight_decay, momentum=momentum,
        eta=eta, nesterov=nesterov, activation_capture=capture,
    )
    optimizer.register_param_names(model)
    return optimizer, capture
