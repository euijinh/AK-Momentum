"""
AK-AdamW — AdamW whose first moment is updated by the delta rule.

Only the first moment changes.  The second moment, bias correction, decoupled
weight decay and epsilon are exactly AdamW's:

    standard AdamW
        m_t = beta1 m_{t-1} + (1 - beta1) g_t
        v_t = beta2 v_{t-1} + (1 - beta2) g_t^2
        theta_t = (1 - lr wd) theta_{t-1} - lr m_hat_t / (sqrt(v_hat_t) + eps)

    AK-AdamW (captured 2-D+ weights)
        m_t = beta1 m_{t-1} + eta (G_t - m_{t-1} Sigma_hat_t)
        v_t, m_hat_t, v_hat_t, theta_t : unchanged

    fallback (1-D parameters, embeddings, anything uncaptured)
        identical to AdamW.

Dropping the state-dependent forgetting factor recovers the EMA exactly (take
``Sigma_hat = 0`` and ``eta = 1 - beta1``), which is what makes the
AdamW-vs-AK-AdamW comparison a controlled one: the two differ only in how the
first moment is formed.  Note that ``eta = 0`` is *not* that limit -- it removes
the injection term as well, leaving ``m_t = beta1 m_{t-1}``, which decays to
zero.

The second moment is deliberately *not* delta-corrected.  It tracks per-
coordinate gradient magnitude, a scalar property of each entry, whereas the
delta rule acts on input directions; Sigma_hat carries direction only (it has
unit trace), so the two compose instead of duplicating each other.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .capture import ActivationCapture


def _adamw_tail_forloop(
    params, grads, exp_avgs, exp_avg_sqs, steps, deltas,
    *, beta1, beta2, eps, lr, wd, eta,
) -> None:
    """Per-tensor AdamW tail; the reference the foreach path must reproduce."""
    for p, grad, exp_avg, exp_avg_sq, step, delta in zip(
        params, grads, exp_avgs, exp_avg_sqs, steps, deltas
    ):
        exp_avg_sq.lerp_(grad * grad, 1 - beta2)
        if delta is not None:
            exp_avg.mul_(beta1).add_(delta, alpha=eta)
        else:
            exp_avg.lerp_(grad, 1 - beta1)
        bias_correction1 = 1 - beta1 ** step
        bias_correction2_sq = (1 - beta2 ** step) ** 0.5
        if wd > 0:
            p.mul_(1 - lr * wd)
        denom = (exp_avg_sq.sqrt() / bias_correction2_sq).add_(eps)
        p.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)


def _adamw_tail_foreach(
    params, grads, exp_avgs, exp_avg_sqs, steps, deltas,
    *, beta1, beta2, eps, lr, wd, eta,
) -> None:
    """The same scalar operations, dispatched once across all parameters.

    Each op is elementwise and per tensor independent, with the same Python
    float scalars, so this reproduces `_adamw_tail_forloop` while issuing ~8
    kernel launches instead of ~8 per parameter.
    """
    gg = torch._foreach_mul(grads, grads)
    torch._foreach_lerp_(exp_avg_sqs, gg, 1 - beta2)

    idx_delta = [i for i, d in enumerate(deltas) if d is not None]
    idx_ema = [i for i, d in enumerate(deltas) if d is None]
    if idx_ema:
        torch._foreach_lerp_(
            [exp_avgs[i] for i in idx_ema], [grads[i] for i in idx_ema], 1 - beta1,
        )
    if idx_delta:
        ea = [exp_avgs[i] for i in idx_delta]
        torch._foreach_mul_(ea, beta1)
        torch._foreach_add_(ea, [deltas[i] for i in idx_delta], alpha=eta)

    if wd > 0:
        torch._foreach_mul_(params, 1 - lr * wd)

    denom = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denom, [(1 - beta2 ** s) ** 0.5 for s in steps])
    torch._foreach_add_(denom, eps)
    torch._foreach_addcdiv_(
        params, exp_avgs, denom, [-lr / (1 - beta1 ** s) for s in steps],
    )


class AKAdamW(torch.optim.Optimizer):
    """AdamW with an activation-keyed (delta-rule) first moment.

    Works unchanged under DDP: parameters and optimizer state stay replicated,
    the gradient is the all-reduced one, and `ActivationCapture` reduces the
    correction inputs, so every rank computes an identical update.

    Args:
        params: parameters or parameter groups, as for any torch optimizer.
        lr: learning rate.  AK-AdamW's optimum is typically well below AdamW's,
            because the buffer estimates a Wiener-type predictor rather than an
            average of gradients; roughly a tenth of the tuned AdamW value is a
            good starting point.
        betas: ``(beta1, beta2)``.  beta1 is the uniform decay of the delta rule,
            and a larger value than AdamW prefers is expected: the per-direction
            coefficient is ``beta1 - eta lambda_i``, so every direction with
            positive input density already forgets faster than nominal.
        eps: AdamW epsilon.
        weight_decay: decoupled weight decay.
        eta: delta-rule coefficient.  Width-invariant under muP; anything in
            [0.3, 0.5] is a safe default.  It scales the whole delta term, so
            ``eta = 0`` freezes the buffer rather than recovering AdamW; use the
            stock ``torch.optim.AdamW`` for that baseline.
        activation_capture: the capture supplying the correction.  When None,
            every parameter uses the standard AdamW first moment.
        foreach: batch the elementwise tail across parameters (default).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        eta: float = 0.05,
        activation_capture: Optional[ActivationCapture] = None,
        foreach: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if eta < 0.0:
            raise ValueError(f"Invalid eta: {eta}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, eta=eta)
        super().__init__(params, defaults)
        self.activation_capture = activation_capture
        self._param_to_name: Dict[int, str] = {}
        self._foreach = bool(foreach)

    def register_param_names(self, model: nn.Module) -> None:
        """Map weights to module names and bind the capture to this optimizer."""
        for name, module in model.named_modules():
            for pname, p in module.named_parameters(recurse=False):
                if pname == "weight":
                    self._param_to_name[id(p)] = name
        if self.activation_capture is not None:
            self.activation_capture.bind_optimizer(self, momentum_key="exp_avg")

    def _init_state(self, p: torch.Tensor) -> dict:
        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
        return state

    @torch.no_grad()
    def flush_microbatch(self) -> None:
        """Fold any live captures into the accumulators and free them.

        Backward hooks already do this, so calling it is optional; it exists for
        loops that want to bound memory explicitly.
        """
        if self.activation_capture is None:
            return
        name_to_mom = {}
        for group in self.param_groups:
            for p in group["params"]:
                name = self._param_to_name.get(id(p))
                if name is not None:
                    name_to_mom[name] = self._init_state(p)["exp_avg"]
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
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            eta = group["eta"]

            params, grads, exp_avgs, exp_avg_sqs, steps, deltas = [], [], [], [], [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self._init_state(p)
                state["step"] += 1
                exp_avg = state["exp_avg"]

                delta = None
                if cap is not None and p.grad.ndim >= 2:
                    name = self._param_to_name.get(id(p))
                    if name is not None:
                        update = cap.compute_delta_update(name, exp_avg, grad=p.grad)
                        if update is not None:
                            delta = update.view(exp_avg.shape)

                params.append(p)
                grads.append(p.grad)
                exp_avgs.append(exp_avg)
                exp_avg_sqs.append(state["exp_avg_sq"])
                steps.append(state["step"])
                deltas.append(delta)

            if not params:
                continue
            tail = _adamw_tail_foreach if self._foreach else _adamw_tail_forloop
            tail(params, grads, exp_avgs, exp_avg_sqs, steps, deltas,
                 beta1=beta1, beta2=beta2, eps=eps, lr=lr, wd=wd, eta=eta)

        if cap is not None:
            cap.clear()
        return loss


def configure_ak_adamw(
    model: nn.Module,
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    eta: float = 0.05,
    factored: bool = False,
    chunk_size: int = 4096,
    chunk_budget_frac: float = 0.02,
    shape_adaptive: bool = True,
    bf16_gram: bool = False,
    share_sigma: bool = True,
    foreach: bool = True,
    param_groups: Optional[List[dict]] = None,
) -> Tuple[AKAdamW, ActivationCapture]:
    """Build AK-AdamW over `model`, with the capture wired up.

    Every Linear / Conv weight gets the delta-corrected first moment; 1-D
    parameters fall back to AdamW.  Pass `param_groups` to control learning
    rates per group (see `ak_momentum.groups.build_mup_param_groups`);
    otherwise a single group covers all parameters.

    Returns:
        ``(optimizer, capture)``.  Keep the capture alive for the whole run;
        call ``capture.remove_hooks()`` to detach it.
    """
    capture = ActivationCapture(
        factored=factored, chunk_size=chunk_size,
        chunk_budget_frac=chunk_budget_frac, shape_adaptive=shape_adaptive,
        bf16_gram=bf16_gram, share_sigma=share_sigma,
    )
    capture.register(model)
    optimizer = AKAdamW(
        param_groups if param_groups is not None else model.parameters(),
        lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, eta=eta,
        activation_capture=capture, foreach=foreach,
    )
    optimizer.register_param_names(model)
    return optimizer, capture
