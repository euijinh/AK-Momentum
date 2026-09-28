"""
Activation capture for AK-MOMENTUM — per-layer keys and the delta correction.

AK-MOMENTUM replaces the EMA first moment of a base optimizer with the
classical delta rule, keyed on the layer's input activation:

    M_t = beta M_{t-1} + eta (delta_t - M_{t-1} x_hat_t) x_hat_t^T          (per sample)
        = M_{t-1} (beta I - eta x_hat_t x_hat_t^T) + eta delta_t x_hat_t^T

with the per-sample normalized key  x_hat_i = x_i / ||x_i||_2.  Batched over
N tokens of a step (Appendix L of the paper), the deployed rule is

    Sigma_hat = X_hat^T X_hat / N       in R^{n x n}        (input second moment)
    G         = the layer's gradient term                   (see `factored`)
    M        <- beta M + eta (G - M Sigma_hat)

This module computes the correction `M Sigma_hat`; the optimizers in
`ak_momentum.ak_adamw` / `ak_momentum.ak_sgd` apply the buffer update itself.

Key normalization is what makes the rule stable and width-transferable:
``||x_hat_i|| = 1`` gives ``tr(Sigma_hat) = 1`` and ``lambda_max(Sigma_hat) <= 1``,
so the per-direction contraction ``beta - eta lambda_i`` stays in
``[beta - eta, beta]`` for any data, and eta is width-invariant under muP.

Two gradient terms (`factored`)
-------------------------------
``factored=False`` (default)
    ``G = param.grad``, the ordinary weight gradient the base optimizer would
    have used.  Nothing extra is captured on the backward side, the correction
    is the only optimizer-specific work, and the rule is safe under DDP and
    gradient accumulation (the gradient is the all-reduced one).

``factored=True``
    ``G = dY^T X_hat``, i.e. the gradient formed against the *normalized* keys,
    captured from a backward hook.  This is the literal batch form of the
    per-sample rule.  It is rank-local, so it must not be combined with DDP's
    all-reduced gradient.

Both are exact instances of the rule; they differ in whether the gradient term
uses raw or normalized keys.  See the repository README for which setting
produced which experiment.

Contraction order
-----------------
``M Sigma_hat`` is evaluated in whichever order is cheaper for the layer shape
(Appendix M, eq. 41).  With m = fan_out and n = fan_in:

    n < 2m  ->  'gram'       M (X_hat^T X_hat) / N     ~ 2 n^2 N + 2 m n^2 FLOPs
    n >= 2m ->  'through_n'  (M X_hat^T) X_hat / N     ~ 4 m n N FLOPs

Both orders are exact and agree up to floating-point reassociation.  The Gram
matrix is additionally shared between layers that consume the *same* input
tensor (e.g. the two gated-MLP projections), which removes duplicate work
without changing the result.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.weak import WeakTensorKeyDictionary


# Documented tolerance for the optional bf16 correction path.  Sigma_hat is an
# N-averaged second moment of unit-norm keys, so per-element bf16 rounding
# (~2^-8) averages down; with fp32 accumulation the result stays well inside
# this bound.
BF16_GRAM_RTOL = 2e-2


def normalize_rows_eager(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample key normalization ``x_hat_i = x_i / ||x_i||_2``.

    Guarantees ``lambda_max(X_hat^T X_hat / N) <= 1``: for any unit u,
    ``u^T Sigma_hat u = (1/N) sum_i (u^T x_hat_i)^2 <= 1`` by Cauchy-Schwarz.

    This is the reference arithmetic: the norm is taken along dim=1 and the
    *norm* is clamped at `eps` (rather than `eps` being added to it).
    """
    return x / x.norm(dim=1, keepdim=True).clamp(min=eps)


# The norm+divide pair is three passes over X and two kernel launches; compiled
# it is one kernel and two passes.  On CUDA this cuts about 70% of the
# normalization term and is also *more* accurate than eager (inductor reduces in
# fp32).  It is CUDA-only and 2-D only; everywhere else, and on any compile or
# runtime failure, the eager reference runs instead.
_FUSED_NORM = True
_NORM_COMPILED = None
_NORM_COMPILE_FAILED = False


def set_fused_norm(enabled: bool) -> None:
    """Enable or disable the compiled row normalization process-wide.

    ``set_fused_norm(False)`` pins the eager reference, which is the arithmetic
    the paper's published runs used.  Compilation specializes on shape, so a
    model with many distinct token counts can exhaust dynamo's cache and fall
    back to eager silently; disable this lever if you would rather have one
    predictable path.
    """
    global _FUSED_NORM
    _FUSED_NORM = bool(enabled)


def _get_compiled_norm():
    global _NORM_COMPILED, _NORM_COMPILE_FAILED
    if _NORM_COMPILE_FAILED:
        return None
    if _NORM_COMPILED is None:
        try:
            _NORM_COMPILED = torch.compile(normalize_rows_eager, dynamic=False)
        except Exception as e:                                   # pragma: no cover
            _NORM_COMPILE_FAILED = True
            warnings.warn(
                f"[ak_momentum] torch.compile of the row normalization failed "
                f"({type(e).__name__}: {e}); using the eager reference."
            )
            return None
    return _NORM_COMPILED


def normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample key normalization, compiled where that is available."""
    global _NORM_COMPILE_FAILED
    if _FUSED_NORM and x.is_cuda and x.dim() == 2:
        fn = _get_compiled_norm()
        if fn is not None:
            try:
                return fn(x, eps)
            except Exception as e:                               # pragma: no cover
                _NORM_COMPILE_FAILED = True
                warnings.warn(
                    f"[ak_momentum] the compiled row normalization raised "
                    f"({type(e).__name__}: {e}); using the eager reference."
                )
    return normalize_rows_eager(x, eps)


def _bf16_gram_selfcheck(rtol: float = BF16_GRAM_RTOL) -> bool:
    """Check that the bf16 correction reproduces the fp32 one within `rtol`.

    Returns False when bf16 matmul is unavailable or out of tolerance, so the
    caller can fall back to the exact fp32 path.
    """
    try:
        gen = torch.Generator().manual_seed(0)
        m = torch.randn(32, 48, generator=gen)
        x_hat = normalize_rows(torch.randn(512, 48, generator=gen))
        ref = m @ ((x_hat.transpose(0, 1) @ x_hat) / 512)
        xb = x_hat.to(torch.bfloat16)
        sigma = (xb.transpose(0, 1) @ xb).to(torch.float32) / 512
        got = (m.to(torch.bfloat16) @ sigma.to(torch.bfloat16)).to(torch.float32)
        atol = rtol * max(1.0, float(ref.abs().max()))
        return bool(torch.allclose(got, ref, rtol=rtol, atol=atol))
    except Exception:
        return False


class _ShareGroup:
    """The set of captured Linear layers fed one and the same input tensor.

    Membership is established by *tensor identity* during the forward pass, not
    by module name, so it is re-derived every micro-batch and cannot go stale.
    A fresh group is created per micro-batch (a new forward makes a new input
    tensor), which is what makes `gram_done` a correct once-per-micro-batch
    latch.  Only the *formation* of Sigma_hat is shared; the multiply
    ``M @ Sigma_hat`` stays per layer, since each layer has its own M.
    """

    __slots__ = ("members", "gram_done", "_key")

    def __init__(self) -> None:
        self.members: List[str] = []
        self.gram_done: bool = False
        self._key: Optional[str] = None

    @property
    def key(self) -> str:
        # min() is order-independent, so every micro-batch and every rank picks
        # the same accumulator slot.
        if self._key is None:
            self._key = min(self.members)
        return self._key

    @property
    def shared(self) -> bool:
        return len(self.members) > 1


class ActivationCapture:
    """Captures per-layer inputs (and optionally output gradients) and forms
    the AK-MOMENTUM correction for every registered Linear / Conv1d / Conv2d.

    Typical use is indirect: :class:`~ak_momentum.ak_adamw.AKAdamW` and
    :class:`~ak_momentum.ak_sgd.AKSGD` build and bind one of these for you.

    Args:
        factored: ``False`` (default) uses ``param.grad`` as the gradient term
            and is DDP-safe; ``True`` captures ``dY`` and uses ``dY^T X_hat``.
        chunk_size: floor for the through-N token chunk, and the fixed chunk of
            the exact-fp32 path.  Bounds transient memory at
            ``O(fan_out * chunk_size)``.
        chunk_budget_frac: fraction of *total* device memory the through-N
            intermediate may occupy on the bf16 path, which sets the chunk (see
            `_select_chunk`).  Total rather than free, so the choice is
            deterministic for a given config.
        shape_adaptive: choose the contraction order per layer (default).  When
            ``False`` every layer uses the Gram order.
        bf16_gram: run the correction GEMMs with bf16 inputs and fp32
            accumulation.  Off by default, so the correction is exact fp32.
        share_sigma: share one Gram accumulator between layers that consume the
            same input tensor (default).  Numerically inert.
    """

    def __init__(
        self,
        factored: bool = False,
        chunk_size: int = 4096,
        chunk_budget_frac: float = 0.02,
        shape_adaptive: bool = True,
        bf16_gram: bool = False,
        share_sigma: bool = True,
    ) -> None:
        self._factored = bool(factored)
        self._chunk_size = int(chunk_size)
        self._shape_adaptive = bool(shape_adaptive)
        self._share_sigma = bool(share_sigma)
        # Resolved lazily on first use, so construction never touches the
        # device, then cached so the rule is fixed for this capture's lifetime.
        self._chunk_budget_frac = float(chunk_budget_frac)
        self._chunk_budget_bytes: Optional[int] = None

        self._bf16_gram = False
        if bf16_gram:
            if _bf16_gram_selfcheck():
                self._bf16_gram = True
            else:
                import sys
                print(
                    "[ak_momentum] bf16 correction self-check failed; "
                    "falling back to the exact fp32 path.",
                    file=sys.stderr, flush=True,
                )

        # Per-module metadata (type, weight shape, contraction order).
        self._module_info: Dict[str, dict] = {}

        # Live captures for the current micro-batch.
        self._inputs: Dict[str, List[torch.Tensor]] = {}
        self._output_grads: Dict[str, List[torch.Tensor]] = {}

        # Parameter-sized accumulators; these survive across micro-batches so
        # gradient accumulation costs O(fan_out * fan_in), not O(K * N * fan_in).
        #   Gram order      : [n, n]  holding  sum_k X_hat_k^T X_hat_k
        #   through-N order : [m, n]  holding  sum_k (M X_hat_k^T) X_hat_k
        self._correction_accum: Dict[str, torch.Tensor] = {}
        self._grad_term_accum: Dict[str, torch.Tensor] = {}
        self._N_accum: Dict[str, int] = {}

        # Sigma-sharing bookkeeping.
        self._shared_x = WeakTensorKeyDictionary()
        self._share_tok: Dict[str, List[Optional[_ShareGroup]]] = {}
        self._pending_tok: Dict[str, Optional[_ShareGroup]] = {}

        # Hook handles.
        self._hooks: List = []
        self._pre_hooks: List = []

        # Bound optimizer, resolved lazily so device moves are safe.
        self._bound_optimizer = None
        self._momentum_key: str = "momentum_buffer"
        self._name_to_param: Dict[str, nn.Parameter] = {}

        # Counters, reset by clear(); useful for tests and cost accounting.
        self.n_gram_accum = 0
        self.n_gram_apply = 0
        self.n_gram_shared = 0
        self.n_through_n_accum = 0
        self.n_through_n_apply = 0

    # ------------------------------------------------------------------
    # Registration and binding
    # ------------------------------------------------------------------

    @property
    def factored(self) -> bool:
        return self._factored

    def register(self, model: nn.Module, target_modules: Optional[List[str]] = None) -> None:
        """Install hooks on every Linear / Conv1d / Conv2d (or `target_modules`)."""
        for name, module in model.named_modules():
            if target_modules is not None and name not in target_modules:
                continue
            if not isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                continue

            if isinstance(module, nn.Conv2d):
                info = dict(
                    type="conv2d", kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding,
                    dilation=module.dilation, groups=module.groups,
                    weight_shape=tuple(module.weight.shape),
                )
            elif isinstance(module, nn.Conv1d):
                info = dict(
                    type="conv1d", kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding,
                    dilation=module.dilation, groups=module.groups,
                    weight_shape=tuple(module.weight.shape),
                )
            else:
                info = dict(type="linear", weight_shape=tuple(module.weight.shape))

            # The contraction order is a property of the shape, so fix it once.
            info["order"] = self._dispatch_order(*self._weight_mn(info))
            self._module_info[name] = info

            # The share probe must be a *pre*-hook: register_full_backward_hook
            # rewrites a module's inputs, so by the time the ordinary forward
            # hook runs, the identity that defines "same input" is gone.
            if self._share_sigma and self._is_share_eligible(name):
                self._pre_hooks.append(
                    module.register_forward_pre_hook(self._make_forward_pre_hook(name))
                )

            self._hooks.append(module.register_forward_hook(self._make_forward_hook(name)))
            self._hooks.append(
                module.register_full_backward_hook(self._make_backward_hook(name))
            )

    def bind_optimizer(self, optimizer, momentum_key: str = "momentum_buffer") -> None:
        """Bind the optimizer whose first-moment buffers this capture corrects.

        Buffers are resolved lazily at hook time (never cached as tensors), so
        binding before or after a device move is equally safe.

        Args:
            momentum_key: ``'momentum_buffer'`` for AK-SGD, ``'exp_avg'`` for
                AK-AdamW.
        """
        self._bound_optimizer = optimizer
        self._momentum_key = momentum_key
        self._name_to_param = {}
        param_to_name = getattr(optimizer, "_param_to_name", {})
        for group in optimizer.param_groups:
            for p in group["params"]:
                name = param_to_name.get(id(p))
                if name is not None and name in self._module_info:
                    expected = self._module_info[name].get("weight_shape")
                    if expected is not None and tuple(p.shape) != expected:
                        raise RuntimeError(
                            f"[ak_momentum] '{name}' maps to a parameter of shape "
                            f"{tuple(p.shape)} but the module's weight is {expected}."
                        )
                    self._name_to_param[name] = p

    def _resolve_momentum(self, name: str) -> Optional[torch.Tensor]:
        """Current first-moment buffer for `name`, or None if not corrected."""
        if self._bound_optimizer is None:
            return None
        p = self._name_to_param.get(name)
        if p is None:
            return None
        state = self._bound_optimizer.state.get(p, {})
        if self._momentum_key not in state:
            if self._momentum_key == "exp_avg":
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            else:
                state[self._momentum_key] = torch.zeros_like(p)
            self._bound_optimizer.state[p] = state
        return state[self._momentum_key]

    # ------------------------------------------------------------------
    # Shapes and contraction order
    # ------------------------------------------------------------------

    @staticmethod
    def _weight_mn(info: dict) -> Tuple[int, int]:
        """(fan_out, fan_in) for the correction; conv fan_in is the patch size."""
        ws = info["weight_shape"]
        n = 1
        for s in ws[1:]:
            n *= s
        return ws[0], n

    def _dispatch_order(self, m: int, n: int) -> str:
        """Gram costs ~2 n^2 N, through-N ~4 m n N, so Gram wins iff n < 2m."""
        if not self._shape_adaptive:
            return "gram"
        return "gram" if n < 2 * m else "through_n"

    def _correction_order(self, name: str) -> str:
        return self._module_info.get(name, {}).get("order", "gram")

    # ------------------------------------------------------------------
    # Correction GEMMs
    # ------------------------------------------------------------------

    def _gram_gemm(self, x_hat: torch.Tensor) -> torch.Tensor:
        """One micro-batch's Gram ``X_hat^T X_hat``."""
        if self._bf16_gram:
            xb = x_hat.to(torch.bfloat16)
            return (xb.transpose(0, 1) @ xb).to(torch.float32)
        return x_hat.transpose(0, 1) @ x_hat

    def _grad_gemm(self, dy_2d: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Factored gradient term ``dY^T X_hat`` for one micro-batch."""
        if self._bf16_gram:
            return (
                dy_2d.to(torch.bfloat16).transpose(0, 1) @ x_hat.to(torch.bfloat16)
            ).to(torch.float32)
        return dy_2d.to(x_hat.dtype).transpose(0, 1) @ x_hat

    def _apply_gram(self, m_2d: torch.Tensor, sigma_hat: torch.Tensor) -> torch.Tensor:
        """The once-per-step apply ``M @ Sigma_hat``."""
        if self._bf16_gram:
            return (m_2d.to(torch.bfloat16) @ sigma_hat.to(torch.bfloat16)).to(m_2d.dtype)
        return m_2d @ sigma_hat

    def _select_chunk(
        self, m: int, n_tokens: int, itemsize: int, device: torch.device,
    ) -> int:
        """Size the through-N token chunk from a memory budget.

        What has to fit is the intermediate ``M X_c^T``, shaped ``[m, chunk]``,
        which the first GEMM materializes before the second consumes it.  If the
        whole token dimension fits the budget, one pair of GEMMs is issued;
        otherwise this falls back to the largest power-of-two chunk that does,
        never below `chunk_size`, so the choice can only improve on the fixed
        chunk.  The budget is a fraction of *total* device memory, so it does
        not move with allocator state.  Off-device (or if the query fails) the
        budget is zero and the fixed chunk is used.
        """
        if self._chunk_budget_bytes is None:
            try:
                total = torch.cuda.get_device_properties(device).total_memory
            except Exception:
                total = 0
            self._chunk_budget_bytes = int(total * self._chunk_budget_frac)
        budget = self._chunk_budget_bytes
        floor = max(1, int(self._chunk_size))
        row = max(1, m * itemsize)
        if budget <= 0:
            return floor
        if n_tokens * row <= budget:
            return n_tokens                       # unchunked: one GEMM pair
        fit = budget // row
        if fit < floor:
            return floor
        return 1 << (int(fit).bit_length() - 1)   # largest power of two that fits

    def _through_n_accum_into(
        self, dest: torch.Tensor, m_2d: torch.Tensor, x_hat: torch.Tensor,
    ) -> None:
        """Accumulate ``sum_k (M X_hat_k^T) X_hat_k`` into `dest`, chunked over N."""
        if self._bf16_gram:
            xb = x_hat.to(torch.bfloat16)
            mb = m_2d.to(torch.bfloat16)
            chunk = self._select_chunk(
                m_2d.shape[0], xb.size(0), xb.element_size(), xb.device,
            )
            for start in range(0, xb.size(0), chunk):
                xc = xb[start:start + chunk]
                dest.add_((mb @ xc.transpose(0, 1) @ xc).to(dest.dtype))
        else:
            # The exact-fp32 escape keeps the fixed chunk: it is not the
            # deployed path, so it is left on predictable, unbudgeted memory.
            xf = x_hat.to(m_2d.dtype)
            chunk = self._chunk_size
            for start in range(0, xf.size(0), chunk):
                xc = xf[start:start + chunk]
                dest.addmm_(m_2d @ xc.transpose(0, 1), xc)

    # ------------------------------------------------------------------
    # Sigma sharing
    # ------------------------------------------------------------------

    def _is_share_eligible(self, name: str) -> bool:
        """Only Gram-order Linear layers can share Sigma formation.

        Conv layers unfold their input with per-module kernel/stride/padding, so
        a shared input does not imply a shared x_hat; the through-N accumulator
        depends on M and is per layer by definition.
        """
        info = self._module_info.get(name)
        if info is None or info.get("type") != "linear":
            return False
        return info.get("order", "gram") != "through_n"

    def _note_shared_input(self, name: str, x: torch.Tensor) -> Optional[_ShareGroup]:
        """Record that `name`'s forward consumed `x`; return that tensor's group."""
        if not self._share_sigma or not self._is_share_eligible(name):
            return None
        tok = self._shared_x.get(x)
        if tok is None:
            tok = _ShareGroup()
            self._shared_x[x] = tok
        if name not in tok.members:
            tok.members.append(name)
        return tok

    def _shared_gram_buffer(
        self,
        tok: Optional[_ShareGroup],
        name: str,
        c_in: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, bool]:
        """Resolve the Gram accumulator and say whether this layer must fill it.

        For a shared group every member's entry aliases one tensor, so all
        by-name readers are unchanged; the first member to reach backward in a
        micro-batch runs the GEMM and the rest reuse the identical result.
        """
        gkey = tok.key if (tok is not None and tok.shared) else name
        buf = self._correction_accum.get(gkey)
        if buf is None:
            buf = torch.zeros(c_in, c_in, dtype=dtype, device=device)
            self._correction_accum[gkey] = buf
        elif tuple(buf.shape) != (c_in, c_in):
            raise RuntimeError(
                f"[ak_momentum] Sigma-sharing key collision for '{name}' "
                f"(group '{gkey}'): accumulator is {tuple(buf.shape)}, "
                f"this layer needs ({c_in}, {c_in})."
            )
        if gkey != name:
            self._correction_accum[name] = buf
        if tok is not None and tok.shared:
            if tok.gram_done:
                self.n_gram_shared += 1
                return buf, False
            tok.gram_done = True
        return buf, True

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _make_forward_pre_hook(self, name: str):
        def pre_hook(module, args):
            if not module.training or not args:
                return
            # Same activation-checkpoint guard as the forward hook, so a
            # recompute replay never mints a group nothing consumes.
            if self._inputs.get(name):
                return
            self._pending_tok[name] = self._note_shared_input(name, args[0])
        return pre_hook

    def _make_forward_hook(self, name: str):
        def hook(module, inputs, output):
            if not module.training:
                return  # eval: no backward follows
            # Activation-checkpoint guard: a captured module's forward can only
            # re-fire before its backward when checkpointing replays it during
            # backward.  The replayed input is identical, so keep the original
            # and drop the replay -- otherwise N and the gradient term double.
            if self._inputs.get(name):
                return
            self._inputs.setdefault(name, []).append(inputs[0].detach())
            self._share_tok.setdefault(name, []).append(self._pending_tok.pop(name, None))
        return hook

    def _make_backward_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            if not module.training or name not in self._inputs:
                return
            m = self._resolve_momentum(name)
            raw_x_list = self._inputs.pop(name)
            tok_list = self._share_tok.pop(name, [])
            if m is None:
                return  # this module's weight is not AK-corrected
            info = self._module_info.get(name, dict(type="linear"))
            raw_dy = grad_output[0].detach() if self._factored else None
            for i, raw_x in enumerate(raw_x_list):
                self._accumulate(
                    name, info, m, raw_x, raw_dy,
                    tok_list[i] if i < len(tok_list) else None,
                )
        return hook

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def flush_microbatch(self, name_to_momentum: Dict[str, torch.Tensor]) -> None:
        """Fold any still-live captures into the accumulators and free them.

        The backward hooks already do this when an optimizer is bound, so this
        is a safety net for the tail of a step.
        """
        for name in list(self._inputs.keys()):
            raw_x_list = self._inputs.get(name, [])
            if not raw_x_list:
                continue
            m = name_to_momentum.get(name)
            if m is None:
                continue
            info = self._module_info.get(name, dict(type="linear"))
            dy_list = self._output_grads.get(name, [])
            tok_list = self._share_tok.get(name, [])
            for i, raw_x in enumerate(raw_x_list):
                self._accumulate(
                    name, info, m, raw_x,
                    dy_list[i] if i < len(dy_list) else None,
                    tok_list[i] if i < len(tok_list) else None,
                )
        self._inputs.clear()
        self._output_grads.clear()
        self._share_tok.clear()

    def _accumulate(
        self,
        name: str,
        info: dict,
        m: torch.Tensor,
        raw_x: torch.Tensor,
        raw_dy: Optional[torch.Tensor],
        tok: Optional[_ShareGroup] = None,
    ) -> None:
        if info["type"] == "linear":
            self._accumulate_linear(name, m, raw_x, raw_dy, tok)
        else:
            self._accumulate_conv(name, info, m, raw_x, raw_dy)

    def _accumulate_linear(
        self,
        name: str,
        m: torch.Tensor,
        raw_x: torch.Tensor,
        raw_dy: Optional[torch.Tensor],
        tok: Optional[_ShareGroup],
    ) -> None:
        x_2d = raw_x.reshape(-1, raw_x.size(-1))                 # [N_k, n]
        n_k = x_2d.size(0)
        x_hat = normalize_rows(x_2d)                             # native dtype
        c_in = x_hat.size(1)

        m_2d = m.view(m.size(0), -1) if m.ndim > 2 else m
        dtype = m_2d.dtype
        if m_2d.shape[1] != c_in:
            raise RuntimeError(
                f"[ak_momentum] shape mismatch for '{name}': buffer expects "
                f"fan_in={m_2d.shape[1]} but the captured activation has {c_in}."
            )

        if self._correction_order(name) == "through_n":
            if name not in self._correction_accum:
                self._correction_accum[name] = torch.zeros_like(m_2d)
            self._through_n_accum_into(self._correction_accum[name], m_2d, x_hat)
            self.n_through_n_accum += 1
        else:
            buf, run = self._shared_gram_buffer(tok, name, c_in, dtype, x_hat.device)
            if run:
                buf.add_(self._gram_gemm(x_hat).to(dtype))
                self.n_gram_accum += 1

        self._N_accum[name] = self._N_accum.get(name, 0) + n_k

        if raw_dy is not None:
            dy_2d = raw_dy.reshape(-1, raw_dy.size(-1))
            if name not in self._grad_term_accum:
                self._grad_term_accum[name] = torch.zeros(
                    dy_2d.size(1), c_in, dtype=dtype, device=x_hat.device,
                )
            self._grad_term_accum[name].add_(self._grad_gemm(dy_2d, x_hat).to(dtype))

    def _unfold(self, info: dict, raw_x: torch.Tensor, d_patch: int) -> torch.Tensor:
        """Explicit unfold, so each patch vector can be normalized individually."""
        if info["type"] == "conv2d":
            x_unf = F.unfold(
                raw_x, kernel_size=info["kernel_size"], stride=info["stride"],
                padding=info["padding"], dilation=info["dilation"],
            )
        else:
            kw = info["kernel_size"][0]
            x_unf = F.unfold(
                raw_x.unsqueeze(-1), kernel_size=(kw, 1),
                stride=(info["stride"][0], 1), padding=(info["padding"][0], 0),
                dilation=(info["dilation"][0], 1),
            )
        return x_unf.permute(0, 2, 1).reshape(-1, d_patch)

    def _accumulate_conv(
        self,
        name: str,
        info: dict,
        m: torch.Tensor,
        raw_x: torch.Tensor,
        raw_dy: Optional[torch.Tensor],
    ) -> None:
        c_out, d_patch = self._weight_mn(info)
        x_2d = self._unfold(info, raw_x, d_patch)
        n_k = x_2d.size(0)
        x_hat = normalize_rows(x_2d)

        m_2d = m.view(c_out, d_patch)
        dtype = m_2d.dtype

        if self._correction_order(name) == "through_n":
            if name not in self._correction_accum:
                self._correction_accum[name] = torch.zeros_like(m_2d)
            self._through_n_accum_into(self._correction_accum[name], m_2d, x_hat)
            self.n_through_n_accum += 1
        else:
            if name not in self._correction_accum:
                self._correction_accum[name] = torch.zeros(
                    d_patch, d_patch, dtype=dtype, device=x_hat.device,
                )
            self._correction_accum[name].add_(self._gram_gemm(x_hat).to(dtype))
            self.n_gram_accum += 1

        self._N_accum[name] = self._N_accum.get(name, 0) + n_k

        if raw_dy is not None:
            if info["type"] == "conv2d":
                dy_2d = raw_dy.permute(0, 2, 3, 1).reshape(-1, c_out)
            else:
                dy_2d = raw_dy.permute(0, 2, 1).reshape(-1, c_out)
            if name not in self._grad_term_accum:
                self._grad_term_accum[name] = torch.zeros(
                    c_out, d_patch, dtype=dtype, device=x_hat.device,
                )
            self._grad_term_accum[name].add_(
                self._grad_gemm(dy_2d.to(x_hat.dtype), x_hat).to(dtype)
            )

    # ------------------------------------------------------------------
    # Step boundary
    # ------------------------------------------------------------------

    def sync_accumulators(self) -> None:
        """All-reduce the correction inputs so every rank forms the same update.

        Under DDP each rank sees different data, so Sigma_hat and the factored
        gradient term are rank-local.  Summing them (and the token counts) makes
        the correction the global one, keeping optimizer state identical across
        ranks.  No-op outside distributed runs.
        """
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        seen = set()
        for buf in self._correction_accum.values():
            # Shared groups alias one tensor; reducing it once per member would
            # multiply it by the number of members.
            if id(buf) in seen:
                continue
            seen.add(id(buf))
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        for buf in self._grad_term_accum.values():
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        self._reduce_token_counts()

    def _reduce_token_counts(self) -> None:
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        if not self._N_accum:
            return
        names = sorted(self._N_accum)
        device = (
            next(iter(self._correction_accum.values())).device
            if self._correction_accum else torch.device("cpu")
        )
        counts = torch.tensor([self._N_accum[n] for n in names], dtype=torch.long, device=device)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        for i, n in enumerate(names):
            self._N_accum[n] = int(counts[i].item())

    def compute_delta_update(
        self, name: str, m: torch.Tensor, grad: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Return ``G - M Sigma_hat`` for `name`, or None if nothing was captured.

        Args:
            m: the layer's current first-moment buffer.
            grad: the weight gradient, used as G on the non-factored path.
        """
        n_total = self._N_accum.get(name, 0)
        if n_total == 0 or name not in self._correction_accum:
            return None

        m_2d = m.view(m.size(0), -1) if m.ndim > 2 else m
        if self._correction_order(name) == "through_n":
            # The accumulator already holds M (N Sigma_hat); only /N remains.
            correction = self._correction_accum[name] / n_total
            self.n_through_n_apply += 1
        else:
            sigma_hat = self._correction_accum[name] / n_total
            correction = self._apply_gram(m_2d, sigma_hat)
            self.n_gram_apply += 1

        if name in self._grad_term_accum:
            result = self._grad_term_accum[name] - correction      # factored
        elif grad is not None:
            result = grad.view(m_2d.shape) - correction            # non-factored
        else:
            return None
        return result.view(m.shape)

    def clear(self) -> None:
        """Drop everything belonging to the finished optimizer step."""
        self._inputs.clear()
        self._output_grads.clear()
        self._correction_accum.clear()
        self._grad_term_accum.clear()
        self._N_accum.clear()
        # Share tokens live exactly as long as the activations that minted them;
        # `_shared_x` is weak-keyed, so dropping these retires last step's groups.
        self._share_tok.clear()
        self._pending_tok.clear()
        self.n_gram_accum = 0
        self.n_gram_apply = 0
        self.n_gram_shared = 0
        self.n_through_n_accum = 0
        self.n_through_n_apply = 0

    def remove_hooks(self) -> None:
        """Remove every hook this capture installed."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for h in self._pre_hooks:
            h.remove()
        self._pre_hooks.clear()
