# AK-Momentum

**Activation-Keyed Momentum: An Anisotropic Momentum Update via the Delta Rule**

Most optimizers form their momentum as an exponential moving average of past
gradients, forgetting every direction at one fixed rate. But the inputs a
network sees are highly anisotropic: a few directions are queried constantly
while most are seen rarely. AK-MOMENTUM builds that asymmetry into the momentum
update itself, by keying the buffer on the layer's input activation and updating
it with the classical delta rule:

```
M_t = β M_{t-1} + η (G_t − M_{t-1} Σ̂_t),     Σ̂ = X̂ᵀX̂ / N,   x̂_i = x_i / ‖x_i‖
```

Each direction is then forgotten at a rate set by how often it appears
(`β − η λ_i` along eigendirection `i`), instead of the single rate `β`
everywhere. It is a drop-in replacement for the first-moment buffer of any
optimizer: the second moment, weight decay and any preconditioning are
untouched, so it composes with Muon, Shampoo or SOAP rather than competing.

## Install

```bash
pip install git+https://github.com/99versatile/ak-momentum.git
# or, from a clone
pip install -e .
```

The package depends on PyTorch only. The reproduction scripts need a few more
packages (`pip install -r requirements.txt`).

## Quick start

```python
from ak_momentum import configure_ak_adamw

model = MyModel()
optimizer, capture = configure_ak_adamw(
    model, lr=3e-4, betas=(0.99, 0.99), weight_decay=0.1, eta=0.37,
)

for batch in loader:
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()       # consumes the activations captured this step
    optimizer.zero_grad()
```

`configure_ak_adamw` installs forward/backward hooks on every `Linear`, `Conv1d`
and `Conv2d`, so the optimizer needs a reference to the model — that is the one
way it differs from a plain `torch.optim` constructor. Keep `capture` alive for
the run; call `capture.remove_hooks()` to detach.

`AKSGD` / `configure_ak_sgd` do the same for SGD-with-momentum.

### Recommended settings

The marginal tuning cost over EMA momentum is **one coefficient**.

| Setting | Value | Why |
|---|---|---|
| `eta` | 0.4 (anything in [0.3, 0.5]) | Final loss moves ≤ 0.015 nats across η ∈ [0.10, 0.99] |
| `betas[0]` | 0.99 | The per-direction coefficient is `β − η λ_i`, so a larger nominal β restores the effective horizon |
| `betas[1]`, `weight_decay` | your AdamW values | Unchanged by the method |
| `lr` | ≈ 1/10 of your tuned AdamW lr, then tune | The buffer is a Wiener-type predictor, not an average of gradients |

Both `eta` and `beta` are width-invariant under μP, so the whole recipe can be
tuned once at a small proxy width and transferred. Hidden-weight learning rates
follow the base optimizer's own μP rule, unchanged; `ak_momentum.groups` has
helpers for that.

## How it works

* **Key normalization.** Keys are normalized per sample, so `tr Σ̂ = 1` and
  `λ_max(Σ̂) ≤ 1`. Contraction along every direction stays in `[β − η, β]` for
  any data, which is what makes η both stable and width-invariant.
* **Contraction order.** `M Σ̂` is evaluated in whichever order is cheaper for
  the layer's shape: the Gram order `M (X̂ᵀX̂)` when `n < 2m`, the through-N
  order `(M X̂ᵀ) X̂` otherwise. Both are exact.
* **Σ̂ sharing.** Layers reading the same input tensor (e.g. the two gated-MLP
  projections) share one Gram matrix. Identity is re-derived each micro-batch,
  so it cannot go stale, and the per-layer multiply `M @ Σ̂` stays per layer.
* **Cost.** The correction costs `(6+4γ)/(24+18γ)` of a transformer block's
  linear forward+backward — between 22.2% and 25.0% for any gated-MLP ratio γ,
  23.1% at the Llama-2 ratio. The realized per-step overhead is lower, because
  attention, embeddings and the output head carry no correction and dilute it,
  and it rises with width as the linear layers take a larger share of the step
  (measured figures below). **Persistent optimizer-state overhead is zero**: no
  buffer beyond the one the base optimizer already keeps.
* **Two performance levers, both on by default.** The through-N token loop is
  chunked from a memory budget rather than a fixed width, so at production
  shapes it issues a single pair of GEMMs; and the row normalization is fused
  through `torch.compile` on CUDA, which is both faster and slightly more
  accurate than the eager pair (inductor reduces in fp32). Both fall back
  safely: off-device or on any compile failure the eager reference runs.
  `set_fused_norm(False)` pins the eager path.

### Measured cost

AK-AdamW against AdamW on the paper's language models, one optimizer step of
524,288 tokens, single NVIDIA H200, bf16.

**Wall-clock per step**, measured in the real training loop:

| Scale | Micro-batch | AdamW | AK-AdamW | Overhead |
|---|---|---|---|---|
| 67M | 32×8 both | 2.5104 s | 2.8935 s | **15.26%** |
| 370M | 32×8 both | 6.5566 s | 7.7205 s | **17.75%** |
| 1B | 8×32 both | 10.4213 s | 12.3431 s | **18.44%** |

**FLOPs per step**, counted under the full-attention convention:

| Scale | AdamW | AK-AdamW | Overhead |
|---|---|---|---|
| 67M | 310,870,806,626,304 | 341,968,517,332,992 | **10.003%** |
| 370M | 1,423,266,262,548,480 | 1,644,439,898,423,296 | **15.540%** |
| 1B | 3,738,373,894,176,768 | 4,415,673,056,886,784 | **18.117%** |

Both overheads rise with width, as the analysis predicts, and stay below the
22.2–25.0% per-block band. The wall-clock figures use a non-fused
implementation of the correction.

### The gradient term: `factored`

| `factored` | Gradient term `G` | Use when |
|---|---|---|
| `False` (default) | `param.grad`, the ordinary weight gradient | Default; DDP-safe; used for the paper's language-model runs |
| `True` | `dY^T X̂`, formed against normalized keys | Single-device; the literal batch form; used for the paper's CIFAR-10 MLP run |

Both are exact instances of the rule and differ only in whether the gradient
term uses raw or normalized keys. The factored path captures `dY`, which is
rank-local, so it must not be combined with DDP's all-reduced gradient.

### Distributed training

`AKAdamW` and `AKSGD` are replicated, not sharded: every rank computes the same
update. `ActivationCapture.sync_accumulators()` (called for you in `step()`)
all-reduces Σ̂, the token counts and — on the factored path — the gradient term,
so optimizer state stays identical across ranks. Use `factored=False` under DDP.

Gradient accumulation is supported: activations are folded into
parameter-sized accumulators at each backward, so memory stays `O(fan_out ×
fan_in)` per layer regardless of the number of micro-batches.

## What's here

```
ak_momentum/        the method: capture + optimizers + μP helpers
  capture.py        hooks, key normalization, Σ̂ formation, the correction
  ak_adamw.py       AKAdamW  — AdamW with a delta-rule first moment
  ak_sgd.py         AKSGD    — SGD-momentum with a delta-rule buffer
  groups.py         parameter grouping and μP learning-rate rules
reproduce/          scripts to regenerate the paper's experiments
  lm/               FineWeb-Edu pre-training at 67M / 370M / 1B
  vision/           CIFAR-10: MLP, ResNet-18, ViT-Tiny
```

See [reproduce/README.md](reproduce/README.md) for exact commands and the full
hyperparameter tables.

## Results

AK-AdamW against AdamW on FineWeb-Edu, over matched validation-loss levels:

| Scale | Mean loss gap (nats) | Mean step reduction | Max step reduction |
|---|---|---|---|
| 67M (3 runs, 10BT) | 0.0936 ± 0.0150 | 39.25 ± 4.55% | 46.39 ± 4.32% |
| 370M (3 runs, 10BT) | 0.0383 ± 0.0028 | 17.61 ± 0.95% | 22.12 ± 0.80% |
| 1B (1 run, 20BT) | 0.0245 | 13.43% | 19.37% |

Every hyperparameter at 370M and 1B is transferred from the 67M sweep with no
re-tuning. The gain also holds for AK-SGD against SGD-momentum on a CIFAR-10
MLP (+5.55% final test accuracy), and for AK-AdamW against AdamW on ResNet-18
and ViT-Tiny.

## Citation

```bibtex
@article{hong2026akmomentum,
  title   = {Activation-Keyed Momentum: An Anisotropic Momentum Update via the Delta Rule},
  author  = {Hong, Euijin and Qu, Guannan},
  journal = {arXiv preprint arXiv:2608.19491},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
