# Reproducing the paper's experiments

Everything here is a thin layer over the `ak_momentum` package: the model, the
data pipeline and the training loop are shared by both arms of every
comparison, so the only difference between an arm and its baseline is the
optimizer (and its separately tuned learning rate).

```bash
pip install -e .                  # the package
pip install -r requirements.txt   # accelerate, datasets, torchvision, PyYAML
```

Run every command from the repository root.

---

## 1. Language models — FineWeb-Edu

### Data

The trainer expects a Hugging Face dataset saved with `save_to_disk()` holding
an `input_ids` column with at least `seq_length = 2048` tokens per row. The
paper used the FineWeb-Edu 10BT subset (20BT for the 1B run), tokenized with the
Llama-2 SentencePiece tokenizer (vocabulary 32,000), EOS between documents.
Neither the corpus nor the tokenizer is redistributed here.

```text
data/fineweb-edu/fineweb_edu_10bt_tokenized_2048/
    dataset_info.json  state.json  data-*.arrow          # a saved dataset
    ── or ──
    shard_0000_tokenized/  shard_0001_tokenized/  ...     # shards
```

A validation split of 5.71e-5 (~280 sequences) is carved out with a fixed seed
of 42, independent of the training seed, so both arms evaluate on identical rows.

### Runs

```bash
# 67M proxy — where every hyperparameter was tuned
accelerate launch --num_processes 1 --mixed_precision bf16 \
    reproduce/lm/train.py --config reproduce/lm/configs/67m_adamw.yaml
accelerate launch --num_processes 1 --mixed_precision bf16 \
    reproduce/lm/train.py --config reproduce/lm/configs/67m_ak_adamw.yaml

# 370M and 1B — hyperparameters transferred, nothing re-tuned
accelerate launch --num_processes 1 --mixed_precision bf16 \
    reproduce/lm/train.py --config reproduce/lm/configs/370m_ak_adamw.yaml
accelerate launch --num_processes 1 --mixed_precision bf16 \
    reproduce/lm/train.py --config reproduce/lm/configs/1b_ak_adamw.yaml
```

Repeat each with `--seed 42`, `--seed 777`, `--seed 1234` for the three-run
bands. A smoke test that needs no corpus:

```bash
python reproduce/lm/train.py --config reproduce/lm/configs/67m_ak_adamw.yaml \
    --synthetic --max_steps 3
```

For a short run that follows the opening segment of the full schedule, pass
`--stop_after N`. `--max_steps` also sets the cosine horizon, so shortening
it would decay the learning rate early.

### Hyperparameters

Selected on the 67M proxy with Optuna (TPE + successive halving), then
transferred by the μP rule `lr_target = lr_proxy × (384 / d_model)`. The configs
carry the **proxy-width base rate**; the trainer applies the multiplier, so the
effective rates are:

| Optimizer | lr @67M | lr @370M | lr @1B | aux lr @67M | β₁ | β₂ | η | wd |
|---|---|---|---|---|---|---|---|---|
| AdamW | 4.0e-3 | 1.5e-3 | 8.6e-4 | — | 0.95 | 0.99 | — | 0.1 |
| AK-AdamW | 3.0e-4 | 1.13e-4 | 6.4e-5 | 1.24e-2 | 0.99 | 0.99 | 0.37 | 0.1 |

η and β₁ do not rescale with width. The auxiliary rate (embeddings, 1-D
parameters, and — scaled by `384/d` — the output head) is tuned as its own axis
for AK-AdamW, because tying it to a learning rate ~13× smaller starves those
parameters. AdamW drives all three groups from one rate, which is what its own
μP rule expects.

Shared: sequence length 2048, effective batch 256 sequences (524,288 tokens per
step), gradient clipping 1.0, cosine schedule with 500 warmup steps decaying to
0.1× the peak, bf16 mixed precision, no dropout. 19,100 steps at 67M/370M
(10BT); 38,200 at 1B (20BT, Chinchilla-optimal).

Architecture: Llama-2-style decoder, `n_layers = 24` and `d_head = 64` fixed at
every scale, SwiGLU with expansion 2.67, RoPE base 10,000, pre-RMSNorm, no
biases, untied embeddings. Widths 384 / 1024 / 1792.

Both arms at a given scale use the same micro-batching (32×8 at 67M and 370M,
8×32 at 1B), so their step times compare directly. The original 370M AdamW run
used 16×16; micro-batching does not change the update, only its speed.

Cost on a single H200, from the measured step times and excluding evaluation:

| Scale | Steps | AdamW | AK-AdamW |
|---|---|---|---|
| 67M | 19,100 | 2.5104 s/step, ≈13.3 h | 2.8935 s/step, ≈15.4 h |
| 370M | 19,100 | 6.5566 s/step, ≈34.8 h | 7.7205 s/step, ≈41.0 h |
| 1B | 38,200 | 10.4213 s/step, ≈110.6 h | 12.3431 s/step, ≈131.0 h |

---

## 2. CIFAR-10

TorchVision downloads CIFAR-10 on first use. All runs: 120 epochs, batch 256,
cosine schedule with 5 warmup epochs decaying to a minimum lr of 1e-6 (1e-5
for the MLP, set explicitly below), seeds 42 / 777 / 1234.

### Optimizer family check — AK-SGD vs SGD-momentum (3-layer MLP)

MLP of 4096-4096-2048 units with LayerNorm, GELU and dropout 0.5. Weight decay
1e-4, min lr 1e-5, Nesterov on.

```bash
python reproduce/vision/train.py --model mlp --optimizer sgd \
    --lr 0.1 --momentum 0.9 --weight-decay 1e-4 --min-lr 1e-5 --seed 42

python reproduce/vision/train.py --model mlp --optimizer ak_sgd \
    --lr 1.7 --momentum 0.99 --eta 0.99 --aux-lr 3e-4 --factored \
    --weight-decay 1e-4 --min-lr 1e-5 --seed 42
```

Expected final test accuracy: **61.51%** (SGD) vs **67.06%** (AK-SGD).

Note the asymmetry, which is part of the published setup: AK-SGD drives the
hidden matrices while the head and 1-D parameters go to an auxiliary AdamW at
`--aux-lr`, whereas the SGD baseline trains everything with SGD. This run also
uses `--factored`.

### Architecture check — AK-AdamW vs AdamW (ResNet-18, ViT-Tiny)

Weight decay 5e-4, β₂ 0.999, η 0.37, non-factored, a single parameter group for
both arms.

```bash
# ResNet-18
python reproduce/vision/train.py --model resnet18 --optimizer adamw \
    --lr 5e-3 --momentum 0.95 --seed 777
python reproduce/vision/train.py --model resnet18 --optimizer ak_adamw \
    --lr 1e-4 --momentum 0.99 --eta 0.37 --seed 777

# ViT-Tiny
python reproduce/vision/train.py --model vit_tiny --optimizer adamw \
    --lr 1e-3 --momentum 0.95 --seed 777
python reproduce/vision/train.py --model vit_tiny --optimizer ak_adamw \
    --lr 5e-5 --momentum 0.99 --eta 0.37 --seed 777
```

Read ResNet-18 on validation loss and ViT-Tiny on training loss: both saturate
on CIFAR-10 (≈94% and ≈84% test accuracy under either optimizer), so the
measurable quantity is convergence speed, and ViT-Tiny's validation curve is
dominated by the architecture's generalization gap at this dataset size rather
than by the optimizer.

---

## 3. The Muon baseline

The paper's Muon reference point is not shipped here, since it is a baseline
rather than a contribution. To reproduce it, use the reference implementation at
<https://github.com/KellerJordan/Muon> with the settings the paper's protocol
fixed:

* Newton–Schulz, 5 steps, Nesterov on, momentum 0.95, with the
  `sqrt(max(1, fan_out/fan_in))` shape adjustment.
* Hidden-matrix lr 8.0e-3 at 67M, transferred as `lr × sqrt(384/d_model)`
  (Muon's own μP rule), giving 4.90e-3 at 370M.
* Auxiliary AdamW for embeddings, the head and 1-D parameters at 1.28e-2
  (with the head additionally scaled by `384/d_model`), betas (0.95, 0.99),
  eps 1e-10, weight decay 0.1.
* Same architecture, data, schedule, batch size and seeds as above.

Muon received the largest per-dimension search budget of the three arms in the
paper's sweep.

---

## 4. Differences from the original runs

These scripts are a cleaned-up reimplementation of the private research code,
verified to produce bit-identical optimizer updates. Three deliberate
differences are worth stating:

1. **Seeding.** `--seed` here really seeds Python, NumPy, PyTorch and the data
   order. The original language-model runs did not seed, so their three runs per
   arm were independent random initializations rather than controlled seeds.
   Expect the same distribution of results, not identical curves.
2. **The fused optimizer tail.** The original 370M and 1B runs used a
   `torch.compile`-fused elementwise AdamW tail. This repository uses the plain
   `foreach` tail: identical arithmetic, differing only in floating-point
   rounding.
3. **Diagnostics and FLOP accounting are not included.** The mechanistic
   diagnostics of Section 4.2, the μP coordinate check and the FLOP/wall-clock
   counters live in the private research harness. The quantities they measure
   are defined in the paper and can be recomputed from the buffers this package
   exposes (`optimizer.state[p]['exp_avg']`, `capture` for per-layer inputs).
4. **Two performance levers are on by default here.** The published runs used a
   fixed 4096-token through-N chunk and the eager row normalization. This
   package sizes that chunk from a memory budget (pure reassociation of the same
   sum) and compiles the normalization on CUDA (slightly *more* accurate than
   eager, which reduces in lower precision). To pin the published arithmetic
   exactly, call `ak_momentum.set_fused_norm(False)` and pass
   `chunk_budget_frac=0.0` to the capture.
5. **No activation-norm tracing.** The private trainer carried an always-on
   per-module tracing hook that cost roughly a quarter of each step; it is not
   reproduced here, so these scripts run on the same hook-off axis the paper's
   wall-clock constants are measured on.

Exact bit-level reproduction of a GPU run is not achievable in any case:
cuDNN/cuBLAS kernel selection and atomics make large-scale training
non-deterministic even at a fixed seed.
