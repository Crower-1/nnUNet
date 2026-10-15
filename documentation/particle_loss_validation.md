# Particle r02 implementation validation — 2026-10-15

All Python commands used the `synapseseg` conda environment. No real training,
real dataset evaluation, threshold tuning or RELION run was performed, per user
instruction. This report describes synthetic correctness/performance only.

## Executed tests

```sh
conda run -n synapseseg python -m unittest discover -s tests/training -v
conda run -n synapseseg python -m unittest discover -s tests/evaluation -v
```

Training: 25 tests, 24 passed, 1 skipped (CUDA/CuPy unavailable).
Evaluation: 4 passed. Total: 28 passed, 1 explicitly skipped.
The pipeline subset was also re-run after adding vanilla pre-geometry checkpoint
coverage. CPU DDP tests required permission for local loopback communication;
initial sandbox network restrictions are not an algorithm failure.

Coverage:

- Every bound cfg particle, including arbitrary new names; unbound reporting,
  role/flag conflicts, explicit override provenance and no whitelist.
- New-run auto enable; legacy geometry and vanilla checkpoints keep their saved
  objectives with particle disabled; new snapshot changes are rejected.
- Particle disabled / zero weight preserves baseline values and gradients and
  never calls CC; single-scale particle objective leaves low-resolution gradients unchanged.
- One-vs-rest full-softmax values and all-class gradients; FP32 reductions under
  CPU autocast with extreme logits; finite empty/all-invalid losses.
- Scatter loss and logits gradients versus a naive region loop and pinned upstream
  pure reductions for **nonempty two-class** CC-Dice and CC-DiceCE. This does not
  claim equivalence to the upstream empty fallback or saturated/clamped probability CE.
- Full-view background/remote false-positive gradients; 26-connectivity, singleton,
  full foreground, different touching classes, padding/ignore invariance and diagnostics.
- Actual loader collation and spatial augmentation validity capture; optimizer
  train step, validation, logging, save/resume roundtrip on a tiny CPU network.
- CPU DDP with empty-class rank and unequal local sizes (1 and 2): particle values
  and parameter gradients match the same three-patch single-device global batch.
- Original model-folder predictor reloads a tiny PlainConvUNetHead without CuPy or
  particle training modules, reproducing identical outputs.
- Recall matching prioritizes cardinality over distance, handles duplicates/empty
  cases, macro averages rare classes, and reports explicit threshold/budget changes.
- Optional aligned instance-ID and center audits detect synthetic merged/split
  correspondences without changing masks.

## CPU loss-only benchmark

```sh
conda run -n synapseseg python -m nnunetv2.evaluation.benchmark_particle_loss \
  --output /tmp/nnunet_particle_benchmark.json
```

Raw result saved in `particle_loss_benchmark_cpu.json`.
PyTorch 2.14.0+cu130, CPU, 2 threads, logits shape `[2,3,32,32,32]`, two particle
classes with 63 and 62 singleton CCs per patch. One warm-up, five measured iterations.
Both P0 and P1 include base Dice/CE; P1 adds particle CC-DiceCE at weight 0.25.

| Measurement | P0 particle off | P1 particle on |
|---|---:|---:|
| Median loss forward+backward | 1.139 ms | 13.393 ms |
| Loss value (synthetic random logits) | 1.3881705 | 1.7596569 |
| Finite gradients | yes | yes |

Last P1 geometry construction total: 7.935 ms; tensor reduction total: 0.911 ms.
This small loss-only CPU benchmark is not full-network throughput, GPU performance,
or evidence of improved recall. CPU CCL/EDT overhead is material and scales with
patch size and class count. Timings depend on hardware and concurrent workload.

## Executed CLI smoke

Generated **synthetic** inputs under `/tmp/nnunet_particle_cli_smoke` with the
script `/tmp/nnunet_particle_cli_smoke.py`; no source dataset was modified.

```sh
conda run -n synapseseg python -m nnunetv2.dataset_conversion.import_synapseseg_cfg \
  --cfg /tmp/nnunet_particle_cli_smoke/cfg.json \
  --dataset /tmp/nnunet_particle_cli_smoke/dataset.json \
  --plans /tmp/nnunet_particle_cli_smoke/plans.json --configuration 3d_fullres \
  --output /tmp/nnunet_particle_cli_smoke/label_geometry.json
conda run -n synapseseg python -m nnunetv2.evaluation.particle_audit \
  --mask /tmp/nnunet_particle_cli_smoke/mask.npy \
  --dataset /tmp/nnunet_particle_cli_smoke/dataset.json \
  --geometry /tmp/nnunet_particle_cli_smoke/label_geometry.json \
  --output /tmp/nnunet_particle_cli_smoke/audit --patch-size 12 12 12 \
  --resampled-mask /tmp/nnunet_particle_cli_smoke/resampled.npy
conda run -n synapseseg python -m nnunetv2.evaluation.particle_recall \
  --input /tmp/nnunet_particle_cli_smoke/candidates.json \
  --output /tmp/nnunet_particle_cli_smoke/recall.json \
  --thresholds .1 .5 .9 --budgets 0 1 2 3
```

Synthetic importer mapping: ribo→label/channel 1, proteasome→2, granule→3,
all `particle`; `future_particle` correctly reported unbound. These numeric IDs
are test fixtures, **not mappings inferred for the user's training dataset**.
Audit found CC counts 2/1/1 and generated eight patch summaries plus PNG/NPZ
previews. Actual nearest-neighbor synthetic downsampling removed the two singleton
ribo CCs and the report recorded a -2 delta; labels were not repaired.
Recall smoke: ribo TP=2, FN=0, FP=1, duplicate=1; proteasome FN=1. Per-class recall
was 1.0 and 0.0; macro recall 0.5. These are deliberately constructed examples,
not model predictions or biological performance results.

## Not executed

- CUDA AMP, CUDA/CuPy partition/value/gradient parity and stream/device behavior.
- Multi-GPU/NCCL tests (CPU/Gloo DDP was executed).
- Whole-device peak GPU memory and full-network/data-loader throughput profiling.
- Real mask audit, P0/P1 training, recall/non-particle regression or RELION evaluation.

No nnInteractive modifications: `git status --porcelain` remained empty there.
The existing local nnUNet edits/untracked files were retained. No model architecture,
head or inference-protocol change was introduced and no upstream fork was installed.
