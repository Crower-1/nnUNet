# Particle CC-DiceCE (r02)

The existing `nnUNetTrainerFinetuneGeometry` now adds particle CC-DiceCE on the
highest-resolution output. Base Dice/CE, deep supervision, membrane/filament
objectives, head names/order, fine-tuning modes and inference stay compatible.
Only nnUNet is changed; nnInteractive and RELION are outside this implementation.

## Enable all cfg particles in a new experiment

Re-run the existing importer against the actual dataset/plans:

```sh
conda run -n synapseseg python -m nnunetv2.dataset_conversion.import_synapseseg_cfg \
  --cfg /path/to/synapseSeg.cfg --dataset /path/to/dataset.json \
  --plans /path/to/nnUNetPlans.json --configuration 3d_fullres \
  --output /path/to/label_geometry.json
```

Embed the generated object as `label_geometry` in the selected plans
configuration, as for membrane/filament supervision. All bound `particle=true`
body targets receive `geometry_type: "particle"`; there is no name whitelist.
Unbound cfg categories are reported, never added to the network or denominator.
Old imported metadata contains no particle flag: reimporting is required for
**new** particle runs; the trainer never guesses particle identity from names.

Missing `geometry_loss.enabled` defaults to true when particle targets exist;
explicit false is respected. Missing `geometry_loss.particle` uses these defaults:

```json
{
  "enabled": true,
  "targets": "all_cfg_particles",
  "profile": "cc_dice_ce",
  "weight": 0.25,
  "class_weights": {},
  "apply_to": "highest_resolution",
  "component_source": "semantic_cc",
  "connectivity": 26,
  "distance_units": "voxel",
  "empty_target": "skip_aux",
  "backend": "scipy_reference"
}
```

Place this object under `geometry_loss.particle`, not at the plans root.
All class weights default to 1 and are expanded in the saved snapshot. Explicit
named weights must be positive and bound to particle targets; the group weight
or enabled flag provides the baseline switch. Ambiguous particle+filament or
membrane/body roles fail with a diagnostic. Resolve through importer
`--overrides` (a JSON list of targets) with an `override_reason` on every entry;
explicit overrides and their reasons are printed and saved in target metadata.
Particle unions are rejected: each particle semantic label is evaluated separately.

Use `-tr nnUNetTrainerFinetuneGeometry` with normal training/fine-tuning arguments.
The trainer supports integer-label 3D softmax only; it rejects region training
when geometry is active. Distance is explicitly **voxel-grid Euclidean distance**,
including on anisotropic grids. Physical-distance training is not implemented;
ID/OD/radius fields never determine spacing, labels or channels.

## Objective, valid regions and distributed reduction

For each patch and particle class: label foreground with 26-connectivity, assign
every valid voxel (including distant background) to its nearest foreground voxel's
component, and compute object-mean foreground Dice loss plus region-mean binary
CE. Dice has no numerator smoothing; denominator epsilon is 1e-7. The stable
one-vs-rest logit is `z_c - logsumexp(z_other)` and includes all competitor gradients.
`cc_dice` is an explicit optional ablation. Both terms have unit coefficients.

Components/Voronoi are GT-only, eager and no-grad. Scatter reductions use O(N+K)
storage/work after partition construction; there are no `[K,D,H,W]` one-hot masks.
Only one class/sample partition is built at a time. FP32 loss arithmetic is used
outside autocast. Empty foreground skips CCL/EDT and contributes connected zero;
a singleton component owns the whole valid field. Empty class/patch values remain
in fixed class and patch denominators. Original global loss still supervises them.

The data pipeline captures `particle_valid` after spatial transforms/mirroring and
before `-1` padding becomes background. It is collated separately, never an image
channel or semantic target channel. Global ignore is removed from it at loss time.
Only the particle term uses this mask, preserving previous base/geometry behavior.
Internal ignore touching a CC is reported as `unknown_touching_cc_count`; such
truncated objects need separate data QA. No hole filling, watershed, small-object
removal or source-label rewriting occurs.

Particle DDP uses one collective on every rank regardless of local foreground.
Its backward scale is `world_size / global_patch_count` before DDP's gradient
average, so unequal local batch sizes agree with the same single-device global
batch. The existing base/membrane/filament reductions are unchanged (including
their prior rank-mean semantics). Each rank writes its own diagnostics; aggregate
patch/CC counts across ranks, not unweighted means of per-rank percentages.

## Backend, timing and logs

`scipy_reference` is the explicit default. It synchronizes GT to CPU, labels it
and constructs EDT, then returns IDs to the training device. It is a correctness
reference and can be costly for large patches. Select `backend: "cupy"` explicitly
for GPU partitions; install a compatible CuPy separately. Missing CUDA/CuPy fails
early on initialization; no automatic CPU fallback. Disabled/zero-weight particle
runs do not call CC/EDT or import CuPy.

CuPy runs on the rank's PyTorch device/current stream with DLPack-owned arrays.
Nearest-distance ties may select different valid seeds on SciPy and CuPy; this is
not a floating-point tolerance issue. The optional parity test uses a no-tie grid.
GPU behavior is implemented but was not executed in the available CPU environment.

`geometry_steps_rankN.jsonl` records each train/validation step: base loss, existing
geometry raw terms, per-class CC Dice/CE, weighted particle contribution, patch/CC
counts, empty/single-object fractions, boundary/unknown-touching counts, volume and
equivalent-scale quantiles, geometry/reduction/step wall times and data wait.
On CUDA it also records PyTorch peak allocation, sampled whole-device usage and
CuPy pool reservation. Whole-device usage is a sample, not a continuously measured
peak; GPU total-peak profiling remains to be done. JSON logging and CPU transfers
have overhead and are included in step timing. Logs do not imply biological recall.

## Checkpoints and old runs

Snapshots include resolved mappings, class weights, effective options, backend,
`semantic-cc-voronoi-dicece-v1`, upstream commit, valid-region protocol and SHA-256
configuration hash in `init_args.geometry_state` and `geometry_resolved.json`.
New particle snapshots must match on resume; changing weights, mapping, profile
or backend requires a new experiment.

Pre-particle geometry snapshots restore their saved objective/mapping with particle
disabled, even if today's metadata auto-routes particles. The migration flag remains
in subsequent snapshots, preserving this behavior across repeated resumes. Vanilla
`nnUNetTrainer`/`nnUNetTrainerFinetune` checkpoints without a geometry snapshot restore
a disabled base objective. Unidentified/malformed missing snapshots fail rather than
guessing an objective. Start a new fine-tuning run to change the objective; do not
reuse an old optimizer state as unchanged continuation. Inference does not initialize
losses or backends and also accepts old checkpoints without geometry metadata.

## Audit masks and evaluate candidates

Audit a final-grid 3D semantic NumPy array (export/copy only, no source modification):

```sh
conda run -n synapseseg python -m nnunetv2.evaluation.particle_audit \
  --mask final_mask.npy --dataset dataset.json --geometry label_geometry.json \
  --output audit --patch-size 64 64 64 --max-patches 16
```

Outputs: per-case/per-class and sampled patch CC statistics, equivalent diameters,
empty/small/singleton/boundary statistics, target/CC/Voronoi PNG/NPZ slices chosen
at the largest foreground cross-section (slice index recorded).
Optional `--resampled-mask actual_resampled_mask.npy` reports before/after count
changes; these flag candidates for QA, not proven merge/split rates. With semantic
labels alone true instance identity is unknown. Optional `--instance-ids ids.npy`
compares reliable aligned IDs with CCs, reporting multiple IDs in one CC or one ID
split across CCs. Optional `--centers-json centers.json` accepts
`{"coordinate_units":"voxel","axis_order":"DHW","centers":{"ribo":[[3,4,5]]}}`
in the final training grid; it reports multiple centers per CC, missing-CC and
out-of-volume centers. Centers use nearest-voxel lookup (`numpy.rint`) only for QA;
they never replace semantic masks, create CC seeds or become instance labels.

Evaluate candidate lists in **physical coordinates**, with preselected class-specific
tolerances. All GT/candidates must cover the same complete valid tomo region:

```json
{
  "coordinate_units": "nm",
  "tolerances": {"ribo": 10, "proteasome": 12},
  "cases": [{
    "case_id": "tomo_001",
    "source": "preparation_A",
    "valid_region_confirmed": true,
    "valid_volume": 1000000,
    "classes": {
      "ribo": {"gt": [[10,20,30]], "predictions": [{"center": [11,20,30], "score": 0.2}]},
      "proteasome": {"gt": [], "predictions": []}
    }
  }]
}
```

Coordinates, tolerance and valid-volume units must be consistent. Values above
are illustrative, not a recommended biological tolerance. `source` controls
bootstrap grouping; otherwise a tomogram is the unit. Optional `predictions_before`
for every case/class enables a before/after-processing comparison.

```sh
conda run -n synapseseg python -m nnunetv2.evaluation.particle_recall \
  --input candidates.json --output recall.json \
  --thresholds 0.1 0.2 0.3 0.5 --budgets 100 500 1000 --bootstrap 500
```

The evaluator maximizes one-to-one match count, then minimizes distance. It reports
recall, FN/miss rate, precision, FP/tomo/volume, candidate burden/density, duplicates,
localization error, per-class and macro recall, PR points, and recall-versus-candidate
budget points. Budgets are explicit **evaluation-only** per-tomo/per-class confidence
prefixes; they add no production top-K filtering. No high-precision admission threshold.
Existing probability export can be used for a separate low-threshold candidate study;
this change neither replaces semantic argmax outputs nor changes RELION interfaces.
Threshold/weight choice belongs to validation, not the test set.

## Synthetic verification and limits

See `particle_loss_validation.md` for actual commands/results and the CPU benchmark.
No real P0/P1 training is run in this request, as instructed by the user. No claim of
particle recall improvement is made. Future P0/P1 use identical splits, initialization,
sampling/steps, membrane/filament weights and candidate extraction, with all particle
classes enabled in P1 at 0.25; optional weights 0.5/1.0 and cc_dice are separate runs.
Keep non-particle metrics and candidate computational burden in the comparison.
