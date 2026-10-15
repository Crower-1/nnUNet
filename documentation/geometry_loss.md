# SynapseSeg geometry loss

Particle CC-DiceCE extends this same trainer/configuration; see [particle_loss.md](particle_loss.md).

Use `nnUNetTrainerFinetuneGeometry` to preserve the existing fine-tuning modes,
multiclass Dice/CE and full deep supervision. No network or inference changes.
Version 1 supports 3D integer-label datasets with global ignore only; region
labels and per-class partial annotation are rejected/unsupported.

## Import and enable

Run in the `synapseseg` conda environment, from the nnUNet checkout:

```sh
conda run -n synapseseg python -m nnunetv2.dataset_conversion.import_synapseseg_cfg \
  --cfg /path/to/default_cfg.cfg --dataset /path/to/dataset.json \
  --plans /path/to/nnUNetPlans.json --configuration 3d_fullres \
  --output /path/to/label_geometry.json
```

The importer prints category, role, names, IDs and channels plus unbound cfg
categories and unmatched labels. It never changes masks, heads or channel order.
An inconsistent `class_names` order raises an error. Correct plans only when
creating a compatible new experiment; do not relabel existing checkpoint heads.
`--overrides` accepts a JSON list of named targets, including label unions.
Explicit unknown labels fail. mt and mt_memb remain default until mask QA.

Embed the generated JSON object under `label_geometry` in the selected plans
configuration, next to `architecture`, and add:

```json
"geometry_loss": {
  "enabled": true,
  "apply_to": "highest_resolution",
  "surface_weight": 0.25,
  "skeleton_weight": 0.25,
  "reduction": "mean_per_geometry"
}
```

Train using the normal command with `-tr nnUNetTrainerFinetuneGeometry`.
Existing `--finetune_mode` and component weight arguments still apply.
No geometry work or imports of geometry losses occur when enabled is false.
For the four baseline/surface/skeleton/combined experiments, use distinct plans
identifiers (and thus output folders); toggle enabled and the two group weights.
The weights are starting experiment values, not validated hyperparameters.

## Objective and reproducibility

The complete original DS loss remains unchanged. Only the largest output adds
geometry terms; each union probability sums the selected multiclass softmax
channels. Arithmetic and morphology use FP32 outside autocast. Skeleton support
is generated lazily from each selected GT after spatial augmentation, on CPU,
then returned to the logits device. Prediction skeletonization is only used for
membranes and retains gradients. Current implementation incurs device sync and
CPU skeletonization each loss call; worker-side caching is a future optimization.

Each declared target contributes one batch mean; an empty skeleton/middle or
fully ignored score region contributes zero. Targets are averaged within each
geometry group using a fixed declared denominator, including empty targets.
DDP uses rank-local means followed by its usual gradient averaging; there are no
geometry collectives or class-dependent collective branches. Unequal rank batch
sizes therefore receive equal rank weight, not global sample weighting.
Small patches may have no reliable membrane interior and contribute zero.
The default reliable radius is 14 voxels; inspect patch sizes and label geometry.
Dilation defaults to 2 voxels; cfg ID/OD are never used as labels or radii.

Resolved mappings and all effective parameters are stored in checkpoint
`init_args.geometry_state`, and in `geometry_resolved.json` plus the training log.
New snapshots reject changed mappings/options. Pre-particle checkpoints restore
their saved objective with particle disabled (see the migration rules in particle_loss.md).
Use existing pretrained weight options for a new
experiment. Inference reads only the network/plans and never builds this loss.

## Validation limits

Run `conda run -n synapseseg python -m unittest discover -s tests/training -v`.
Synthetic tests cover mapping, unions/default import behavior, disabled loss and
gradient equality, softmax coupling, ignore/empty targets, highest-resolution
selection, surface gradients and pinned MemBrain numerical agreement.
No real dataset training or GPU memory benchmark is implied by these tests.
Before adoption compare identical-split baseline/surface/skeleton/combined runs:
membrane Dice, MemBrain Surface-Dice, boundary distance and connectivity;
filament centerline precision/recall, clDice, breaks/merges and trace lengths;
all remaining classes' Dice. Inspect mt masks before explicit override.
