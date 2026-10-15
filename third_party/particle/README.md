# Particle CC-DiceCE reference

Upstream: TIO-IKIM/Learning-to-Look-Closer, commit
`e5013bdb560eee544a94e1f068cb5cb4d003a51c`, Apache-2.0 (`LICENSE`).

Unmodified reference downloads:

- https://github.com/TIO-IKIM/Learning-to-Look-Closer/blob/e5013bdb560eee544a94e1f068cb5cb4d003a51c/nnunetv2/training/loss/instance_losses.py
- https://github.com/TIO-IKIM/Learning-to-Look-Closer/blob/e5013bdb560eee544a94e1f068cb5cb4d003a51c/nnunetv2/utilities/connected_components.py

These files are **not runtime imports**. Tests extract only the pinned pure
scatter/reduction functions with Python AST. Importing the complete upstream
module would change global Dynamo settings and require its backend imports;
neither behavior is part of this implementation.

Runtime adaptations are `training/loss/particle_losses.py` and
`training/data_augmentation/custom_transforms/particle_targets.py`:

- Apply class-wise to selected multiclass softmax channels, highest resolution
  only, in addition to the complete existing loss.
- Use stable one-vs-rest BCE logits instead of clamped probability log CE;
  finite, ordinary two-class logits agree with the reference. Extreme logits
  intentionally do not inherit the upstream probability clamp/saturation.
- Empty foreground yields graph-connected zero instead of upstream CE+constant
  fallback; ignored voxels and padding are excluded from seeds and reductions.
- SciPy CPU reference plus explicitly selected optional CuPy, never silent fallback.
- Fixed class weights and global-sample-weighted particle DDP reduction.
- No change to global compiler limits or network/inference architecture.

This is a multiclass engineering adaptation, not a complete paper reproduction
or evidence of improved cryo-ET particle recall.
