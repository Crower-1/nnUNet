# Geometry reference sources

Copied from the sibling nnInteractive repository. Reference files are test-only,
not runtime dependencies. Runtime adaptation: nnunetv2/training/loss/geometry_losses.py.

- MemBrain Surface-Dice: commit e4183688a5eb4834faf6a06d4d47ff992a1c781a,
  https://github.com/teamtomo/membrain-seg/blob/e4183688a5eb4834faf6a06d4d47ff992a1c781a/src/membrain_seg/segmentation/training/surface_dice.py
  BSD license in MemBrain-LICENSE; clDice MIT notice in the reference and clDice-LICENSE.
- Skeleton-Recall: commit 56c19155f91097e4b15cfd7106cd960f888e040a,
  https://github.com/MIC-DKFZ/Skeleton-Recall/blob/56c19155f91097e4b15cfd7106cd960f888e040a/skeleton_recall/training/data_augmentation/skeletonization.py
  Apache-2.0 license in Skeleton-Recall-LICENSE.

Runtime receives multiclass softmax probabilities without sigmoid. GT skeleton
support is dilated and intersected with its own foreground. Recall loss is
1-recall (upstream -recall differs by a constant). Surface loss uses a reliable
interior region excluding ignored voxels and patch exterior with radius
kernel//2 + iterations + 2. It is MemBrain soft middle-surface coverage, not
physical-distance tolerance Surface Dice.
