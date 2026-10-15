# Dataset008 particle training submission — 2026-10-15

Host: `172.20.175.234` (mn02); Slurm partition `tao`; compute node `t001`.
Environment: `/share/home/liushuo/miniconda3/envs/synapseseg`.
Code: `/share/data/CryoET_Data/liushuo/code/nnUNet`.

| Active job | Physical GPU | Initialization | Plans identifier |
|---|---:|---|---|
| 102380 | 0 (RTX 3090) | Random, all trainable modules | nnUNetPlans_particle_scratch_20261015 |
| 102381 | 1 (RTX 3090) | Specified checkpoint encoder; decoder/heads random; all trainable modules | nnUNetPlans_particle_encoder_20261015 |

Both jobs use Dataset008_proteasome, `3d_fullres`, fold 0 (26 train/7 validation),
batch size 2, patch `[48,224,224]`, the existing split/preprocessed data, main-process
seed 20261015, particle CC-DiceCE weight 0.25 and explicit `scipy_reference` backend.
CuPy is not installed. The existing base loss/deep supervision remains enabled.
`proteasome` binds to label/channel 1; background=0, global ignore=2 outside ROI.
No supplemental data was required. Raw masks/plans/checkpoint were not overwritten.

Per-job Slurm request: one GPU, 16 CPUs, 96 GiB RAM, seven-day time limit.
Normal trainer schedule remains 1000 epochs, 250 train iterations/epoch; the time
limit does not guarantee completion. `nnUNet_compile=false`, eight augmentation
workers, final validation probability export (`--npz`) enabled for both runs.
The launch script refuses to run if Slurm assigns a different physical GPU.

Remote scripts, extracted weights, deployment backup, hashes and experiment metadata:

```
/share/data/CryoET_Data/liushuo/code/nnUNet/runs/particle_Dataset008_20261015/
```

Scripts: `scratch.sbatch`, `encoder.sbatch`, `train_seeded.py`.
Logs: `scratch_102380.out/.err`, `encoder_102381.out/.err`.
Current job IDs: `scratch.jobid`, `encoder.jobid`.

Output folders:

```
/share/data/CryoET_Data/liushuo/dataset/nnUNet/nnUNet_results/Dataset008_proteasome/nnUNetTrainerFinetuneGeometry__nnUNetPlans_particle_scratch_20261015__3d_fullres/fold_0
/share/data/CryoET_Data/liushuo/dataset/nnUNet/nnUNet_results/Dataset008_proteasome/nnUNetTrainerFinetuneGeometry__nnUNetPlans_particle_encoder_20261015__3d_fullres/fold_0
```

Encoder source:
`/share/data/CryoET_Data/liushuo/checkpoints/nnunet/checkpoint_best.pth`.
96 encoder state tensors were extracted with exact key/shape matching and strict
load into the Dataset008 encoder. Dataset008 architecture/strides remain common to
both experiments; the source checkpoint used a different patch and last two Z
strides, so only its compatible weights were transferred. Decoder/head weights,
optimizer and previous training epoch were not imported.

Deployment preserved the remote uncommitted network modification. Existing files
replaced by the loss upgrade were backed up in `code_before_upgrade.tar.gz`.
Core particle tests passed on the cluster login node (12 passed, one CUDA/CuPy skip),
and a short Slurm probe verified CUDA tensor execution on t001. No dependency was
installed or upstream fork replaced.

First attempts 102378/102379 exited before training because the seeded launcher
lacked a main guard for Python 3.14 forkserver imports. The launcher was corrected
and its worker import tested; the final submitted jobs above use that correction.
Failed attempt logs and job IDs are retained for provenance.

Monitoring stopped immediately after observing both active jobs in Epoch 0 with
actual optimization-step records: scratch 83 records and encoder 39 records at
that observation. First weighted particle contributions were 0.30128175 and
0.63115132 respectively, with finite base losses. Each reported approximately
7.54 GB PyTorch peak allocation on its first step. These are startup checks, not
quality comparisons or final results. Training remains running; no recurring
monitor was created.
