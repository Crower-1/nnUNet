"""
for f in $(find . -type f -name "*_wbp_corrected.mrc"); do
    echo "Processing $f ..."
    /usr/bin/env python3 "/home/liushuo/Documents/code/nnUNet/predict_ribo.py" \
        --img_path "$f" \
        --save ribo \
        --smallest_size 50 \
        --score_thr 0.05
done

"""

#!/usr/bin/env python3
"""
A command-line tool to perform nnUNet segmentation on a 3D MRC image,
apply ribosome post-processing (optional opening, instance labeling, size filtering),
save specified results and export instance coordinates (always saved).

Usage:
    python segment_ribo.py --img_path /path/to/image.mrc [--save semantic ribo ribo_prob] [--do_opening] [--smallest_size 300]

By default, only 'ribo' results are saved. To save other outputs, include them in the --save list.
Coordinate file (.coords) is always generated.
Opening operation is off by default; enable with --do_opening.
"""
import os

# os.environ["nnUNet_raw"] = "/home/liushuo/Documents/data/nnUNet/"
# os.environ["nnUNet_preprocessed"] = "/home/liushuo/Documents/data/nnUNet/"
# os.environ["nnUNet_results"] = "/home/liushuo/Documents/data/nnUNet/"

import argparse
from pathlib import Path
import torch
import numpy as np
import mrcfile
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.mrc_reader_writer import MRCIO
from skimage.morphology import ball, opening
from skimage.measure import label



# Paths to trained model files (update these if needed)
DATASET_JSON = "/home/liushuo/Documents/data/nnUNet/nnUNet_results/Dataset010_6tomo_11classes/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json"
PLANS_JSON = "/home/liushuo/Documents/data/nnUNet/nnUNet_results/Dataset010_6tomo_11classes/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json"
CHECKPOINT = "/home/liushuo/Documents/data/nnUNet/nnUNet_results/Dataset010_6tomo_11classes/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"

# Save methods

def save_tomo_uint(data: np.ndarray, path: str, voxel_size: float, dtype=np.uint8):
    """Save a 3D numpy array as an MRC file with specified dtype."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data.astype(dtype))
        mrc.voxel_size = voxel_size


def save_tomo_float32(data: np.ndarray, path: str, voxel_size: float):
    """Save a 3D numpy float32 array as an MRC file."""
    save_tomo_uint(data, path, voxel_size, dtype=np.float32)


def remove_small_spots(instance_label: np.ndarray, smallest_size: int=300) -> np.ndarray:
    flat = instance_label.ravel()
    counts = np.bincount(flat)
    large_labels = np.nonzero(counts >= smallest_size)[0]
    mapping = np.zeros_like(counts, dtype=instance_label.dtype)
    mapping[large_labels] = large_labels
    flat[:] = mapping[flat]
    return flat.reshape(instance_label.shape)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment a 3D MRC image with nnUNet and save specified outputs."
    )
    parser.add_argument(
        "--img_path", required=True,
        help="Path to the input .mrc image"
    )
    parser.add_argument(
        "--save", nargs='+', choices=['semantic','ribo','ribo_prob'],
        default=['ribo'],
        help="Which results to save: semantic, ribo, ribo_prob. Default: ['ribo']"
    )
    parser.add_argument(
        "--do_opening", action='store_true',
        help="Apply ball opening (radius=3) to ribo mask before instance labeling. Default: off"
    )
    parser.add_argument(
        "--smallest_size", type=int, default=150,
        help="Minimum voxel count to keep an instance (default: 150)"
    )
    parser.add_argument(
        "--pixel_size", type=float, default=None,
        help="Override voxel spacing for the input (applies the same value to z/y/x)"
    )
    parser.add_argument(
        "--score_thr", type=float, default=None,
        help="Optional probability threshold (0-1) applied to ribo_prob to create the binary mask"
    )
    args = parser.parse_args()
    if args.score_thr is not None and not (0.0 < args.score_thr < 1.0):
        parser.error("--score_thr must be between 0 and 1 (exclusive)")
    return args


def main():
    args = parse_args()
    img_path = args.img_path
    save_targets = set(args.save)
    do_opening = args.do_opening
    smallest_size = args.smallest_size
    pixel_size = args.pixel_size
    score_thr = args.score_thr

    # Initialize predictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=False,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_files(
        dataset_json_path=DATASET_JSON,
        plans_json_path=PLANS_JSON,
        checkpoint_paths=CHECKPOINT,
    )

    # Read input image
    img, props = MRCIO().read_images([img_path])
    if pixel_size is not None:
        props = {'spacing': (pixel_size, pixel_size, pixel_size)}

    # Prediction
    semantic_results, prob_results = predictor.predict_single_npy_array(
        img, props, None, None, True
    )

    # Derive ribo mask and probability
    ribo_binary = np.zeros_like(semantic_results, dtype=np.uint8)
    ribo_binary[semantic_results == 11] = 1
    ribo_prob = prob_results[11]

    # Optional opening
    processed = ribo_binary
    if score_thr is not None:
        processed = (ribo_prob > score_thr).astype(np.uint8)
    if do_opening:
        processed = opening(processed, ball(3))

    # Instance labeling and filtering
    inst_label = label(processed)
    inst_label = remove_small_spots(inst_label, smallest_size)
    inst_label = label(inst_label)
    num_instances = inst_label.max()

    # Prepare outputs
    base = os.path.splitext(img_path)[0]
    out_paths = {
        'semantic': f"{base}_semantic.mrc",
        'ribo': f"{base}_ribo.mrc",
        'ribo_prob': f"{base}_ribo_prob.mrc",
        'coords': f"{base}.coords"
    }
    voxel_size = props.get('spacing', None) or props.get('voxel_size', 1.0)

    # Save semantic
    if 'semantic' in save_targets:
        save_tomo_uint(semantic_results, out_paths['semantic'], voxel_size)
        print(f"Saved semantic labels: {out_paths['semantic']}")
    # Save ribo instances
    if 'ribo' in save_targets:
        dtype = np.uint8 if num_instances < 126 else np.int16
        save_tomo_uint(inst_label, out_paths['ribo'], voxel_size, dtype=dtype)
        print(f"Saved ribo instances: {out_paths['ribo']} (dtype={dtype.__name__})")
    # Save ribo probability
    if 'ribo_prob' in save_targets:
        save_tomo_float32(ribo_prob, out_paths['ribo_prob'], voxel_size)
        print(f"Saved ribo probability map: {out_paths['ribo_prob']}")

    # Always save coords
    coords_file = out_paths['coords']
    with open(coords_file, 'w') as f:
        f.write('id z y x\n')
        for inst_id in range(1, num_instances+1):
            positions = np.argwhere(inst_label == inst_id)
            if positions.size == 0:
                continue
            centroid = positions.mean(axis=0)
            z, y, x = centroid
            f.write(f"{inst_id} {z:.2f} {y:.2f} {x:.2f}\n")
    print(f"Saved instance coordinates: {coords_file}")

if __name__ == '__main__':
    main()
