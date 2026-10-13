# Compressed MRC volumes

`MRCIO` reads and writes `.mrc.7z` as well as plain `.mrc`, `.mrcs`, `.rec`,
and `.map` files. A 7z archive must contain exactly one `.mrc` file; its inner
name may differ from the archive name. Reading extracts that member into memory
without creating an uncompressed file alongside the dataset. `mrcfile>=1.5` and
`py7zr>=1.0` are required.

Arrays retain MRC Z,Y,X ordering, and spacing comes from the MRC header. As with
plain MRC, nnU-Net receives float32 arrays and segmentation output is encoded as
float32 MRC. Compressed writes use a temporary MRC, then atomically replace the
destination archive after compression succeeds.

For uniformly compressed datasets, use `file_ending: ".mrc.7z"`,
`imagesTr/<case>_0000.mrc.7z`, and `labelsTr/<case>.mrc.7z`.

Plain images can also be paired with compressed labels through nnU-Net's existing
explicit `dataset` mapping in `dataset.json`:

```json
{
  "channel_names": {"0": "cryoET"},
  "labels": {"background": 0, "activezone": 1},
  "numTraining": 1,
  "file_ending": ".mrc.7z",
  "overwrite_image_reader_writer": "MRCIO",
  "dataset": {
    "case": {
      "images": ["imagesTr/case_0000.mrc"],
      "label": "labelsTr/case.mrc.7z"
    }
  }
}
```

`file_ending` also determines prediction and copied ground-truth filenames, so
use `.mrc.7z` when labels are archived. Folder-based inference searches for that
same suffix. To infer from plain `.mrc` images with such a model, pass explicit
image-path lists to `nnUNetPredictor.predict_from_files` instead of an input
folder. The reader supports both formats in the same channel list.
