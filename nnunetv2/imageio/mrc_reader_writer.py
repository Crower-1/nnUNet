# mrc_reader_writer.py
#    Copyright
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""Read 3D MRC volumes, including single-volume .mrc.7z archives, for nnU-Net."""

import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple, Union, List
import numpy as np
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
import mrcfile
from nnunetv2.imageio.mrc_archive import open_compressed_mrc, write_compressed_mrc


def _voxel_size_to_spacing(voxel_size) -> Tuple[float, float, float]:
    if voxel_size is None or not hasattr(voxel_size, 'x'):
        return 1.0, 1.0, 1.0
    x = float(voxel_size['x']) if voxel_size['x'] > 0 else 1.0
    y = float(voxel_size['y']) if voxel_size['y'] > 0 else 1.0
    z = float(voxel_size['z']) if voxel_size['z'] > 0 else 1.0
    return z, y, x


def _spacing_to_voxel_size(spacing: Union[List[float], Tuple[float, ...]]) -> Tuple[float, float, float]:
    if not isinstance(spacing, (list, tuple)) or len(spacing) != 3:
        raise ValueError(f"Spacing must be a tuple/list of three floats. Provided spacing: {spacing}")
    z, y, x = spacing
    return float(x), float(y), float(z)


@contextmanager
def _open_mrc(filename):
    """Open a plain MRC or compressed MRC using the same volume interface."""
    if os.fspath(filename).lower().endswith('.mrc.7z'):
        with open_compressed_mrc(filename) as volume:
            yield volume
    else:
        with mrcfile.open(filename, permissive=True) as volume:
            yield volume


class MRCIO(BaseReaderWriter):
    """Read/write 3D MRC data; spacing uses the MRC header in Z,Y,X order."""

    supported_file_endings = ['.mrc', '.mrcs', '.rec', '.map', '.mrc.7z']

    def _read_volume(self, filename):
        if not os.fspath(filename).lower().endswith(tuple(self.supported_file_endings)):
            raise ValueError(f'Unsupported MRC file ending: {filename}')
        with _open_mrc(filename) as volume:
            if volume.data is None or volume.data.ndim != 3:
                raise RuntimeError(f'Only 3D MRC volumes are supported: {filename}')
            data = volume.data.astype(np.float32)
            spacing = _voxel_size_to_spacing(volume.voxel_size)
        return data, spacing

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        images, spacings = [], []
        for filename in image_fnames:
            data, spacing = self._read_volume(filename)
            images.append(data)
            spacings.append(spacing)
        if not images:
            raise ValueError('At least one image channel is required.')
        if not self._check_all_same([image.shape for image in images]):
            raise RuntimeError(f'Inconsistent image shapes: {image_fnames}')
        if not np.allclose(spacings, spacings[0]):
            raise RuntimeError(f'Inconsistent image spacings: {image_fnames}')
        return np.stack(images, axis=0), {'spacing': spacings[0]}

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        data, spacing = self._read_volume(seg_fname)
        return data[None], {'spacing': spacing}

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        if seg.ndim != 3:
            raise RuntimeError(f'Only 3D segmentations are supported: {output_fname}')
        voxel_size = _spacing_to_voxel_size(properties.get('spacing', (1.0, 1.0, 1.0)))
        if os.fspath(output_fname).lower().endswith('.mrc.7z'):
            # Keep the existing MRC float32 encoding and avoid a permanent expanded volume.
            with TemporaryDirectory(prefix='nnunet-mrc-') as temporary:
                mrc_path = Path(temporary) / Path(output_fname).name[:-3]
                self._write_plain(seg, mrc_path, voxel_size)
                write_compressed_mrc(mrc_path, output_fname)
        else:
            self._write_plain(seg, output_fname, voxel_size)

    @staticmethod
    def _write_plain(seg, output_fname, voxel_size):
        with mrcfile.new(output_fname, overwrite=True) as volume:
            volume.set_data(seg.astype(np.float32, copy=False))
            volume.voxel_size = voxel_size
