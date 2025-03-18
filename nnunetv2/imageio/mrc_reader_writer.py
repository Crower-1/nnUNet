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

import os.path
from typing import Tuple, Union, List
import numpy as np
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
import mrcfile
from batchgenerators.utilities.file_and_folder_operations import isfile, split_path, join

class MRCIO(BaseReaderWriter):
    """
    Reads and writes 3D MRC images. Uses mrcfile package.
    
    If you have 2D MRCs, ensure they are handled appropriately or raise an error.
    
    This Reader/Writer no longer uses auxiliary .json files for spacing information.
    Instead, it directly sets and reads the voxel_size attribute in the .mrc files.
    """

    supported_file_endings = [
        '.mrc',
        '.mrcs',
        '.rec'
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        """
        Reads 3D MRC images and retrieves spacing information from the voxel_size attribute.
        """
        # Check file extension
        ending = '.' + image_fnames[0].split('.')[-1]
        assert ending.lower() in self.supported_file_endings, f'Ending {ending} not supported by {self.__class__.__name__}'

        images = []
        for f in image_fnames:
            with mrcfile.open(f, permissive=True) as mrc:
                image = mrc.data.astype(np.float32)
                if image.ndim != 3:
                    raise RuntimeError(f"Only 3D images are supported! File: {f}")
                images.append(image)

        # Stack images along the first axis (channels)
        images = np.stack(images, axis=0)

        # Retrieve spacing information
        with mrcfile.open(image_fnames[0], permissive=True) as mrc:
            voxel_size = mrc.voxel_size
            if voxel_size is None or not hasattr(voxel_size, 'x'):
                print(f'WARNING: No valid voxel_size found in {image_fnames[0]}. Assuming spacing (1, 1, 1).')
                spacing = (1, 1, 1)
            else:
                spacing = (
                    float(voxel_size['x']) if voxel_size['x'] > 0 else 1.0,
                    float(voxel_size['y']) if voxel_size['y'] > 0 else 1.0,
                    float(voxel_size['z']) if voxel_size['z'] > 0 else 1.0,
                )

        # 验证所有图像具有相同的形状
        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError("Inconsistent image shapes detected.")

        return images, {'spacing': spacing}

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        """
        将分割图像写入 .mrc 文件，并设置 voxel_size 属性。
        """
        # 确保分割图像是3D的
        if seg.ndim != 3:
            raise RuntimeError(f"Only 3D segmentations are supported! File: {output_fname}")

        # 写入分割数据并设置 voxel_size
        with mrcfile.new(output_fname, overwrite=True) as mrc:
            mrc.set_data(seg.astype(np.float32, copy=False))
            voxel_size = properties.get('spacing', (1.0, 1.0, 1.0))
            if isinstance(voxel_size, (list, tuple)) and len(voxel_size) == 3:
                mrc.voxel_size = voxel_size
            else:
                raise ValueError(f"Spacing must be a tuple of three floats. Provided spacing: {voxel_size}")

        # 不再需要保存 .json 文件，因此移除相关代码
        # save_json({'spacing': properties['spacing']}, join(out_dir, file[:-(len(ending))] + '.json'))

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        """
        Reads a 3D MRC segmentation and retrieves spacing information from the voxel_size attribute.
        """
        # Check file extension
        ending = '.' + seg_fname.split('.')[-1]
        assert ending.lower() in self.supported_file_endings, f'Ending {ending} not supported by {self.__class__.__name__}'

        with mrcfile.open(seg_fname, permissive=True) as mrc:
            seg = mrc.data.astype(np.float32)
            if seg.ndim != 3:
                raise RuntimeError(f"Only 3D segmentations are supported! File: {seg_fname}")

        # Add channel axis for segmentation
        seg = seg[None]  # Shape: (1, x, y, z)

        # Retrieve spacing information
        with mrcfile.open(seg_fname, permissive=True) as mrc:
            voxel_size = mrc.voxel_size
            if voxel_size is None or not hasattr(voxel_size, 'x'):
                print(f'WARNING: No valid voxel_size found in {seg_fname}. Assuming spacing (1, 1, 1).')
                spacing = (1.0, 1.0, 1.0)
            else:
                spacing = (
                    float(voxel_size['x']) if voxel_size['x'] > 0 else 1.0,
                    float(voxel_size['y']) if voxel_size['y'] > 0 else 1.0,
                    float(voxel_size['z']) if voxel_size['z'] > 0 else 1.0,
                )

        # Ensure spacing matches (1, x, y, z)
        spacing = (spacing[2], spacing[1], spacing[0])  # Adjust for z, y, x order



        return seg, {'spacing': spacing}


if __name__ == '__main__':
    # 创建一个测试 MRCIO 对象
    mrc_io = MRCIO()

    # # 创建一个随机的 3D NumPy 数组
    # test_data = np.random.rand(64, 64, 64).astype(np.float32)

    # # 写入测试文件，并设置 voxel_size 为 (17.14, 17.14, 17.14)
    # mrc_io.write_seg(test_data, 'test_seg.mrc', {'spacing': (17.14, 17.14, 17.14)})

    # # 读取测试文件
    # try:
    #     read_data, metadata = mrc_io.read_seg('test_seg.mrc')
    #     assert np.array_equal(test_data, read_data[0]), "数据读取与写入不一致!"
    #     print(f"MRCIO 读取与写入测试通过! Metadata: {metadata}")
    # except Exception as e:
    #     print(f"测试失败: {e}")
    read_data, metadata = mrc_io.read_seg('/home/liushuo/Documents/data/nnUNet/nnUNet_raw/Dataset001_Vesicle/labelsTr/vesicle_001.mrc')
    print(metadata)