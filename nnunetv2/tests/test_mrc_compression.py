"""Check compressed MRC round trips, format discovery, and mixed training paths."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import mrcfile
import numpy as np
import py7zr

from nnunetv2.imageio.mrc_reader_writer import MRCIO
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_file_ending
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets


class TestMRCCompression(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.rw = MRCIO()
        self.data = np.zeros((5, 8, 11), dtype=np.float32)
        self.data[1:4, 2:5, 3:9] = 1
        self.spacing = (6.0, 3.0, 2.0)

    def test_plain_and_compressed_round_trip(self):
        for suffix in ('.mrc', '.mrc.7z', '.MRC.7Z'):
            with self.subTest(suffix=suffix):
                path = self.folder / ('case' + suffix)
                self.rw.write_seg(self.data, path, {'spacing': self.spacing})
                for reader in (self.rw.read_seg, lambda p: self.rw.read_images([p])):
                    actual, properties = reader(path)
                    np.testing.assert_array_equal(actual, self.data[None])
                    self.assertEqual(actual.dtype, np.float32)
                    self.assertEqual(properties['spacing'], self.spacing)
                if suffix.lower().endswith('.7z'):
                    with py7zr.SevenZipFile(path) as archive:
                        self.assertEqual(archive.getnames(), [path.name[:-3]])

    def test_renamed_archive_with_nested_int8_member(self):
        source = self.folder / 'original.mrc'
        with mrcfile.new(source) as volume:
            volume.set_data(self.data.astype(np.int8))
            volume.voxel_size = (2.0, 3.0, 6.0)
        archive_path = self.folder / 'renamed.mrc.7z'
        with py7zr.SevenZipFile(archive_path, 'w') as archive:
            archive.write(source, 'nested/original.mrc')
        actual, properties = self.rw.read_images([archive_path])
        np.testing.assert_array_equal(actual, self.data[None])
        self.assertEqual(properties['spacing'], self.spacing)
        self.assertFalse((self.folder / 'nested').exists())

    def test_registry_and_compound_suffix_discovery(self):
        (self.folder / 'imagesTr').mkdir()
        (self.folder / 'labelsTr').mkdir()
        image = self.folder / 'imagesTr/case_0000.mrc.7z'
        label = self.folder / 'labelsTr/case.mrc.7z'
        for path in (image, label):
            self.rw.write_seg(self.data, path, {'spacing': self.spacing})
        self.assertIs(determine_reader_writer_from_file_ending('.mrc.7z', str(image), verbose=False), MRCIO)
        dataset = get_filenames_of_train_images_and_targets(str(self.folder), {'file_ending': '.mrc.7z'})
        self.assertEqual(dataset, {'case': {'images': [str(image)], 'label': str(label)}})

    def test_explicit_mapping_accepts_plain_image_and_compressed_label(self):
        image = self.folder / 'case_0000.mrc'
        label = self.folder / 'case.mrc.7z'
        for path in (image, label):
            self.rw.write_seg(self.data, path, {'spacing': self.spacing})
        config = {'file_ending': '.mrc.7z', 'dataset': {
            'case': {'images': [image.name], 'label': label.name}}}
        (self.folder / 'dataset.json').write_text(json.dumps(config))
        case = get_filenames_of_train_images_and_targets(str(self.folder))['case']
        images, image_properties = self.rw.read_images(case['images'])
        labels, label_properties = self.rw.read_seg(case['label'])
        np.testing.assert_array_equal(images, labels)
        self.assertEqual(image_properties, label_properties)

    def test_multiple_or_missing_mrc_members_are_rejected(self):
        source = self.folder / 'source.mrc'
        self.rw.write_seg(self.data, source, {})
        for members in (['a.mrc', 'b.mrc'], ['readme.txt']):
            archive_path = self.folder / 'ambiguous.mrc.7z'
            with py7zr.SevenZipFile(archive_path, 'w') as archive:
                for name in members:
                    archive.write(source, name)
            with self.assertRaisesRegex(ValueError, 'exactly one MRC'):
                self.rw.read_seg(archive_path)

    def test_mixed_channels_and_mismatched_geometry(self):
        first, second = self.folder / 'a.mrc', self.folder / 'b.mrc.7z'
        self.rw.write_seg(self.data, first, {'spacing': self.spacing})
        self.rw.write_seg(self.data + 3, second, {'spacing': self.spacing})
        images, _ = self.rw.read_images([first, second])
        np.testing.assert_array_equal(images, np.stack([self.data, self.data + 3]))
        self.rw.write_seg(self.data, second, {'spacing': (1, 1, 1)})
        with self.assertRaisesRegex(RuntimeError, 'spacings'):
            self.rw.read_images([first, second])
        self.rw.write_seg(self.data[:2], second, {'spacing': self.spacing})
        with self.assertRaisesRegex(RuntimeError, 'shapes'):
            self.rw.read_images([first, second])

    def test_failed_compression_preserves_destination(self):
        target = self.folder / 'preserved.mrc.7z'
        target.write_bytes(b'existing destination')
        with patch.object(py7zr.SevenZipFile, 'write', side_effect=OSError('write failed')):
            with self.assertRaises(OSError):
                self.rw.write_seg(self.data, target, {})
        self.assertEqual(target.read_bytes(), b'existing destination')
        self.assertEqual(list(self.folder.iterdir()), [target])

    def test_2d_archive_is_rejected(self):
        source = self.folder / 'slice.mrc'
        with mrcfile.new(source) as volume:
            volume.set_data(self.data[0])
        target = self.folder / 'slice.mrc.7z'
        with py7zr.SevenZipFile(target, 'w') as archive:
            archive.write(source, source.name)
        with self.assertRaisesRegex(RuntimeError, '3D'):
            self.rw.read_seg(target)


if __name__ == '__main__':
    unittest.main()
