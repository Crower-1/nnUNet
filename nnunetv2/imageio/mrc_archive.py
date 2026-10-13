"""Open single-volume 7z archives in memory and atomically archive MRC files."""

from contextlib import contextmanager
from io import BytesIO
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from mrcfile.mrcinterpreter import MrcInterpreter


@contextmanager
def open_compressed_mrc(filename):
    """Read one unambiguous MRC member without extracting archive paths to disk."""
    import py7zr
    from py7zr.io import BytesIOFactory

    with py7zr.SevenZipFile(filename, mode='r') as archive:
        members = [member for member in archive.list()
                   if member.is_file and member.filename.lower().endswith('.mrc')]
        if len(members) != 1:
            raise ValueError(f'Expected exactly one MRC volume in {filename}; found {len(members)}')
        member = members[0]
        factory = BytesIOFactory(limit=member.uncompressed)
        archive.extract(targets=[member.filename], factory=factory)
        payload = factory.get(member.filename).read()
    del factory
    with BytesIO(payload) as stream:
        del payload
        with MrcInterpreter(stream, permissive=True) as volume:
            yield volume


def write_compressed_mrc(mrc_path, output_fname):
    """Write a single MRC member, replacing the destination only after success."""
    import py7zr

    destination = Path(output_fname)
    with NamedTemporaryFile(dir=destination.parent, prefix=f'.{destination.name}.',
                            suffix='.tmp', delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with py7zr.SevenZipFile(temporary_path, mode='w',
                               filters=[{'id': py7zr.FILTER_LZMA2, 'preset': 1}]) as archive:
            archive.write(mrc_path, arcname=destination.name[:-3])
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
