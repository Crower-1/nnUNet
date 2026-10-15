"""CPU DDP smoke test with different locally present geometry classes."""
import tempfile
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from nnunetv2.utilities.label_geometry import resolve_geometry, geometry_options
from nnunetv2.training.loss.geometry_wrapper import GeometryLossWrapper


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(1)
        model = DistributedDataParallel(torch.nn.Conv3d(1, 3, 1))
        targets = [{'label_names': ['actin'], 'geometry_type': 'filament'},
                   {'label_names': ['membrane'], 'geometry_type': 'membrane'}]
        metadata = resolve_geometry({'schema_version': 1, 'targets': targets},
                                    {'background': 0, 'actin': 1, 'membrane': 2})
        options = geometry_options({'enabled': True, 'membrane_kernel': 3, 'membrane_iterations': 1})
        output = model(torch.ones(1, 1, 17, 17, 17))
        target = torch.zeros(1, 1, 17, 17, 17, dtype=torch.long)
        if rank == 0:
            target[:, :, 8, 8, 3:14] = 1
        else:
            target[:, :, 8, 3:14, 3:14] = 2
        loss = GeometryLossWrapper(lambda o, t: o.sum()*0, metadata, options)(output, target)
        loss.backward()
        grad = model.module.weight.grad
        gathered = [torch.zeros_like(grad) for _ in range(2)]
        dist.all_gather(gathered, grad)
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
        assert torch.equal(gathered[0], gathered[1])
    finally:
        dist.destroy_process_group()


class DDPTests(unittest.TestCase):
    def test_rank_local_missing_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(worker, args=(directory + '/rendezvous',), nprocs=2, join=True)
