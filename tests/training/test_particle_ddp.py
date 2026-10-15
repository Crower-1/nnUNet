"""DDP particle gradients must equal the same single-device global batch, even for unequal local sizes."""
import tempfile
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.utilities.label_geometry import resolve_geometry, geometry_options
from nnunetv2.training.loss.geometry_wrapper import GeometryLossWrapper
from nnunetv2.training.loss.particle_losses import particle_class_loss


def unequal_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(12)
        model = DDP(torch.nn.Conv3d(1, 3, 1))
        reference = torch.nn.Conv3d(1, 3, 1)
        reference.load_state_dict(model.module.state_dict())
        data = torch.randn(3, 1, 7, 7, 7)
        labels = torch.zeros(3, 1, 7, 7, 7, dtype=torch.long)
        # Rank 0 has one empty patch; rank 1 has two patches and two classes.
        labels[1, 0, 2, 2, 2] = 1
        labels[2, 0, 4, 4, 4] = 2
        targets = [dict(label_names=[n], geometry_type='particle') for n in ('ribo', 'rare')]
        metadata = resolve_geometry({'schema_version': 1, 'targets': targets}, {'background': 0, 'ribo': 1, 'rare': 2})
        options = geometry_options({'enabled': True, 'particle': {'class_weights': {'ribo': 1., 'rare': 2.}}})
        sl = slice(0, 1) if rank == 0 else slice(1, 3)
        output = model(data[sl])
        loss = GeometryLossWrapper(lambda o, t: o.sum()*0, metadata, options)(
            output, labels[sl], particle_valid=torch.ones_like(labels[sl], dtype=torch.bool))
        loss.backward()
        global_output = reference(data)
        expected = global_output.sum()*0
        for entry, weight in zip(metadata['targets'], (1., 2.)):
            values, _ = particle_class_loss(global_output, labels, torch.ones_like(labels, dtype=torch.bool),
                                            entry, options['particle'])
            expected = expected + .25*weight/3*values.mean()
        expected.backward()
        assert torch.allclose(loss, expected, atol=1e-6)
        for a, b in zip(model.module.parameters(), reference.parameters()):
            assert torch.allclose(a.grad, b.grad, atol=1e-6), (a.grad, b.grad)
    finally:
        dist.destroy_process_group()


class ParticleDDPTests(unittest.TestCase):
    def test_unequal_batches_global_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(unequal_worker, args=(directory+'/rendezvous',), nprocs=2, join=True)
