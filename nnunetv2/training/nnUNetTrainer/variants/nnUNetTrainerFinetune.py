import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerFinetune(nnUNetTrainer):
    """Trainer supporting different fine-tuning modes."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'), *,
                 finetune_mode: str = 'scratch',
                 encoder_weights: str | None = None,
                 decoder_weights: str | None = None,
                 head_weights: str | None = None):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.finetune_mode = finetune_mode
        self.encoder_weights = encoder_weights
        self.decoder_weights = decoder_weights
        self.head_weights = head_weights
        self.my_init_kwargs.update({
            'finetune_mode': finetune_mode,
            'encoder_weights': encoder_weights,
            'decoder_weights': decoder_weights,
            'head_weights': head_weights,
        })

    def _load_module_weights(self, module: nn.Module, path: str | None):
        if path is not None:
            sd = torch.load(path, map_location=self.device)
            module.load_state_dict(sd)

    def _freeze_module(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    def initialize(self):
        if not self.was_initialized:
            super().initialize()
            # unwrap if compiled/DDP
            if self.is_ddp:
                mod = self.network.module
            else:
                mod = self.network
            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod

            # load pretrained weights
            self._load_module_weights(mod.encoder, self.encoder_weights)
            self._load_module_weights(mod.decoder, self.decoder_weights)
            if hasattr(mod, 'head'):
                self._load_module_weights(mod.head, self.head_weights)

            # freeze according to mode
            mode = self.finetune_mode.lower()
            if mode == 'head':
                self._freeze_module(mod.encoder)
                self._freeze_module(mod.decoder)
            elif mode == 'decoder_head':
                self._freeze_module(mod.encoder)
            elif mode in ('all', 'scratch'):
                pass
            else:
                raise ValueError(f'Unknown finetune_mode {self.finetune_mode}')

            # reconfigure optimizer with correct parameters
            self.optimizer, self.lr_scheduler = self.configure_optimizers()
        else:
            raise RuntimeError('initialize called twice')

    def save_checkpoint(self, filename: str) -> None:
        super().save_checkpoint(filename)
        if self.local_rank == 0 and not self.disable_checkpointing:
            if self.is_ddp:
                mod = self.network.module
            else:
                mod = self.network
            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod
            torch.save(mod.encoder.state_dict(), self.output_folder + '/encoder.pth')
            torch.save(mod.decoder.state_dict(), self.output_folder + '/decoder.pth')
            if hasattr(mod, 'head'):
                torch.save(mod.head.state_dict(), self.output_folder + '/head.pth')

