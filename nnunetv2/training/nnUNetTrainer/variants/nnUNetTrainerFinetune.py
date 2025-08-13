import os
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
                 head_weights: str | dict[str, str] | None = None):
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
        if path is not None and os.path.isfile(path):
            sd = torch.load(path, map_location=self.device, weights_only=True)
            msd = module.state_dict()
            filtered_sd = {}
            dropped = []
            for k, v in sd.items():
                if k in msd and msd[k].shape == v.shape:
                    filtered_sd[k] = v
                else:
                    dropped.append(k)
            if dropped:
                self.print_to_log_file(
                    f"Skipped loading weights for keys with incompatible shapes: {dropped}",
                    also_print_to_console=True,
                )
            missing, unexpected = module.load_state_dict(filtered_sd, strict=False)
            if missing:
                self.print_to_log_file(f"Missing keys when loading weights: {missing}")
            if unexpected:
                self.print_to_log_file(f"Unexpected keys when loading weights: {unexpected}")

    def _load_head_weights(self, mod: nn.Module):
        if self.head_weights is None:
            return
        if hasattr(mod, 'heads'):
            mapping: dict[str, str] = {}
            if isinstance(self.head_weights, str) and os.path.isdir(self.head_weights):
                for name in mod.heads.keys():
                    file = os.path.join(self.head_weights, f'{name}_head.pth')
                    if os.path.isfile(file):
                        mapping[name] = file
            elif isinstance(self.head_weights, dict):
                mapping = self.head_weights
            elif isinstance(self.head_weights, str) and os.path.isfile(self.head_weights):
                # single file for all heads not supported, ignore
                mapping = {}
            else:
                mapping = {}
            for name, file in mapping.items():
                if name in mod.heads and os.path.isfile(file):
                    self._load_module_weights(mod.heads[name], file)
        elif hasattr(mod, 'head'):
            if isinstance(self.head_weights, str):
                self._load_module_weights(mod.head, self.head_weights)

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
            self._load_head_weights(mod)

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

            # determine target directory based on checkpoint type
            subdir = (
                'best_component_pth'
                if filename.endswith('checkpoint_best.pth')
                else 'last_component_pth'
            )
            target_dir = os.path.join(self.output_folder, subdir)
            os.makedirs(target_dir, exist_ok=True)

            torch.save(mod.encoder.state_dict(), os.path.join(target_dir, 'encoder.pth'))
            torch.save(mod.decoder.state_dict(), os.path.join(target_dir, 'decoder.pth'))
            if hasattr(mod, 'heads'):
                for name, head in mod.heads.items():
                    torch.save(head.state_dict(), os.path.join(target_dir, f'{name}_head.pth'))
            elif hasattr(mod, 'head'):
                torch.save(mod.head.state_dict(), os.path.join(target_dir, 'head.pth'))

