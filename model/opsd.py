# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from torch import nn
import torch

from pipeline.opsd_streaming_training import OPSDStreamingTrainingPipeline
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class OPSDModel(nn.Module):
    """Minimal model for OPSD streaming training: generator + text encoder + VAE."""

    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = bool(getattr(args, "independent_first_frame", False))
        self.same_step_across_blocks = bool(getattr(args, "same_step_across_blocks", True))
        self.last_step_only = bool(getattr(args, "last_step_only", False))

        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}),
            is_causal=True,
        )
        self.generator.model.requires_grad_(True)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        model_root = getattr(args, "model_kwargs", {}).get("model_root", None)
        self.text_encoder = None
        if bool(getattr(args, "use_text_encoder", True)):
            self.text_encoder = WanTextEncoder(model_root=model_root)
            self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper(model_root=model_root)
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

        self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
        if getattr(args, "warp_denoising_step", False):
            timesteps = torch.cat(
                (self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
            )
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.inference_pipeline = None

    def _initialize_inference_pipeline(self):
        local_attn_size = getattr(self.args, "model_kwargs", {}).get("local_attn_size", -1)
        slice_last_frames = int(getattr(self.args, "slice_last_frames", 21))
        self.inference_pipeline = OPSDStreamingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            same_step_across_blocks=self.same_step_across_blocks,
            last_step_only=self.last_step_only,
            context_noise=int(getattr(self.args, "context_noise", 0)),
            local_attn_size=local_attn_size,
            slice_last_frames=slice_last_frames,
        )
