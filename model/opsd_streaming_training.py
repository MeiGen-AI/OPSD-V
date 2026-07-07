# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
import random
import os
import torch
import torch.distributed as dist
from typing import Tuple, Dict, Any, Optional, Callable
from einops import rearrange
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torchvision.io import write_video

from utils.debug_option import DEBUG
from pipeline.opsd_streaming_training import OPSDStreamingTrainingPipeline


class OPSDStreamingTrainingModel:
    """
    A model wrapper for OPSD streaming training.

    This class owns the OPSD rollout state, student/teacher KV caches, teacher
    EMA or LoRA adapter swapping, and chunk-wise rollout/debug utilities.
    """
    
    def __init__(self, base_model, config):
        """
        Initialize the streaming training model.

        Args:
            base_model: underlying model (DMD, DMDSwitch, etc.)
            config: configuration object
        """
        self.base_model = base_model
        self.config = config
        self.device = base_model.device
        self.dtype = base_model.dtype
        self.image_or_video_shape = getattr(config, 'image_or_video_shape', None)
        
        # Streaming training configuration
        self.chunk_size = getattr(config, "streaming_chunk_size", 21)  # Fixed chunk size used for loss computation
        self.max_length = getattr(config, "streaming_max_length", 57)
        self.possible_max_length = getattr(config, "streaming_possible_max_length", None)
        self.min_new_frame = getattr(config, "streaming_min_new_frame", 18)
        self.opsd_training = getattr(config, "distribution_loss", "") == "opsd_streaming"
        self.opsd_loss_type = getattr(config, "opsd_loss_type", "flow")
        self.opsd_loss_frames = int(getattr(config, "opsd_loss_frames", 42))
        self.opsd_max_loss_frames = int(getattr(config, "opsd_max_loss_frames", 0))
        self.opsd_loss_start_frame = max(0, int(getattr(config, "opsd_loss_start_frame", 21)))
        self.opsd_use_gt_first_chunk = bool(getattr(config, "opsd_use_gt_first_chunk", True))
        self.opsd_teacher_context_mode = getattr(config, "opsd_teacher_context_mode", "gt_kv")
        self.opsd_student_context_mode = getattr(config, "opsd_student_context_mode", "generated_kv")
        self.opsd_uniform_timestep = bool(getattr(config, "opsd_uniform_timestep", True))
        self.opsd_loss_step_mode = getattr(config, "opsd_loss_step_mode", "single")
        self.opsd_loss_flag = int(getattr(config, "opsd_loss_flag", -1))
        self.opsd_shared_flag_per_window = bool(getattr(config, "opsd_shared_flag_per_window", True))
        self.opsd_use_relative_sink = bool(getattr(config, "opsd_use_relative_sink", False))
        self.opsd_teacher_use_future_gt_context = bool(
            getattr(config, "opsd_teacher_use_future_gt_context", False)
        )
        self.opsd_teacher_future_gt_num_chunks = max(
            0,
            int(getattr(config, "opsd_teacher_future_gt_num_chunks", 1)),
        )
        self.opsd_teacher_trajectory_mode = str(
            getattr(config, "opsd_teacher_trajectory_mode", "student")
        ).lower()
        if self.opsd_teacher_trajectory_mode not in {"student", "teacher"}:
            raise ValueError("opsd_teacher_trajectory_mode must be 'student' or 'teacher'")

        # Get required components from the underlying model
        self.generator = base_model.generator
        self.scheduler = base_model.scheduler
        
        # Fetch model configuration
        self.num_frame_per_block = base_model.num_frame_per_block
        self.frame_seq_length = getattr(base_model.inference_pipeline, 'frame_seq_length', 1560)
        
        # Initialize inference pipeline
        self.inference_pipeline = base_model.inference_pipeline
        if self.inference_pipeline is None:
            base_model._initialize_inference_pipeline()
            self.inference_pipeline = base_model.inference_pipeline
        if not isinstance(self.inference_pipeline, OPSDStreamingTrainingPipeline):
            self.inference_pipeline = self._build_opsd_inference_pipeline(base_model)
            base_model.inference_pipeline = self.inference_pipeline
        
        # Streaming state
        self.reset_state()  
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] streamingTrainingModel initialized:")
            print(f"[StreamingTrain-Model] chunk_size={self.chunk_size}, max_length={self.max_length}")
            print(f"[StreamingTrain-Model] min_new_frame={self.min_new_frame}")
            print(f"[StreamingTrain-Model] base_model type: {type(self.base_model).__name__}")

    def _build_opsd_inference_pipeline(self, base_model):
        local_attn_size = getattr(base_model.args, "model_kwargs", {}).get("local_attn_size", -1)
        slice_last_frames = getattr(base_model.args, "slice_last_frames", 21)
        return OPSDStreamingTrainingPipeline(
            denoising_step_list=base_model.denoising_step_list,
            scheduler=base_model.scheduler,
            generator=base_model.generator,
            num_frame_per_block=base_model.num_frame_per_block,
            same_step_across_blocks=base_model.args.same_step_across_blocks,
            last_step_only=base_model.args.last_step_only,
            context_noise=base_model.args.context_noise,
            local_attn_size=local_attn_size,
            slice_last_frames=slice_last_frames,
        )

    def reset_state(self):
        """Reset streaming training state"""
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Resetting streaming training state")
            
        self.state = {
            "current_length": 0,
            "conditional_info": None,
            "has_switched": False,  # Track whether prompt has been switched
            "previous_frames": None,  # Store last generated frames for overlap (up to 21)
            "temp_max_length": None,  # Temporary max length for the current sequence
        }

        self.inference_pipeline.clear_kv_cache()

        self.teacher_kv_cache1 = None
        self.teacher_crossattn_cache = None
        self.student_kv_cache1 = None
        self.student_crossattn_cache = None

    def _get_current_conditional_dict(self, chunk_start_frame: int) -> dict:
        """Get the conditional_dict to use for the current chunk"""
        cond_info = self.state["conditional_info"]
        
        # Check whether it has switched already or should switch now
        switch_info = cond_info.get("switch_info", {})
        if switch_info:
            switch_frame_index = switch_info.get("switch_frame_index")
            if switch_frame_index is not None:
                if self.state.get("has_switched", False) or chunk_start_frame >= switch_frame_index:
                    # If already switched, or current frame has reached the switch point, use the switched prompt
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[StreamingTrain-Model] Using switch conditional_dict for chunk starting at frame {chunk_start_frame}")
                    return switch_info.get("switch_conditional_dict", cond_info["conditional_dict"])
        
        # Otherwise use the original prompt
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Using original conditional_dict for chunk starting at frame {chunk_start_frame}")
        return cond_info["conditional_dict"]

    def _build_empty_kv_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        kv_cache = []
        num_heads, head_dim = self._attention_cache_shape()
        kv_cache_size = int(self.inference_pipeline.kv_cache_size)
        for _ in range(self._num_transformer_blocks()):
            kv_cache.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })
        return kv_cache

    def _build_empty_crossattn_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        crossattn_cache = []
        num_heads, head_dim = self._attention_cache_shape()
        for _ in range(self._num_transformer_blocks()):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False,
            })
        return crossattn_cache

    def _generator_backbone(self):
        wrapper = getattr(self.generator, "module", self.generator)
        backbone = getattr(wrapper, "model", wrapper)
        if hasattr(backbone, "get_base_model"):
            backbone = backbone.get_base_model()
        return backbone

    def _attention_cache_shape(self):
        backbone = self._generator_backbone()
        num_heads = int(getattr(backbone, "num_heads"))
        dim = int(getattr(backbone, "dim"))
        return num_heads, dim // num_heads

    def _num_transformer_blocks(self):
        backbone = self._generator_backbone()
        if hasattr(backbone, "blocks"):
            return len(backbone.blocks)
        return int(getattr(self.inference_pipeline, "num_transformer_blocks"))

    @staticmethod
    def _set_empty_cache_global_start(kv_cache_lists: list, global_start_index: int):
        for kv_cache in kv_cache_lists:
            for cache in kv_cache:
                cache["global_end_index"].fill_(global_start_index)
                cache["local_end_index"].zero_()

    @staticmethod
    def _detach_cache_tensors(cache_lists: list):
        for cache_list in cache_lists:
            if cache_list is None:
                continue
            for cache in cache_list:
                for key, value in list(cache.items()):
                    if torch.is_tensor(value) and value.requires_grad:
                        cache[key] = value.detach()

    def set_opsd_teacher_ema_shadow(self, ema_shadow: Optional[dict]):
        self.opsd_teacher_ema_shadow = ema_shadow

    def _active_lora_adapter(self):
        wrapper = getattr(self.generator, "module", self.generator)
        lora_model = getattr(wrapper, "model", None)
        if lora_model is None or not hasattr(lora_model, "set_adapter"):
            return None, None
        active = getattr(lora_model, "active_adapter", None)
        if active is None:
            active_adapters = getattr(lora_model, "active_adapters", None)
            if isinstance(active_adapters, (list, tuple)) and active_adapters:
                active = active_adapters[0]
        return lora_model, active

    @staticmethod
    def _clean_fsdp_param_name(name: str) -> str:
        return name.replace("_fsdp_wrapped_module.", "")

    def _swap_generator_to_ema(self):
        debug_swap = bool(getattr(self.config, "opsd_debug_teacher_swap", False))
        debug_rank0 = not dist.is_initialized() or dist.get_rank() == 0
        debug_prints = int(getattr(self, "_opsd_debug_teacher_swap_prints", 0))
        debug_max_prints = int(getattr(self.config, "opsd_debug_teacher_swap_max_prints", 20))
        should_print_debug = debug_swap and debug_rank0 and debug_prints < debug_max_prints

        lora_model, active_adapter = self._active_lora_adapter()
        if lora_model is not None and "teacher" in getattr(lora_model, "peft_config", {}):
            lora_model.set_adapter("teacher")
            if should_print_debug:
                print(
                    "[TEACHER SWAP DEBUG] mode=lora_adapter "
                    f"active_before={active_adapter} switched_to=teacher"
                )
                self._opsd_debug_teacher_swap_prints = debug_prints + 1
            return ("adapter", active_adapter or "student")
        if not self.opsd_teacher_ema_shadow:
            if should_print_debug:
                print("[TEACHER SWAP DEBUG] mode=student_fallback reason=no_ema_shadow")
                self._opsd_debug_teacher_swap_prints = debug_prints + 1
            if bool(getattr(self.config, "opsd_require_teacher_ema", False)):
                raise RuntimeError(
                    "OPSD teacher EMA shadow is missing, so teacher would fall back to student. "
                    "Check Trainer.generator_ema initialization and ema_weight/ema_start_step."
                )
            return None
        backup = {}
        matched = 0
        missing = 0
        max_student_ema_diff = 0.0
        with FSDP.summon_full_params(self.generator, writeback=True, rank0_only=False):
            for raw_name, p in self.generator.named_parameters():
                n = self._clean_fsdp_param_name(raw_name)
                backup[n] = p.detach().cpu().clone()
                ema_v = self.opsd_teacher_ema_shadow.get(n, None)
                if ema_v is not None:
                    matched += 1
                    if should_print_debug:
                        diff = (backup[n].float() - ema_v.float()).abs().max().item()
                        max_student_ema_diff = max(max_student_ema_diff, diff)
                    p.data.copy_(ema_v.to(device=p.device, dtype=p.dtype))
                else:
                    missing += 1
        if should_print_debug:
            print(
                "[TEACHER SWAP DEBUG] mode=ema "
                f"matched={matched}/{len(backup)} missing={missing} "
                f"shadow_keys={len(self.opsd_teacher_ema_shadow)} "
                f"max_student_ema_diff={max_student_ema_diff:.6e}"
            )
            self._opsd_debug_teacher_swap_prints = debug_prints + 1
        return backup

    def _restore_generator_from_backup(self, backup: Optional[dict]):
        if backup is None:
            return
        if isinstance(backup, tuple) and backup[0] == "adapter":
            lora_model, _ = self._active_lora_adapter()
            if lora_model is not None and backup[1] is not None:
                lora_model.set_adapter(backup[1])
            return
        with FSDP.summon_full_params(self.generator, writeback=True, rank0_only=False):
            for raw_name, p in self.generator.named_parameters():
                n = self._clean_fsdp_param_name(raw_name)
                src = backup.get(n, None)
                if src is not None:
                    p.data.copy_(src.to(device=p.device, dtype=p.dtype))

    def _prepare_relative_sink_mode(self, kv_cache_list: list):
        try:
            sink_tokens = int(self._generator_backbone().blocks[0].self_attn.sink_size) * int(self.frame_seq_length)
        except Exception:
            sink_tokens = 0
        for cache in kv_cache_list:
            cache["use_relative_sink"] = True
            cache["dynamic_sink_alpha"] = 0.0
            cache["history_only_after_first_gt"] = False
            cache["capture_sink_raw"] = True
            cache["capture_sink_raw_any_start"] = True
            cache.pop("persistent_sink_raw_k", None)
            cache.pop("persistent_sink_v", None)
            cache.pop("persistent_sink_num_frames", None)
            cache.pop("captured_sink_raw_k", None)
            cache.pop("captured_sink_v", None)
            cache.pop("captured_sink_num_frames", None)
            cache.pop("captured_sink_raw_k_buf", None)
            cache.pop("captured_sink_v_buf", None)
            cache.pop("captured_sink_num_frames_buf", None)
            cache.pop("dynamic_sink_raw_k", None)
            cache.pop("dynamic_sink_v", None)
            if sink_tokens > 0 and cache["k"].shape[1] >= sink_tokens:
                cache["captured_sink_raw_k_buf"] = torch.empty_like(cache["k"][:, :sink_tokens])
                cache["captured_sink_v_buf"] = torch.empty_like(cache["v"][:, :sink_tokens])
                cache["captured_sink_num_frames_buf"] = torch.zeros(
                    (), device=cache["local_end_index"].device, dtype=torch.int64
                )

    def _finalize_relative_sink_capture(self, kv_cache_list: list):
        missing_cache_indices = []
        for cache_idx, cache in enumerate(kv_cache_list):
            sink_raw_k = cache.pop("captured_sink_raw_k", None)
            sink_v = cache.pop("captured_sink_v", None)
            sink_num_frames = cache.pop("captured_sink_num_frames", None)
            sink_raw_k_buf = cache.pop("captured_sink_raw_k_buf", None)
            sink_v_buf = cache.pop("captured_sink_v_buf", None)
            sink_num_frames_buf = cache.pop("captured_sink_num_frames_buf", None)
            cache.pop("capture_sink_raw", None)
            cache.pop("capture_sink_raw_any_start", None)
            if (
                sink_raw_k is None
                and sink_v is None
                and sink_num_frames is None
                and sink_raw_k_buf is not None
                and sink_v_buf is not None
                and sink_num_frames_buf is not None
                and int(sink_num_frames_buf.item()) > 0
            ):
                sink_raw_k = sink_raw_k_buf.detach().clone()
                sink_v = sink_v_buf.detach().clone()
                sink_num_frames = int(sink_num_frames_buf.item())
            if sink_raw_k is not None and sink_v is not None and sink_num_frames is not None:
                cache["persistent_sink_raw_k"] = sink_raw_k
                cache["persistent_sink_v"] = sink_v
                cache["persistent_sink_num_frames"] = int(sink_num_frames)
            else:
                missing_cache_indices.append(cache_idx)
        if missing_cache_indices:
            raise RuntimeError(
                "Failed to capture OPSD relative sink. "
                f"missing_cache_indices={missing_cache_indices} "
                f"captured={len(kv_cache_list) - len(missing_cache_indices)}/{len(kv_cache_list)}"
            )

    def _replace_cache_chunk_with_gt(
        self,
        kv_cache: list,
        crossattn_cache: list,
        gt_chunk_latents: torch.Tensor,
        conditional_dict: dict,
        chunk_start_frame: int,
        use_teacher_ema: bool = False,
    ):
        saved_end_indices = [
            (cache["global_end_index"].clone(), cache["local_end_index"].clone())
            for cache in kv_cache
        ]
        backup_kv = self.inference_pipeline.kv_cache1
        backup_cross = self.inference_pipeline.crossattn_cache
        self.inference_pipeline.kv_cache1 = kv_cache
        self.inference_pipeline.crossattn_cache = crossattn_cache
        timestep = torch.ones(
            [gt_chunk_latents.shape[0], gt_chunk_latents.shape[1]],
            device=gt_chunk_latents.device,
            dtype=torch.int64,
        ) * self.inference_pipeline.context_noise
        self._write_chunk_to_cache(
            chunk_latents=gt_chunk_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            chunk_start_frame=chunk_start_frame,
            use_teacher_ema=use_teacher_ema,
        )
        for cache, (saved_global_end, saved_local_end) in zip(kv_cache, saved_end_indices):
            cache["global_end_index"] = saved_global_end
            cache["local_end_index"] = saved_local_end
        self.inference_pipeline.kv_cache1 = backup_kv
        self.inference_pipeline.crossattn_cache = backup_cross

    def _write_chunk_to_cache(
        self,
        chunk_latents: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: list,
        crossattn_cache: list,
        chunk_start_frame: int,
        use_teacher_ema: bool = False,
    ):
        teacher_backup = self._swap_generator_to_ema() if use_teacher_ema else None
        try:
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=chunk_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=chunk_start_frame * self.frame_seq_length,
                )
        finally:
            if teacher_backup is not None:
                self._restore_generator_from_backup(teacher_backup)

    @staticmethod
    def _clear_future_context(kv_cache: list):
        for cache in kv_cache:
            cache.pop("future_context_raw_k", None)
            cache.pop("future_context_v", None)
            cache.pop("future_context_num_frames", None)
            cache.pop("future_context_start_frame", None)

    def _capture_future_gt_context_for_cache(
        self,
        future_gt_latents: torch.Tensor,
        future_start_frame: int,
        conditional_dict: dict,
        target_kv_cache: list,
        use_teacher_ema: bool = False,
    ) -> bool:
        if future_gt_latents is None or future_gt_latents.shape[1] <= 0:
            self._clear_future_context(target_kv_cache)
            return False

        temp_kv_cache = []
        for cache in target_kv_cache:
            temp_kv_cache.append({
                "k": torch.zeros_like(cache["k"]),
                "v": torch.zeros_like(cache["v"]),
                "global_end_index": torch.zeros_like(cache["global_end_index"]),
                "local_end_index": torch.zeros_like(cache["local_end_index"]),
                "capture_context_raw": True,
            })

        temp_crossattn_cache = self._build_empty_crossattn_cache(
            batch_size=future_gt_latents.shape[0],
            dtype=future_gt_latents.dtype,
            device=future_gt_latents.device,
        )
        timestep = torch.ones(
            [future_gt_latents.shape[0], future_gt_latents.shape[1]],
            device=future_gt_latents.device,
            dtype=torch.int64,
        ) * self.inference_pipeline.context_noise

        self._write_chunk_to_cache(
            chunk_latents=future_gt_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=temp_kv_cache,
            crossattn_cache=temp_crossattn_cache,
            chunk_start_frame=0,
            use_teacher_ema=use_teacher_ema,
        )

        captured = True
        for src_cache, dst_cache in zip(temp_kv_cache, target_kv_cache):
            raw_k = src_cache.get("captured_context_raw_k")
            raw_v = src_cache.get("captured_context_v")
            num_frames = src_cache.get("captured_context_num_frames")
            if raw_k is None or raw_v is None or num_frames is None:
                captured = False
                dst_cache.pop("future_context_raw_k", None)
                dst_cache.pop("future_context_v", None)
                dst_cache.pop("future_context_num_frames", None)
                dst_cache.pop("future_context_start_frame", None)
            else:
                dst_cache["future_context_raw_k"] = raw_k
                dst_cache["future_context_v"] = raw_v
                dst_cache["future_context_num_frames"] = int(num_frames)
                dst_cache["future_context_start_frame"] = int(future_start_frame)
        return captured

    def setup_opsd_sequence(
        self,
        conditional_dict: Dict,
        unconditional_dict: Dict,
        gt_latents: torch.Tensor,
        window_start_override: Optional[int] = None,
    ):
        self.reset_state()
        self.state["conditional_info"] = {
            "conditional_dict": conditional_dict,
            "unconditional_dict": unconditional_dict,
        }
        if gt_latents is None:
            raise ValueError("OPSD streaming requires gt_latents in batch.")
        gt_latents = gt_latents.to(device=self.device, dtype=self.dtype)
        gt_total_frames = gt_latents.shape[1]
        block = int(self.num_frame_per_block)
        max_loss_frames = gt_total_frames if self.opsd_max_loss_frames <= 0 else min(gt_total_frames, self.opsd_max_loss_frames)
        max_loss_frames = max(block, (max_loss_frames // block) * block)
        desired_loss_frames = min(int(self.opsd_loss_frames), max_loss_frames)
        desired_loss_frames = max(block, (desired_loss_frames // block) * block)
        max_start = gt_total_frames - desired_loss_frames
        if max_start < 0:
            raise ValueError(f"Invalid OPSD window: gt_total_frames={gt_total_frames}, desired_loss_frames={desired_loss_frames}")
        if window_start_override is not None:
            window_start = int(window_start_override)
            if window_start < 0 or window_start > max_start:
                raise ValueError(
                    f"window_start_override={window_start} is invalid for "
                    f"gt_total_frames={gt_total_frames}, desired_loss_frames={desired_loss_frames}"
                )
            window_start = (window_start // block) * block
        elif max_start == 0:
            window_start = 0
        else:
            start_choices = list(range(0, max_start + 1, block))
            # Each rank may see a different video length, so the valid start
            # choices are local to that sample. Do not broadcast an index chosen
            # from rank0's list.
            selected_idx = random.randint(0, len(start_choices) - 1)
            window_start = start_choices[selected_idx]

        batch_size = gt_latents.shape[0]
        self.student_kv_cache1 = self._build_empty_kv_cache(batch_size=batch_size, dtype=self.dtype, device=self.device)
        self.student_crossattn_cache = self._build_empty_crossattn_cache(batch_size=batch_size, dtype=self.dtype, device=self.device)
        self.teacher_kv_cache1 = self._build_empty_kv_cache(batch_size=batch_size, dtype=self.dtype, device=self.device)
        self.teacher_crossattn_cache = self._build_empty_crossattn_cache(batch_size=batch_size, dtype=self.dtype, device=self.device)
        # The training window may start in the middle of a video. An empty cache
        # should still treat that frame as the current global origin.
        self._set_empty_cache_global_start(
            [self.student_kv_cache1, self.teacher_kv_cache1],
            int(window_start) * self.frame_seq_length,
        )
        if self.opsd_use_relative_sink:
            self._prepare_relative_sink_mode(self.student_kv_cache1)
            self._prepare_relative_sink_mode(self.teacher_kv_cache1)

        self.state["opsd_gt_latents"] = gt_latents
        self.state["opsd_window_start"] = int(window_start)
        self.state["opsd_window_frames"] = int(desired_loss_frames)
        self.state["opsd_student_chunks"] = []
        self.state["opsd_teacher_chunks"] = []
        self.state["opsd_generated_chunk_spans"] = []
        self.state["current_length"] = 0
        if self.opsd_use_relative_sink and not self.opsd_use_gt_first_chunk:
            raise ValueError("opsd_use_relative_sink requires opsd_use_gt_first_chunk=True to capture persistent sink.")

        if self.opsd_use_gt_first_chunk:
            first_chunk = gt_latents[:, window_start:window_start + block]
            timestep0 = torch.zeros([batch_size, first_chunk.shape[1]], device=self.device, dtype=torch.int64)
            self._write_chunk_to_cache(
                chunk_latents=first_chunk,
                conditional_dict=conditional_dict,
                timestep=timestep0,
                kv_cache=self.student_kv_cache1,
                crossattn_cache=self.student_crossattn_cache,
                chunk_start_frame=window_start,
            )
            self._write_chunk_to_cache(
                chunk_latents=first_chunk,
                conditional_dict=conditional_dict,
                timestep=timestep0,
                kv_cache=self.teacher_kv_cache1,
                crossattn_cache=self.teacher_crossattn_cache,
                chunk_start_frame=window_start,
                use_teacher_ema=True,
            )
            self.state["opsd_student_chunks"].append(first_chunk.detach())
            self.state["opsd_teacher_chunks"].append(first_chunk.detach())
            self.state["current_length"] = int(first_chunk.shape[1])
            if self.opsd_use_relative_sink:
                self._finalize_relative_sink_capture(self.student_kv_cache1)
                self._finalize_relative_sink_capture(self.teacher_kv_cache1)

    def compute_opsd_generator_loss(
        self,
        backward_per_chunk: bool = False,
        loss_scale: float = 1.0,
        post_backward_callback: Optional[Callable[[], None]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.state.get("opsd_gt_latents", None) is None:
            raise ValueError("OPSD sequence is not initialized. Please call setup_opsd_sequence first.")

        gt_latents = self.state["opsd_gt_latents"]
        window_start = int(self.state["opsd_window_start"])
        window_frames = int(self.state["opsd_window_frames"])
        current_length = int(self.state["current_length"])
        block = int(self.num_frame_per_block)
        conditional_dict = self._get_current_conditional_dict(window_start + current_length)
        denoising_steps = self.inference_pipeline.denoising_step_list
        num_denoising_steps = len(denoising_steps)
        if num_denoising_steps <= 0:
            raise ValueError("Empty denoising_step_list for OPSD training.")

        def _sample_flag_idx():
            if self.opsd_loss_flag >= 0:
                flag_idx = min(self.opsd_loss_flag, num_denoising_steps - 1)
                if dist.is_initialized():
                    flag_tensor = torch.tensor(flag_idx, device=self.device, dtype=torch.long)
                    dist.broadcast(flag_tensor, src=0)
                    flag_idx = int(flag_tensor.item())
            else:
                if dist.is_initialized():
                    if dist.get_rank() == 0:
                        flag_tensor = torch.randint(
                            low=0,
                            high=num_denoising_steps,
                            size=(1,),
                            device=self.device,
                            dtype=torch.long,
                        )
                    else:
                        flag_tensor = torch.empty((1,), device=self.device, dtype=torch.long)
                    dist.broadcast(flag_tensor, src=0)
                    flag_idx = int(flag_tensor.item())
                else:
                    flag_idx = int(torch.randint(low=0, high=num_denoising_steps, size=(1,), device=self.device).item())
            return flag_idx

        def _loss_step_indices(flag_idx):
            if self.opsd_loss_step_mode == "all":
                # Z-Image D-OPSD style: supervise every denoising step, while
                # transition dynamics below remain stop-gradient.
                return set(range(num_denoising_steps))
            if self.opsd_loss_step_mode == "prefix":
                # User-defined semantics:
                # flag=0 -> loss on step 0
                # flag=3 -> loss on steps 0,1,2
                prefix_len = 1 if flag_idx == 0 else min(flag_idx, num_denoising_steps)
                return set(range(prefix_len))
            if self.opsd_loss_step_mode == "single":
                # Self-forcing style: only selected step contributes gradient/loss.
                return {flag_idx}
            raise ValueError(
                f"Unsupported opsd_loss_step_mode={self.opsd_loss_step_mode}. "
                f"Use 'single', 'prefix', or 'all'."
            )

        remaining_frames = max(0, window_frames - current_length)
        total_chunks = (remaining_frames + block - 1) // block
        loss_schedules = []
        total_flag_value = 0
        total_loss_terms = 0
        total_supervised_frames = 0
        shared_flag_idx = (
            None
            if self.opsd_loss_step_mode == "all"
            else (_sample_flag_idx() if self.opsd_shared_flag_per_window else None)
        )
        for chunk_idx in range(total_chunks):
            flag_idx = -1 if self.opsd_loss_step_mode == "all" else (
                shared_flag_idx if shared_flag_idx is not None else _sample_flag_idx()
            )
            loss_step_indices = _loss_step_indices(flag_idx)
            scheduled_start = current_length + chunk_idx * block
            scheduled_num_frames = min(block, window_frames - scheduled_start)
            enable_chunk_loss = scheduled_start >= self.opsd_loss_start_frame
            loss_schedules.append((flag_idx, loss_step_indices, enable_chunk_loss))
            total_flag_value += flag_idx
            if enable_chunk_loss:
                total_loss_terms += len(loss_step_indices)
                total_supervised_frames += max(0, scheduled_num_frames)

        total_loss = None

        while current_length < window_frames:
            current_num_frames = min(block, window_frames - current_length)
            abs_chunk_start = window_start + current_length
            _, loss_step_indices, enable_chunk_loss = loss_schedules.pop(0)

            noise_chunk = torch.randn(
                [gt_latents.shape[0], current_num_frames, *gt_latents.shape[2:]],
                device=self.device,
                dtype=self.dtype,
            )
            noisy_input = noise_chunk
            teacher_noisy_input = noise_chunk
            student_chunk = None
            teacher_chunk = None

            # Keep local-attention setup consistent with pipeline path.
            if hasattr(self.inference_pipeline, "local_attn_size"):
                self.generator.model.local_attn_size = int(self.inference_pipeline.local_attn_size)
                if hasattr(self.inference_pipeline, "_set_all_modules_max_attention_size"):
                    self.inference_pipeline._set_all_modules_max_attention_size(int(self.inference_pipeline.local_attn_size))

            if self.opsd_teacher_use_future_gt_context and self.opsd_teacher_future_gt_num_chunks > 0:
                future_start = abs_chunk_start + current_num_frames
                available_future_frames = gt_latents.shape[1] - future_start
                available_future_chunks = max(0, available_future_frames // block)
                capture_future_chunks = min(
                    self.opsd_teacher_future_gt_num_chunks,
                    available_future_chunks,
                )
                if capture_future_chunks > 0:
                    future_num_frames = capture_future_chunks * block
                    future_gt_latents = gt_latents[:, future_start:future_start + future_num_frames].to(
                        device=self.device,
                        dtype=self.dtype,
                    )
                    captured_future = self._capture_future_gt_context_for_cache(
                        future_gt_latents=future_gt_latents,
                        future_start_frame=future_start,
                        conditional_dict=conditional_dict,
                        target_kv_cache=self.teacher_kv_cache1,
                        use_teacher_ema=True,
                    )
                    if not captured_future:
                        self._clear_future_context(self.teacher_kv_cache1)
                else:
                    self._clear_future_context(self.teacher_kv_cache1)
            else:
                self._clear_future_context(self.teacher_kv_cache1)

            for step_idx, current_timestep in enumerate(denoising_steps):
                step_cpu_rng_state = torch.random.get_rng_state()
                step_cuda_rng_state = torch.cuda.get_rng_state(self.device) if torch.cuda.is_available() else None
                timestep = torch.ones(
                    [gt_latents.shape[0], current_num_frames],
                    device=self.device,
                    dtype=torch.int64,
                ) * current_timestep
                need_step_loss = enable_chunk_loss and step_idx in loss_step_indices

                student_backup= self._swap_generator_to_ema()
                try:
                    with torch.no_grad():
                        teacher_flow, teacher_chunk = self.generator(
                            noisy_image_or_video=teacher_noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.teacher_kv_cache1,
                            crossattn_cache=self.teacher_crossattn_cache,
                            current_start=abs_chunk_start * self.frame_seq_length,
                        )
                finally:
                    self._restore_generator_from_backup(student_backup)

                torch.random.set_rng_state(step_cpu_rng_state)
                if step_cuda_rng_state is not None:
                    torch.cuda.set_rng_state(step_cuda_rng_state, device=self.device)
                student_ctx = torch.enable_grad() if need_step_loss else torch.no_grad()
                with student_ctx:
                    student_flow, student_chunk = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.student_kv_cache1,
                        crossattn_cache=self.student_crossattn_cache,
                        current_start=abs_chunk_start * self.frame_seq_length,
                    )

                if need_step_loss:
                    if self.opsd_loss_type == "flow":
                        step_loss = torch.mean((student_flow - teacher_flow.detach()) ** 2)
                    elif self.opsd_loss_type == "x0":
                        step_loss = torch.mean((student_chunk - teacher_chunk.detach()) ** 2)
                    else:
                        raise ValueError(
                            f"Unsupported opsd_loss_type={self.opsd_loss_type}. Use 'flow' or 'x0'."
                        )
                # if (
                #     step_idx == num_denoising_steps - 1
                #     and (not dist.is_initialized() or dist.get_rank() == 0)
                # ):
                #     with torch.no_grad():
                #         gt_chunk_for_debug = gt_latents[:, abs_chunk_start:abs_chunk_start + current_num_frames].to(
                #             device=student_chunk.device,
                #             dtype=student_chunk.dtype,
                #         )
                #         student_teacher_mse = torch.mean((student_chunk.detach() - teacher_chunk.detach()) ** 2)
                #         student_gt_mse = torch.mean((student_chunk.detach() - gt_chunk_for_debug) ** 2)
                #         teacher_gt_mse = torch.mean((teacher_chunk.detach() - gt_chunk_for_debug) ** 2)

                #         print(
                #             "[OPSD DEBUG] "
                #             f"abs_start={abs_chunk_start} "
                #             f"step_idx={step_idx} "
                #             f"t={int(current_timestep)} "
                #             f"student_teacher_mse={student_teacher_mse.item():.6f} "
                #             f"student_gt_mse={student_gt_mse.item():.6f} "
                #             f"teacher_gt_mse={teacher_gt_mse.item():.6f} "
                #             f"student_std={student_chunk.detach().float().std().item():.6f} "
                #             f"teacher_std={teacher_chunk.detach().float().std().item():.6f} "
                #             f"gt_std={gt_chunk_for_debug.detach().float().std().item():.6f} "
                #             f"student_mean={student_chunk.detach().float().mean().item():.6f} "
                #             f"teacher_mean={teacher_chunk.detach().float().mean().item():.6f} "
                #             f"gt_mean={gt_chunk_for_debug.detach().float().mean().item():.6f}"
                #         )                   
                if need_step_loss and backward_per_chunk:
                    (step_loss / float(total_loss_terms) * float(loss_scale)).backward()
                    if post_backward_callback is not None:
                        post_backward_callback()
                    self._detach_cache_tensors([
                        self.student_kv_cache1,
                        self.student_crossattn_cache,
                        self.teacher_kv_cache1,
                        self.teacher_crossattn_cache,
                    ])
                    step_loss_for_log = step_loss.detach()
                    total_loss = step_loss_for_log if total_loss is None else (total_loss + step_loss_for_log)
                elif need_step_loss:
                    total_loss = step_loss if total_loss is None else (total_loss + step_loss)

                if step_idx < num_denoising_steps - 1:
                    next_timestep = denoising_steps[step_idx + 1]
                    with torch.no_grad():
                        transition_noise = torch.randn_like(student_chunk.detach().flatten(0, 1))
                        # Match Algorithm 1's stop-gradient through transition dynamics.
                        noisy_input = self.scheduler.add_noise(
                            student_chunk.detach().flatten(0, 1),
                            transition_noise,
                            next_timestep * torch.ones(
                                [gt_latents.shape[0] * current_num_frames],
                                device=self.device,
                                dtype=torch.long,
                            ),
                        ).unflatten(0, student_chunk.shape[:2])
                        if self.opsd_teacher_trajectory_mode == "teacher":
                            teacher_noisy_input = self.scheduler.add_noise(
                                teacher_chunk.detach().flatten(0, 1),
                                transition_noise,
                                next_timestep * torch.ones(
                                    [gt_latents.shape[0] * current_num_frames],
                                    device=self.device,
                                    dtype=torch.long,
                                ),
                            ).unflatten(0, teacher_chunk.shape[:2])
                        else:
                            teacher_noisy_input = noisy_input

            if student_chunk is None or teacher_chunk is None:
                raise RuntimeError("OPSD chunk rollout produced no outputs.")

            # Student KV should always be updated with fully denoised chunk (no grad).
            context_timestep = torch.ones(
                [gt_latents.shape[0], current_num_frames],
                device=self.device,
                dtype=torch.int64,
            ) * self.inference_pipeline.context_noise
            context_noisy = self.scheduler.add_noise(
                student_chunk.detach().flatten(0, 1),
                torch.randn_like(student_chunk.detach().flatten(0, 1)),
                context_timestep.flatten(0, 1),
            ).unflatten(0, student_chunk.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=context_noisy,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.student_kv_cache1,
                    crossattn_cache=self.student_crossattn_cache,
                    current_start=abs_chunk_start * self.frame_seq_length,
                )

                teacher_context_noisy = self.scheduler.add_noise(
                    teacher_chunk.detach().flatten(0, 1),
                    torch.randn_like(teacher_chunk.detach().flatten(0, 1)),
                    context_timestep.flatten(0, 1),
                ).unflatten(0, teacher_chunk.shape[:2])
                self._write_chunk_to_cache(
                    chunk_latents=teacher_context_noisy,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.teacher_kv_cache1,
                    crossattn_cache=self.teacher_crossattn_cache,
                    chunk_start_frame=abs_chunk_start,
                    use_teacher_ema=True,
                )

            generated_spans = self.state["opsd_generated_chunk_spans"]
            generated_spans.append((int(abs_chunk_start), int(current_num_frames)))

            if self.opsd_teacher_context_mode == "gt_kv" and len(generated_spans) >= 2:
                # Keep the most recent generated chunk for continuation.
                # Replace only the second latest generated chunk with GT, matching inference semantics.
                replace_start, replace_num_frames = generated_spans[-2]
                replace_end = replace_start + replace_num_frames
                if gt_latents.shape[1] >= replace_end:
                    gt_chunk = gt_latents[:, replace_start:replace_end]
                    self._replace_cache_chunk_with_gt(
                        kv_cache=self.teacher_kv_cache1,
                        crossattn_cache=self.teacher_crossattn_cache,
                        gt_chunk_latents=gt_chunk,
                        conditional_dict=conditional_dict,
                        chunk_start_frame=replace_start,
                        use_teacher_ema=True,
                    )

            self.state["opsd_student_chunks"].append(student_chunk.detach())
            self.state["opsd_teacher_chunks"].append(teacher_chunk.detach())
            current_length += current_num_frames

        self.state["current_length"] = current_length
        if self.opsd_loss_type not in {"flow", "x0"}:
            raise ValueError(f"Unsupported opsd_loss_type={self.opsd_loss_type}. Use 'flow' or 'x0'.")
        if total_loss is None or total_loss_terms == 0:
            raise RuntimeError("No OPSD loss terms were accumulated. Check opsd_loss_step_mode/flag settings.")
        loss = total_loss / float(total_loss_terms)
        avg_flag = float(total_flag_value) / float(max(total_chunks, 1))
        log_dict = {
            "loss_time": torch.tensor(0.0, device=self.device),
            "new_frames_supervised": torch.tensor(window_frames, device=self.device),
            "opsd_window_start": torch.tensor(window_start, device=self.device),
            "opsd_window_frames": torch.tensor(window_frames, device=self.device),
            "opsd_loss_frames": torch.tensor(total_supervised_frames, device=self.device),
            "opsd_loss_start_frame": torch.tensor(self.opsd_loss_start_frame, device=self.device),
            "opsd_loss_terms": torch.tensor(total_loss_terms, device=self.device),
            "opsd_avg_flag": torch.tensor(avg_flag, device=self.device),
        }
        return loss, log_dict

    @torch.no_grad()
    def debug_opsd_rollout_videos(
        self,
        student_video_path: str,
        teacher_video_path: str,
        fps: int = 16,
        decode_chunk_size: int = 120,
        teacher_use_future_gt_context: bool = False,
        teacher_future_gt_num_chunks: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Run the OPSD training rollout without loss/backward and save student/teacher videos.

        This mirrors the rollout path in `compute_opsd_generator_loss`: it uses the
        same denoising steps, student/teacher KV caches, optional EMA teacher swap,
        student cache update from the fully denoised student chunk, and GT replacement
        in the teacher cache. Call `setup_opsd_sequence(...)` before this method.
        """
        if self.state.get("opsd_gt_latents", None) is None:
            raise ValueError("OPSD sequence is not initialized. Please call setup_opsd_sequence first.")

        gt_latents = self.state["opsd_gt_latents"]
        window_start = int(self.state["opsd_window_start"])
        window_frames = int(self.state["opsd_window_frames"])
        current_length = int(self.state["current_length"])
        block = int(self.num_frame_per_block)
        conditional_dict = self._get_current_conditional_dict(window_start + current_length)
        denoising_steps = self.inference_pipeline.denoising_step_list
        if len(denoising_steps) <= 0:
            raise ValueError("Empty denoising_step_list for OPSD debug rollout.")
        teacher_future_gt_num_chunks = max(0, int(teacher_future_gt_num_chunks))

        student_chunks = list(self.state.get("opsd_student_chunks", []))
        teacher_chunks = list(self.state.get("opsd_teacher_chunks", []))

        while current_length < window_frames:
            current_num_frames = min(block, window_frames - current_length)
            abs_chunk_start = window_start + current_length
            noise_chunk = torch.randn(
                [gt_latents.shape[0], current_num_frames, *gt_latents.shape[2:]],
                device=self.device,
                dtype=self.dtype,
            )
            noisy_input = noise_chunk
            teacher_noisy_input = noise_chunk
            student_chunk = None
            teacher_chunk = None

            if hasattr(self.inference_pipeline, "local_attn_size"):
                self.generator.model.local_attn_size = int(self.inference_pipeline.local_attn_size)
                if hasattr(self.inference_pipeline, "_set_all_modules_max_attention_size"):
                    self.inference_pipeline._set_all_modules_max_attention_size(int(self.inference_pipeline.local_attn_size))

            if teacher_use_future_gt_context and teacher_future_gt_num_chunks > 0:
                future_start = abs_chunk_start + current_num_frames
                available_future_frames = gt_latents.shape[1] - future_start
                available_future_chunks = max(0, available_future_frames // block)
                capture_future_chunks = min(teacher_future_gt_num_chunks, available_future_chunks)
                if capture_future_chunks > 0:
                    future_num_frames = capture_future_chunks * block
                    future_gt_latents = gt_latents[:, future_start:future_start + future_num_frames]
                    self._capture_future_gt_context_for_cache(
                        future_gt_latents=future_gt_latents,
                        future_start_frame=future_start,
                        conditional_dict=conditional_dict,
                        target_kv_cache=self.teacher_kv_cache1,
                        use_teacher_ema=True,
                    )
                else:
                    self._clear_future_context(self.teacher_kv_cache1)
            else:
                self._clear_future_context(self.teacher_kv_cache1)

            for step_idx, current_timestep in enumerate(denoising_steps):
                step_cpu_rng_state = torch.random.get_rng_state()
                step_cuda_rng_state = torch.cuda.get_rng_state(self.device) if torch.cuda.is_available() else None
                timestep = torch.ones(
                    [gt_latents.shape[0], current_num_frames],
                    device=self.device,
                    dtype=torch.int64,
                ) * current_timestep

                teacher_backup = self._swap_generator_to_ema()
                try:
                    teacher_flow, teacher_chunk = self.generator(
                        noisy_image_or_video=teacher_noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.teacher_kv_cache1,
                        crossattn_cache=self.teacher_crossattn_cache,
                        current_start=abs_chunk_start * self.frame_seq_length,
                    )
                finally:
                    self._restore_generator_from_backup(teacher_backup)

                torch.random.set_rng_state(step_cpu_rng_state)
                if step_cuda_rng_state is not None:
                    torch.cuda.set_rng_state(step_cuda_rng_state, device=self.device)
                student_flow, student_chunk = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.student_kv_cache1,
                    crossattn_cache=self.student_crossattn_cache,
                    current_start=abs_chunk_start * self.frame_seq_length,
                )

                if step_idx < len(denoising_steps) - 1:
                    next_timestep = denoising_steps[step_idx + 1]
                    transition_noise = torch.randn_like(student_chunk.detach().flatten(0, 1))
                    noisy_input = self.scheduler.add_noise(
                        student_chunk.detach().flatten(0, 1),
                        transition_noise,
                        next_timestep * torch.ones(
                            [gt_latents.shape[0] * current_num_frames],
                            device=self.device,
                            dtype=torch.long,
                        ),
                    ).unflatten(0, student_chunk.shape[:2])
                    if self.opsd_teacher_trajectory_mode == "teacher":
                        teacher_noisy_input = self.scheduler.add_noise(
                            teacher_chunk.detach().flatten(0, 1),
                            transition_noise,
                            next_timestep * torch.ones(
                                [gt_latents.shape[0] * current_num_frames],
                                device=self.device,
                                dtype=torch.long,
                            ),
                        ).unflatten(0, teacher_chunk.shape[:2])
                    else:
                        teacher_noisy_input = noisy_input

            if student_chunk is None or teacher_chunk is None:
                raise RuntimeError("OPSD debug rollout produced no outputs.")

            context_timestep = torch.ones(
                [gt_latents.shape[0], current_num_frames],
                device=self.device,
                dtype=torch.int64,
            ) * self.inference_pipeline.context_noise
            context_noisy = self.scheduler.add_noise(
                student_chunk.detach().flatten(0, 1),
                torch.randn_like(student_chunk.detach().flatten(0, 1)),
                context_timestep.flatten(0, 1),
            ).unflatten(0, student_chunk.shape[:2])
            self.generator(
                noisy_image_or_video=context_noisy,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.student_kv_cache1,
                crossattn_cache=self.student_crossattn_cache,
                current_start=abs_chunk_start * self.frame_seq_length,
            )
            teacher_context_noisy = self.scheduler.add_noise(
                teacher_chunk.detach().flatten(0, 1),
                torch.randn_like(teacher_chunk.detach().flatten(0, 1)),
                context_timestep.flatten(0, 1),
            ).unflatten(0, teacher_chunk.shape[:2])
            self._write_chunk_to_cache(
                chunk_latents=teacher_context_noisy,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.teacher_kv_cache1,
                crossattn_cache=self.teacher_crossattn_cache,
                chunk_start_frame=abs_chunk_start,
                use_teacher_ema=True,
            )

            generated_spans = self.state["opsd_generated_chunk_spans"]
            generated_spans.append((int(abs_chunk_start), int(current_num_frames)))
            if self.opsd_teacher_context_mode == "gt_kv" and len(generated_spans) >= 2:
                replace_start, replace_num_frames = generated_spans[-2]
                replace_end = replace_start + replace_num_frames
                if gt_latents.shape[1] >= replace_end:
                    gt_chunk = gt_latents[:, replace_start:replace_end]
                    self._replace_cache_chunk_with_gt(
                        kv_cache=self.teacher_kv_cache1,
                        crossattn_cache=self.teacher_crossattn_cache,
                        gt_chunk_latents=gt_chunk,
                        conditional_dict=conditional_dict,
                        chunk_start_frame=replace_start,
                        use_teacher_ema=True,
                    )

            student_chunks.append(student_chunk.detach())
            teacher_chunks.append(teacher_chunk.detach())
            current_length += current_num_frames

        student_latents = torch.cat(student_chunks, dim=1)
        teacher_latents = torch.cat(teacher_chunks, dim=1)

        self._save_debug_latent_video(student_latents, student_video_path, fps=fps, decode_chunk_size=decode_chunk_size)
        self._save_debug_latent_video(teacher_latents, teacher_video_path, fps=fps, decode_chunk_size=decode_chunk_size)

        self.state["current_length"] = current_length
        self.state["opsd_student_chunks"] = student_chunks
        self.state["opsd_teacher_chunks"] = teacher_chunks
        return {"student_latents": student_latents, "teacher_latents": teacher_latents}

    def _save_debug_latent_video(
        self,
        latents: torch.Tensor,
        output_path: str,
        fps: int = 16,
        decode_chunk_size: int = 120,
    ):
        if not output_path:
            return
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if getattr(self.config.model_kwargs, "use_infinite_attention", False):
            video = self.base_model.vae.decode_to_pixel_chunk(
                latents.to(self.device),
                use_cache=False,
                chunk_size=decode_chunk_size,
            )
        else:
            video = self.base_model.vae.decode_to_pixel(latents.to(self.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        video_uint8 = (255.0 * rearrange(video, "b t c h w -> b t h w c").cpu()).to(torch.uint8)
        self.base_model.vae.model.clear_cache()
        write_video(output_path, video_uint8[0], fps=fps)
