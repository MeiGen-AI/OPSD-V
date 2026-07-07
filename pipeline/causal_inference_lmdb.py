from typing import List, Optional

import torch

from pipeline.causal_inference import CausalInferencePipeline
from utils.memory import (
    gpu,
    get_cuda_free_memory_gb,
    move_model_to_device_with_memory_preservation,
)


class CausalInferencePipelineLmdb(CausalInferencePipeline):
    """LMDB/text inference pipeline with optional GT cache correction."""

    def _set_relative_sink_mode(self, enabled: bool):
        for cache in self.kv_cache1:
            cache["use_relative_sink"] = bool(enabled)
            cache["dynamic_sink_alpha"] = 0.0
            cache["history_only_after_first_gt"] = False
            cache.pop("persistent_sink_raw_k", None)
            cache.pop("persistent_sink_v", None)
            cache.pop("persistent_sink_num_frames", None)
            cache.pop("capture_sink_raw", None)
            cache.pop("captured_sink_raw_k", None)
            cache.pop("captured_sink_v", None)
            cache.pop("captured_sink_num_frames", None)
            cache.pop("dynamic_sink_raw_k", None)
            cache.pop("dynamic_sink_v", None)
            cache.pop("capture_context_raw", None)
            cache.pop("captured_context_raw_k", None)
            cache.pop("captured_context_v", None)
            cache.pop("captured_context_num_frames", None)
            cache.pop("future_context_raw_k", None)
            cache.pop("future_context_v", None)
            cache.pop("future_context_num_frames", None)
            cache.pop("future_context_start_frame", None)

    def _begin_sink_capture(self):
        for cache in self.kv_cache1:
            cache["capture_sink_raw"] = True
            cache["capture_sink_raw_any_start"] = True
            cache.pop("captured_sink_raw_k", None)
            cache.pop("captured_sink_v", None)
            cache.pop("captured_sink_num_frames", None)

    def _finalize_sink_capture(self) -> bool:
        all_captured = True
        for cache in self.kv_cache1:
            sink_raw_k = cache.pop("captured_sink_raw_k", None)
            sink_v = cache.pop("captured_sink_v", None)
            sink_num_frames = cache.pop("captured_sink_num_frames", None)
            cache.pop("capture_sink_raw", None)
            cache.pop("capture_sink_raw_any_start", None)

            if sink_raw_k is None or sink_v is None or sink_num_frames is None:
                all_captured = False
                cache.pop("persistent_sink_raw_k", None)
                cache.pop("persistent_sink_v", None)
                cache.pop("persistent_sink_num_frames", None)
                continue

            cache["persistent_sink_raw_k"] = sink_raw_k
            cache["persistent_sink_v"] = sink_v
            cache["persistent_sink_num_frames"] = int(sink_num_frames)
        return all_captured

    def _capture_gt_chunk_sink(
        self,
        gt_chunk_latents: torch.Tensor,
        conditional_dict: dict,
        current_start_frame: int,
    ) -> bool:
        if gt_chunk_latents is None:
            return False

        sink_token_count = int(self.args.model_kwargs.sink_size) * self.frame_seq_length
        if sink_token_count <= 0:
            return False

        temp_kv_cache = []
        for cache in self.kv_cache1:
            temp_kv_cache.append({
                "k": torch.zeros_like(cache["k"]),
                "v": torch.zeros_like(cache["v"]),
                "global_end_index": torch.zeros_like(cache["global_end_index"]),
                "local_end_index": torch.zeros_like(cache["local_end_index"]),
                "capture_sink_raw": True,
                "capture_sink_raw_any_start": True,
            })

        temp_crossattn_cache = []
        for cache in self.crossattn_cache:
            temp_crossattn_cache.append({
                "k": torch.zeros_like(cache["k"]),
                "v": torch.zeros_like(cache["v"]),
                "is_init": False,
            })

        timestep = torch.ones(
            [gt_chunk_latents.shape[0], gt_chunk_latents.shape[1]],
            device=gt_chunk_latents.device,
            dtype=torch.int64,
        ) * self.args.context_noise

        self.generator(
            noisy_image_or_video=gt_chunk_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=temp_kv_cache,
            crossattn_cache=temp_crossattn_cache,
            # We only need pre-RoPE sink raw K/V content; use start=0 to avoid
            # large global index interactions with a temporary empty cache.
            current_start=0,
        )

        captured = True
        for src_cache, dst_cache in zip(temp_kv_cache, self.kv_cache1):
            sink_raw_k = src_cache.get("captured_sink_raw_k")
            sink_v = src_cache.get("captured_sink_v")
            if sink_raw_k is None or sink_v is None:
                captured = False
                dst_cache.pop("dynamic_sink_raw_k", None)
                dst_cache.pop("dynamic_sink_v", None)
            else:
                dst_cache["dynamic_sink_raw_k"] = sink_raw_k
                dst_cache["dynamic_sink_v"] = sink_v
        return captured

    def _capture_future_gt_context(
        self,
        future_gt_latents: torch.Tensor,
        future_start_frame: int,
        conditional_dict: dict,
    ) -> bool:
        if future_gt_latents is None or future_gt_latents.shape[1] <= 0:
            return False

        temp_kv_cache = []
        for cache in self.kv_cache1:
            temp_kv_cache.append({
                "k": torch.zeros_like(cache["k"]),
                "v": torch.zeros_like(cache["v"]),
                "global_end_index": torch.zeros_like(cache["global_end_index"]),
                "local_end_index": torch.zeros_like(cache["local_end_index"]),
                "capture_context_raw": True,
            })

        temp_crossattn_cache = []
        for cache in self.crossattn_cache:
            temp_crossattn_cache.append({
                "k": torch.zeros_like(cache["k"]),
                "v": torch.zeros_like(cache["v"]),
                "is_init": False,
            })

        timestep = torch.ones(
            [future_gt_latents.shape[0], future_gt_latents.shape[1]],
            device=future_gt_latents.device,
            dtype=torch.int64,
        ) * self.args.context_noise

        self.generator(
            noisy_image_or_video=future_gt_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=temp_kv_cache,
            crossattn_cache=temp_crossattn_cache,
            current_start=0,
        )

        captured = True
        for src_cache, dst_cache in zip(temp_kv_cache, self.kv_cache1):
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

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        prompt_embeds: Optional[torch.Tensor] = None,
        gt_latents: Optional[torch.Tensor] = None,
        cache_update_source: str = "generated",
        use_gt_first_chunk: bool = False,
        start_gt_chunk: int = 0,
        replace_latest_chunk_with_gt: bool = False,
        use_relative_sink: bool = False,
        relative_sink_gt_blend_alpha: float = 0.0,
        history_only_after_first_gt: bool = False,
        use_future_gt_context: bool = False,
        future_gt_num_chunks: int = 1,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        enable_fixed_gt_window_after_gt: bool = True,
    ) -> torch.Tensor:
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        if prompt_embeds is not None:
            conditional_dict = {
                "prompt_embeds": prompt_embeds.to(device=noise.device, dtype=noise.dtype)
            }
        else:
            conditional_dict = self.text_encoder(text_prompts=text_prompts)

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )

        output_device = torch.device("cpu") if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype,
        )

        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(
            f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, "
            f"frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})"
        )

        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size,
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
        )
        self._set_relative_sink_mode(use_relative_sink)
        blend_alpha = max(0.0, min(1.0, float(relative_sink_gt_blend_alpha)))
        future_gt_num_chunks = max(0, int(future_gt_num_chunks))
        history_only_enabled = bool(history_only_after_first_gt and use_gt_first_chunk)
        for cache in self.kv_cache1:
            cache["dynamic_sink_alpha"] = blend_alpha
            cache["history_only_after_first_gt"] = history_only_enabled

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        start_gt_chunk = max(0, int(start_gt_chunk))
        gt_start_frame = start_gt_chunk * self.num_frame_per_block if gt_latents is not None else 0
        gt_total_chunks = 0
        fixed_gt_context_chunks = 0
        fixed_gt_context_frames = 0
        if gt_latents is not None:
            if gt_start_frame >= gt_latents.shape[1]:
                raise ValueError(
                    f"start_gt_chunk={start_gt_chunk} starts at frame {gt_start_frame}, "
                    f"but gt_latents only has {gt_latents.shape[1]} frames."
                )
            gt_total_chunks = (gt_latents.shape[1] - gt_start_frame) // self.num_frame_per_block
            fixed_gt_context_chunks = max(0, gt_total_chunks - 1)
            fixed_gt_context_frames = fixed_gt_context_chunks * self.num_frame_per_block
            print(f"gt_start_frame: {gt_start_frame}, gt_total_chunks_from_start: {gt_total_chunks}")

        generated_chunk_spans = []
        last_generated_chunk_latents = None

        all_num_frames = [self.num_frame_per_block] * num_blocks
        for current_num_frames in all_num_frames:
            global_start_frame = gt_start_frame + current_start_frame
            use_gt_first_for_this_chunk = (
                use_gt_first_chunk
                and gt_latents is not None
                and current_start_frame == 0
                and gt_latents.shape[1] >= global_start_frame + current_num_frames
            )

            use_fixed_gt_window = (
                cache_update_source == "gt"
                and gt_latents is not None
                and last_generated_chunk_latents is not None
                and len(generated_chunk_spans) >= gt_total_chunks
                and fixed_gt_context_chunks > 0
                and enable_fixed_gt_window_after_gt
            )

            if use_fixed_gt_window:
                cache_window_num_frames = fixed_gt_context_frames + self.num_frame_per_block
                cache_window_start_frame = max(0, global_start_frame - cache_window_num_frames)
                cache_write_start_frame = cache_window_start_frame
                for gt_chunk_idx in range(1, 1 + fixed_gt_context_chunks):
                    gt_start = gt_start_frame + gt_chunk_idx * self.num_frame_per_block
                    gt_end = gt_start + self.num_frame_per_block
                    gt_chunk_latents = gt_latents[:, gt_start:gt_end].to(
                        device=noise.device,
                        dtype=noise.dtype,
                    )
                    self.generator(
                        noisy_image_or_video=gt_chunk_latents,
                        conditional_dict=conditional_dict,
                        timestep=torch.zeros(
                            [batch_size, self.num_frame_per_block],
                            device=noise.device,
                            dtype=torch.int64,
                        ),
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=cache_write_start_frame * self.frame_seq_length,
                    )
                    cache_write_start_frame += self.num_frame_per_block

            if (
                use_relative_sink
                and blend_alpha > 0.0
                and gt_latents is not None
                and current_start_frame >= 2 * self.num_frame_per_block
            ):
                gt_start = global_start_frame
                gt_end = gt_start + current_num_frames
                if gt_latents.shape[1] >= gt_end:
                    gt_chunk_latents_for_sink = gt_latents[:, gt_start:gt_end].to(
                        device=noise.device,
                        dtype=noise.dtype,
                    )
                    captured_gt_sink = self._capture_gt_chunk_sink(
                        gt_chunk_latents=gt_chunk_latents_for_sink,
                        conditional_dict=conditional_dict,
                        current_start_frame=global_start_frame,
                    )
                    if not captured_gt_sink:
                        for cache in self.kv_cache1:
                            cache.pop("dynamic_sink_raw_k", None)
                            cache.pop("dynamic_sink_v", None)
                else:
                    for cache in self.kv_cache1:
                        cache.pop("dynamic_sink_raw_k", None)
                        cache.pop("dynamic_sink_v", None)

            if use_future_gt_context and gt_latents is not None and future_gt_num_chunks > 0:
                future_start = global_start_frame + current_num_frames
                available_future_frames = gt_latents.shape[1] - future_start
                available_future_chunks = max(0, available_future_frames // self.num_frame_per_block)
                capture_future_chunks = min(future_gt_num_chunks, available_future_chunks)
                if capture_future_chunks > 0:
                    future_num_frames = capture_future_chunks * self.num_frame_per_block
                    future_gt_latents = gt_latents[:, future_start:future_start + future_num_frames].to(
                        device=noise.device,
                        dtype=noise.dtype,
                    )
                    captured_future = self._capture_future_gt_context(
                        future_gt_latents=future_gt_latents,
                        future_start_frame=future_start,
                        conditional_dict=conditional_dict,
                    )
                    if not captured_future:
                        for cache in self.kv_cache1:
                            cache.pop("future_context_raw_k", None)
                            cache.pop("future_context_v", None)
                            cache.pop("future_context_num_frames", None)
                            cache.pop("future_context_start_frame", None)
                else:
                    for cache in self.kv_cache1:
                        cache.pop("future_context_raw_k", None)
                        cache.pop("future_context_v", None)
                        cache.pop("future_context_num_frames", None)
                        cache.pop("future_context_start_frame", None)

            if use_gt_first_for_this_chunk:
                current_chunk_output_latents = gt_latents[:, global_start_frame:global_start_frame + current_num_frames].to(
                    device=noise.device,
                    dtype=noise.dtype,
                )
                denoised_pred = current_chunk_output_latents
            else:
                noisy_input = noise[
                    :, current_start_frame:current_start_frame + current_num_frames
                ]

                for index, current_timestep in enumerate(self.denoising_step_list):
                    timestep = (
                        torch.ones(
                            [batch_size, current_num_frames],
                            device=noise.device,
                            dtype=torch.int64,
                        )
                        * current_timestep
                    )

                    if index < len(self.denoising_step_list) - 1:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=global_start_frame * self.frame_seq_length,
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep
                            * torch.ones(
                                [batch_size * current_num_frames],
                                device=noise.device,
                                dtype=torch.long,
                            ),
                        ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=global_start_frame * self.frame_seq_length,
                        )

                current_chunk_output_latents = denoised_pred

            timestep = torch.ones(
                [batch_size, current_num_frames],
                device=noise.device,
                dtype=torch.int64,
            ) * self.args.context_noise

            output[:, current_start_frame:current_start_frame + current_num_frames] = (
                current_chunk_output_latents.to(output.device)
            )

            context_timestep = timestep
            if use_relative_sink and current_start_frame == 0:
                self._begin_sink_capture()
            self.generator(
                noisy_image_or_video=current_chunk_output_latents,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=global_start_frame * self.frame_seq_length,
            )
            if use_relative_sink and current_start_frame == 0:
                captured = self._finalize_sink_capture()
                if not captured:
                    print("[relative_sink] capture failed; falling back to default sink behavior.")
                    self._set_relative_sink_mode(False)
                    use_relative_sink = False

            generated_chunk_spans.append((global_start_frame, current_num_frames))
            last_generated_chunk_latents = denoised_pred.detach()

            if cache_update_source == "gt":
                if (not use_fixed_gt_window) and gt_latents is not None and len(generated_chunk_spans) >= 3:
                    replace_idx = -1 if replace_latest_chunk_with_gt else -2
                    replace_start, replace_num_frames = generated_chunk_spans[replace_idx]
                    replace_end = replace_start + replace_num_frames
                    if gt_latents.shape[1] >= replace_end:
                        gt_cache_latents = gt_latents[:, replace_start:replace_end].to(
                            device=noise.device,
                            dtype=noise.dtype,
                        )
                        saved_end_indices = [
                            (cache["global_end_index"].clone(), cache["local_end_index"].clone())
                            for cache in self.kv_cache1
                        ]
                        self.generator(
                            noisy_image_or_video=gt_cache_latents,
                            conditional_dict=conditional_dict,
                            timestep=torch.ones(
                                [batch_size, replace_num_frames],
                                device=noise.device,
                                dtype=torch.int64,
                            ) * self.args.context_noise,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=replace_start * self.frame_seq_length,
                        )
                        for cache, (saved_global_end, saved_local_end) in zip(self.kv_cache1, saved_end_indices):
                            cache["global_end_index"] = saved_global_end
                            cache["local_end_index"] = saved_local_end

            current_start_frame += current_num_frames

        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output.to(noise.device)
        return video
