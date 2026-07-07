import argparse
import os
import re
import sys
from pathlib import Path

import peft
import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.io import write_video
from tqdm import tqdm

from pipeline import CausalInferencePipeline, CausalInferencePipelineLmdb
from utils.dataset import InferencePromptEmbedsVideoLMDBDataset, TextDataset
from utils.lora_utils import configure_lora_for_model
from utils.memory import (
    DynamicSwapInstaller,
    get_cuda_free_memory_gb,
    gpu,
)
from utils.misc import set_seed


REPO_ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, default="", help="Optional generator checkpoint override")
parser.add_argument("--data_path", type=str, required=True, help="Path to a prompt text file or LMDB")
parser.add_argument("--output_folder", type=str, required=True, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21, help="Number of latent frames to generate")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--use_lora_ema", action="store_true", help="Use LoRA EMA weights from the LoRA checkpoint")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--seed_list", type=str, default="", help="Comma-separated seeds. If set, run full prompts for each seed.")
parser.add_argument(
    "--per_prompt_seed",
    action="store_true",
    help=(
        "Derive an independent deterministic noise seed from (seed, prompt index). "
        "This makes resumed/skipped runs use the same noise per prompt."
    ),
)
parser.add_argument(
    "--per_prompt_seed_stride",
    type=int,
    default=100000,
    help="Stride used for per-prompt seeds: case_seed = seed * stride + prompt_index.",
)
parser.add_argument("--use_lmdb", action="store_true", help="Read prompts/prompt_embeds/(optional) gt latents from LMDB")
parser.add_argument("--use_lmdb_pipeline", action="store_true", help="Use the LMDB-enhanced inference pipeline even when reading prompts from txt")
parser.add_argument("--lmdb_cache_update_source", type=str, default="generated", choices=["generated", "gt"], help="Update KV cache from model prediction or GT latents")
parser.add_argument("--lmdb_use_gt_first_chunk", action="store_true", help="Replace the first generated chunk in both output and KV cache with GT")
parser.add_argument("--lmdb_start_gt_chunk", type=int, default=0, help="GT chunk index used as the first output chunk; RoPE/KV positions are shifted accordingly")
parser.add_argument(
    "--lmdb_replace_latest_chunk_with_gt",
    action="store_true",
    help="When lmdb_cache_update_source=gt, replace latest generated chunk in KV (instead of second-latest)",
)
parser.add_argument(
    "--lmdb_use_relative_sink",
    action="store_true",
    help="Use first-chunk persistent sink with relative RoPE during LMDB inference",
)
parser.add_argument(
    "--lmdb_relative_sink_gt_blend_alpha",
    type=float,
    default=0.0,
    help="Blend ratio for per-chunk GT sink into persistent sink (0.0 disables GT sink blending).",
)
parser.add_argument(
    "--lmdb_history_only_after_first_gt",
    action="store_true",
    help="After the first GT chunk, use history-only KV for attention (exclude current chunk K/V).",
)
parser.add_argument(
    "--lmdb_use_future_gt_context",
    action="store_true",
    help="Expose future GT chunk(s) as extra attention context during LMDB inference.",
)
parser.add_argument(
    "--lmdb_future_gt_num_chunks",
    type=int,
    default=1,
    help="Number of future GT chunks to expose as extra attention context when enabled.",
)
parser.add_argument(
    "--disable_fixed_gt_window_after_gt",
    dest="enable_fixed_gt_window_after_gt",
    action="store_false",
    help="After GT is exhausted, disable fixed GT-window rebuilding and continue with pure rollout",
)
parser.set_defaults(enable_fixed_gt_window_after_gt=True)
args = parser.parse_args()


def _save_run_config(output_folder: str, config, args, local_rank: int):
    if local_rank != 0 or not output_folder:
        return
    os.makedirs(output_folder, exist_ok=True)
    OmegaConf.save(config, os.path.join(output_folder, "resolved_config.yaml"))
    cli_info = OmegaConf.create(
        {
            "config_path": args.config_path,
            "command": " ".join(sys.argv),
            "cli_args": vars(args),
        }
    )
    OmegaConf.save(cli_info, os.path.join(output_folder, "run_args.yaml"))


def _safe_output_stem(prompt: str, idx: int, max_len: int = 80):
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", prompt).strip("._-")
    if not stem:
        stem = "sample"
    stem = stem[:max_len]
    return f"{idx:06d}_{stem}"


def _parse_seed_list(seed: int, seed_list: str):
    if not seed_list:
        return [int(seed)]
    seeds = []
    for x in seed_list.split(","):
        x = x.strip()
        if not x:
            continue
        seeds.append(int(x))
    if not seeds:
        seeds = [int(seed)]
    return seeds


def _case_seed(base_seed: int, idx: int, stride: int) -> int:
    stride = max(1, int(stride))
    return int(base_seed) * stride + int(idx)


def _randn_noise(shape, device, dtype, seed=None):
    if seed is None:
        return torch.randn(shape, device=device, dtype=dtype)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def _load_lora_ema_weights(generator, ema_state_dict):
    def _clean_key(name: str) -> str:
        return name.replace("_fsdp_wrapped_module.", "").removeprefix("module.")

    cleaned_ema = {_clean_key(k): v for k, v in ema_state_dict.items()}
    loaded = 0
    missing = []
    with torch.no_grad():
        for name, param in generator.named_parameters():
            if "lora_" not in name:
                continue
            ema_value = cleaned_ema.get(_clean_key(name), None)
            if ema_value is None:
                missing.append(name)
                continue
            param.copy_(ema_value.to(device=param.device, dtype=param.dtype))
            loaded += 1
    return loaded, missing


seed_values = _parse_seed_list(args.seed, args.seed_list)
set_seed(seed_values[0])

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load(REPO_ROOT / "configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)
config.use_ema = bool(args.use_ema)


if "LOCAL_RANK" in os.environ:
    os.environ["NCCL_CROSS_NIC"] = "1"
    os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
    os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout,
        )
    set_seed(seed_values[0] + local_rank)
else:
    local_rank = 0
    rank = 0
    device = torch.device("cuda")

print(f"Free VRAM {get_cuda_free_memory_gb(gpu)} GB")
low_memory = get_cuda_free_memory_gb(gpu) < 40
torch.set_grad_enabled(False)


use_lmdb_pipeline = bool(args.use_lmdb or args.use_lmdb_pipeline)

if use_lmdb_pipeline:
    pipeline = CausalInferencePipelineLmdb(config, device=device)
else:
    pipeline = CausalInferencePipeline(config, device=device)

checkpoint_path = args.checkpoint_path if args.checkpoint_path else getattr(config, "generator_ckpt", "")
if checkpoint_path:
    if local_rank == 0:
        print(f"Loading base generator checkpoint: {checkpoint_path} (use_ema={args.use_ema})")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "generator" in state_dict or "generator_ema" in state_dict:
        generator_key = "generator_ema" if args.use_ema else "generator"
        raw_gen_state_dict = state_dict[generator_key]
    elif "model" in state_dict:
        generator_key = "model"
        raw_gen_state_dict = state_dict["model"]
    else:
        raise ValueError(f"Generator state dict not found in {checkpoint_path}")

    if args.use_ema:
        def _clean_key(name: str) -> str:
            return name.replace("_fsdp_wrapped_module.", "")

        cleaned_state_dict = {_clean_key(k): v for k, v in raw_gen_state_dict.items()}
        missing, unexpected = pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
        if local_rank == 0:
            if missing:
                print(f"[Warning] {len(missing)} parameters missing: {missing[:8]} ...")
            if unexpected:
                print(f"[Warning] {len(unexpected)} unexpected params: {unexpected[:8]} ...")
    else:
        pipeline.generator.load_state_dict(raw_gen_state_dict)
    if local_rank == 0:
        print(f"Loaded base generator checkpoint key: {generator_key}")
else:
    if local_rank == 0:
        print("No base generator checkpoint provided; using initialized generator weights.")

pipeline.is_lora_enabled = False
if getattr(config, "adapter", None) and configure_lora_for_model is not None:
    if local_rank == 0:
        print(f"LoRA enabled with config: {config.adapter}")
        print("Applying LoRA to generator (inference)...")
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )

    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        if local_rank == 0:
            print(f"Loading LoRA checkpoint: {lora_ckpt_path} (use_lora_ema={args.use_lora_ema})")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        if args.use_lora_ema and isinstance(lora_checkpoint, dict) and "generator_ema" in lora_checkpoint:
            if local_rank == 0:
                print("LoRA checkpoint branch: generator_ema")
            loaded, missing = _load_lora_ema_weights(pipeline.generator, lora_checkpoint["generator_ema"])
            if loaded == 0:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_ema"])
                missing = []
            if local_rank == 0:
                if loaded > 0:
                    print(f"LoRA EMA weights loaded for generator: {loaded} tensors")
                else:
                    print("LoRA EMA PEFT state loaded for generator")
                if missing:
                    print(f"[Warning] Missing {len(missing)} LoRA EMA tensors, first few: {missing[:8]}")
        elif isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            if args.use_lora_ema and local_rank == 0:
                print("[Warning] --use_lora_ema was set, but generator_ema was not found; loading generator_lora instead.")
            if local_rank == 0:
                print("LoRA checkpoint branch: generator_lora")
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
        else:
            if args.use_lora_ema and local_rank == 0:
                print("[Warning] --use_lora_ema was set, but checkpoint is not an OPSD LoRA EMA checkpoint; loading as LoRA weights.")
            if local_rank == 0:
                print("LoRA checkpoint branch: raw PEFT state dict")
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)
        if local_rank == 0:
            print("LoRA weights loaded for generator")
    pipeline.is_lora_enabled = True
else:
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if local_rank == 0:
        if lora_ckpt_path:
            print(
                f"[Warning] lora_ckpt is set ({lora_ckpt_path}) but adapter config is missing/empty; "
                "LoRA will NOT be loaded."
            )
        else:
            print("LoRA disabled: no adapter config.")

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

if args.use_lmdb:
    dataset = InferencePromptEmbedsVideoLMDBDataset(
        args.data_path,
        require_gt_latents=(args.lmdb_cache_update_source == "gt") or args.lmdb_use_gt_first_chunk,
    )
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.data_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)
    _save_run_config(args.output_folder, config, args, local_rank)

if dist.is_initialized():
    dist.barrier()

if local_rank == 0:
    print(f"Running inference for seeds: {seed_values}")

for current_seed in seed_values:
    set_seed(current_seed)
    seed_output_folder = (
        os.path.join(args.output_folder, f"seed_{current_seed}")
        if len(seed_values) > 1
        else args.output_folder
    )
    if local_rank == 0:
        os.makedirs(seed_output_folder, exist_ok=True)
        print(f"Start seed={current_seed}, output={seed_output_folder}")

    for _, batch_data in tqdm(
        enumerate(dataloader),
        disable=(local_rank != 0),
        desc=f"seed={current_seed}",
    ):
        idx = batch_data["idx"].item()
        batch = batch_data if isinstance(batch_data, dict) else batch_data[0]

        if args.use_lmdb:
            prompt = batch["prompts"][0]
            prompts = [prompt]
            prompt_embeds = batch["prompt_embeds"].to(device=device)

            output_path = os.path.join(seed_output_folder, f"{_safe_output_stem(prompt, idx)}.mp4")
            if os.path.exists(output_path):
                print("Video has been generated. Pass!")
                continue

            gt_latents = batch["gt_latents"].to(device=device, dtype=torch.bfloat16) if "gt_latents" in batch else None
            noise_seed = _case_seed(current_seed, idx, args.per_prompt_seed_stride) if args.per_prompt_seed else None
            sampled_noise = _randn_noise(
                [1, args.num_output_frames, 16, 60, 104],
                device=device,
                dtype=torch.bfloat16,
                seed=noise_seed,
            )
            inference_kwargs = dict(
                noise=sampled_noise,
                text_prompts=prompts,
                prompt_embeds=prompt_embeds,
                gt_latents=gt_latents,
                cache_update_source=args.lmdb_cache_update_source,
                use_gt_first_chunk=args.lmdb_use_gt_first_chunk,
                start_gt_chunk=args.lmdb_start_gt_chunk,
                replace_latest_chunk_with_gt=args.lmdb_replace_latest_chunk_with_gt,
                use_relative_sink=args.lmdb_use_relative_sink,
                relative_sink_gt_blend_alpha=args.lmdb_relative_sink_gt_blend_alpha,
                history_only_after_first_gt=args.lmdb_history_only_after_first_gt,
                use_future_gt_context=args.lmdb_use_future_gt_context,
                future_gt_num_chunks=args.lmdb_future_gt_num_chunks,
                return_latents=True,
                low_memory=low_memory,
                enable_fixed_gt_window_after_gt=args.enable_fixed_gt_window_after_gt,
            )
        else:
            prompt = batch["prompts"][0]
            prompts = [prompt]
            output_path = os.path.join(seed_output_folder, f"{_safe_output_stem(prompt, idx)}.mp4")
            if os.path.exists(output_path):
                print("Video has been generated. Pass!")
                continue

            noise_seed = _case_seed(current_seed, idx, args.per_prompt_seed_stride) if args.per_prompt_seed else None
            sampled_noise = _randn_noise(
                [1, args.num_output_frames, 16, 60, 104],
                device=device,
                dtype=torch.bfloat16,
                seed=noise_seed,
            )
            if use_lmdb_pipeline:
                inference_kwargs = dict(
                    noise=sampled_noise,
                    text_prompts=prompts,
                    gt_latents=None,
                    cache_update_source=args.lmdb_cache_update_source,
                    use_gt_first_chunk=args.lmdb_use_gt_first_chunk,
                    start_gt_chunk=args.lmdb_start_gt_chunk,
                    replace_latest_chunk_with_gt=args.lmdb_replace_latest_chunk_with_gt,
                    use_relative_sink=args.lmdb_use_relative_sink,
                    relative_sink_gt_blend_alpha=args.lmdb_relative_sink_gt_blend_alpha,
                    history_only_after_first_gt=args.lmdb_history_only_after_first_gt,
                    use_future_gt_context=args.lmdb_use_future_gt_context,
                    future_gt_num_chunks=args.lmdb_future_gt_num_chunks,
                    return_latents=True,
                    low_memory=low_memory,
                    enable_fixed_gt_window_after_gt=args.enable_fixed_gt_window_after_gt,
                )
            else:
                inference_kwargs = dict(
                    noise=sampled_noise,
                    text_prompts=prompts,
                    return_latents=True,
                    low_memory=low_memory,
                )

        video, latents = pipeline.inference(**inference_kwargs)
        current_video = rearrange(video, "b t c h w -> b t h w c").cpu()
        video_uint8 = (255.0 * current_video).to(torch.uint8)
        pipeline.vae.model.clear_cache()
        write_video(output_path, video_uint8[0], fps=16)

    if dist.is_initialized():
        dist.barrier()

if dist.is_initialized():
    dist.destroy_process_group()
