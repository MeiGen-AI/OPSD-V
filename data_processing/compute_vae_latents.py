#!/usr/bin/env python3
"""Encode prepared video tensors into Wan VAE latents."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta
from glob import glob
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.wan_wrapper import WanVAEWrapper  # noqa: E402


def _launch_distributed_job(backend: str = "nccl") -> tuple[int, int, int]:
    required_envs = ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    if all(k in os.environ for k in required_envs):
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return 0, 0, 1


def _load_video_tensor(video_pt_path: str) -> torch.Tensor:
    video = torch.load(video_pt_path, map_location="cpu", weights_only=False)
    if isinstance(video, dict):
        for key in ("video", "frames", "pixel_values"):
            if key in video:
                video = video[key]
                break
    if not isinstance(video, torch.Tensor) or video.ndim != 4:
        raise ValueError(f"Expected [T,C,H,W] or [C,T,H,W] tensor in {video_pt_path}, got {type(video)}")

    if video.shape[0] in (1, 3):
        video = video.unsqueeze(0)  # [1, C, T, H, W]
    elif video.shape[1] in (1, 3):
        video = video.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
    else:
        raise ValueError(f"Cannot infer channel dimension from shape {tuple(video.shape)} in {video_pt_path}")

    video = video.float()
    if video.max() > 1.5:
        video = video / 255.0
    if video.min() >= 0.0 and video.max() <= 1.0:
        video = video * 2.0 - 1.0
    return video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_pt_dir", type=Path, required=True)
    parser.add_argument("--output_latent_folder", type=Path, required=True)
    parser.add_argument("--model_root", type=str, default=None, help="Optional Wan2.1-T2V-1.3B path.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    rank, _, world_size = _launch_distributed_job()
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    args.output_latent_folder.mkdir(parents=True, exist_ok=True)

    files = sorted(glob(str(args.video_pt_dir / "*_video.pt")))
    if not files:
        raise FileNotFoundError(f"No *_video.pt files found in {args.video_pt_dir}")

    model = WanVAEWrapper(model_root=args.model_root).to(device=device, dtype=torch.bfloat16).eval()
    local_files = files[rank::world_size]
    for video_pt_path in tqdm(local_files, desc=f"VAE rank {rank}", disable=rank != 0):
        sample_id = Path(video_pt_path).name.replace("_video.pt", "")
        output_path = args.output_latent_folder / f"{sample_id}_latent.pt"
        if args.resume and output_path.exists():
            continue
        video = _load_video_tensor(video_pt_path).to(device=device, dtype=torch.bfloat16)
        latent = model.encode_to_latent(video).cpu().half()  # [1, T, 16, H, W]
        torch.save(latent, output_path)
        del video, latent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
