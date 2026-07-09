#!/usr/bin/env python3
"""Prepare normalized video tensors and prompt files for OPSD-V LMDB creation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def _read_manifest(path: Path) -> list[dict]:
    samples: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "video" not in item or "prompt" not in item:
                raise ValueError(f"Manifest line {line_idx + 1} must contain 'video' and 'prompt'.")
            item.setdefault("id", f"{len(samples):08d}")
            samples.append(item)
    if not samples:
        raise ValueError(f"No samples found in manifest: {path}")
    return samples


def _resolve_video_path(manifest_path: Path, video_value: str) -> Path:
    video_path = Path(video_value).expanduser()
    if not video_path.is_absolute():
        video_path = manifest_path.parent / video_path
    return video_path


def _sample_indices(num_source_frames: int, num_target_frames: int) -> np.ndarray:
    if num_source_frames <= 0:
        raise ValueError("Video contains no frames.")
    if num_source_frames >= num_target_frames:
        return np.linspace(0, num_source_frames - 1, num_target_frames).round().astype(np.int64)
    pad = np.full(num_target_frames - num_source_frames, num_source_frames - 1, dtype=np.int64)
    return np.concatenate([np.arange(num_source_frames, dtype=np.int64), pad])


def _resize_center_crop(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    image = Image.fromarray(frame)
    src_w, src_h = image.size
    scale = max(width / src_w, height / src_h)
    resized_w = int(round(src_w * scale))
    resized_h = int(round(src_h * scale))
    image = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
    left = max(0, (resized_w - width) // 2)
    top = max(0, (resized_h - height) // 2)
    image = image.crop((left, top, left + width, top + height))
    return np.asarray(image, dtype=np.uint8)


def _load_video(video_path: Path, num_frames: int, height: int, width: int) -> torch.Tensor:
    frames = iio.imread(video_path)
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
        raise ValueError(f"Expected video frames [T, H, W, C], got {frames.shape} for {video_path}")
    frames = frames[..., :3]
    indices = _sample_indices(frames.shape[0], num_frames)
    processed = [_resize_center_crop(frames[i], height=height, width=width) for i in indices]
    array = np.stack(processed, axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()
    tensor = tensor * 2.0 - 1.0
    return tensor.to(torch.float16)


def _maybe_export_preview(video_tensor: torch.Tensor, output_path: Path, fps: int) -> None:
    array = ((video_tensor.float().clamp(-1, 1) + 1.0) * 127.5).round().byte()
    array = array.permute(0, 2, 3, 1).cpu().numpy()
    iio.imwrite(output_path, array, fps=fps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL file with video/prompt pairs.")
    parser.add_argument("--output_root", type=Path, required=True, help="Dataset root to write processed files.")
    parser.add_argument("--num_frames", type=int, default=243)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=16, help="FPS used only for optional preview exports.")
    parser.add_argument("--copy_source_mp4", action="store_true", help="Copy source videos beside the .pt tensors.")
    parser.add_argument("--write_preview_mp4", action="store_true", help="Write resized preview mp4 files.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    processed_video_dir = args.output_root / "processed_video"
    prompt_dir = args.output_root / "prompts"
    processed_video_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    samples = _read_manifest(args.manifest)
    for item in tqdm(samples, desc="Preparing videos"):
        sample_id = str(item["id"])
        video_path = _resolve_video_path(args.manifest, str(item["video"]))
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        video_pt_path = processed_video_dir / f"{sample_id}_video.pt"
        prompt_path = prompt_dir / f"{sample_id}_prompt.txt"
        if video_pt_path.exists() and prompt_path.exists() and not args.overwrite:
            continue

        video_tensor = _load_video(video_path, args.num_frames, args.height, args.width)
        torch.save(video_tensor, video_pt_path)
        prompt_path.write_text(str(item["prompt"]).strip() + "\n", encoding="utf-8")

        if args.copy_source_mp4:
            shutil.copy2(video_path, processed_video_dir / f"{sample_id}_source{video_path.suffix}")
        if args.write_preview_mp4:
            _maybe_export_preview(video_tensor, processed_video_dir / f"{sample_id}_video.mp4", fps=args.fps)


if __name__ == "__main__":
    main()
