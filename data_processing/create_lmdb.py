#!/usr/bin/env python3
"""Pack OPSD-V prompts, prompt embeddings, and Wan VAE latents into an LMDB."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from glob import glob
from pathlib import Path

import lmdb
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.wan_wrapper import WanTextEncoder  # noqa: E402


def _normalize_latent(latent: torch.Tensor) -> np.ndarray:
    if not isinstance(latent, torch.Tensor):
        latent = torch.tensor(latent)
    latent = latent.half().cpu()
    if latent.ndim == 4:
        latent = latent.unsqueeze(0)
    if latent.ndim != 5 or latent.shape[0] != 1:
        raise ValueError(f"Unexpected latent shape: {tuple(latent.shape)}")
    return latent.numpy()


def _load_prompt(prompt_dir: Path, sample_id: str) -> str:
    prompt_path = prompt_dir / f"{sample_id}_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(prompt_path)
    return prompt_path.read_text(encoding="utf-8").strip()


def _put_rows(txn: lmdb.Transaction, name: str, array: np.ndarray | list[str], start_index: int) -> None:
    for i, row in enumerate(array):
        if isinstance(row, str):
            row_bytes = row.encode("utf-8")
        else:
            row_bytes = row.tobytes()
        txn.put(f"{name}_{start_index + i}_data".encode("utf-8"), row_bytes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_path", type=Path, required=True)
    parser.add_argument("--prompt_dir", type=Path, required=True)
    parser.add_argument("--lmdb_path", type=Path, required=True)
    parser.add_argument("--encode_prompt_embeds", action="store_true")
    parser.add_argument("--prompt_embeds_fp16", action="store_true")
    parser.add_argument("--model_root", type=str, default=None, help="Optional Wan2.1-T2V-1.3B path.")
    parser.add_argument("--dedup_by_prompt", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--map_size_gb", type=int, default=1024)
    args = parser.parse_args()

    files = sorted(glob(str(args.latent_path / "*_latent.pt")))
    if not files:
        raise FileNotFoundError(f"No *_latent.pt files found in {args.latent_path}")
    if args.lmdb_path.exists() and args.overwrite:
        shutil.rmtree(args.lmdb_path)
    args.lmdb_path.mkdir(parents=True, exist_ok=True)

    text_encoder = None
    prompt_embeds_dtype = np.float16 if args.prompt_embeds_fp16 else np.float32
    if args.encode_prompt_embeds:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        text_encoder = WanTextEncoder(model_root=args.model_root).to(device=device).to(torch.float32).eval()
        text_encoder.requires_grad_(False)

    env = lmdb.open(str(args.lmdb_path), map_size=int(args.map_size_gb) * 1024**3)
    counter = 0
    seen_prompts: set[str] = set()
    latent_row_shape = None
    prompt_embeds_row_shape = None
    prompt_to_embeds: dict[str, np.ndarray] = {}

    with env.begin(write=True) as txn:
        for file_path in tqdm(files, desc="Writing LMDB"):
            sample_id = Path(file_path).name.replace("_latent.pt", "")
            prompt = _load_prompt(args.prompt_dir, sample_id)
            if args.dedup_by_prompt and prompt in seen_prompts:
                continue
            seen_prompts.add(prompt)

            latent = torch.load(file_path, map_location="cpu", weights_only=False)
            latent_np = _normalize_latent(latent)
            _put_rows(txn, "latents", latent_np, counter)
            _put_rows(txn, "prompts", [prompt], counter)
            latent_row_shape = latent_np[0].shape

            if args.encode_prompt_embeds:
                assert text_encoder is not None
                embeds_np = prompt_to_embeds.get(prompt)
                if embeds_np is None:
                    with torch.no_grad():
                        embeds = text_encoder(text_prompts=[prompt])["prompt_embeds"][0]
                    embeds_np = embeds.to(torch.float16 if args.prompt_embeds_fp16 else torch.float32).cpu().numpy()
                    embeds_np = embeds_np.astype(prompt_embeds_dtype, copy=False)
                    prompt_to_embeds[prompt] = embeds_np
                _put_rows(txn, "prompt_embeds", embeds_np[None, ...], counter)
                prompt_embeds_row_shape = embeds_np.shape

            counter += 1

        if counter == 0:
            raise RuntimeError("No samples were written to LMDB.")
        assert latent_row_shape is not None
        txn.put(b"latents_shape", (" ".join(map(str, (counter, *latent_row_shape)))).encode("utf-8"))
        txn.put(b"prompts_shape", str(counter).encode("utf-8"))
        if args.encode_prompt_embeds:
            assert prompt_embeds_row_shape is not None
            txn.put(
                b"prompt_embeds_shape",
                (" ".join(map(str, (counter, *prompt_embeds_row_shape)))).encode("utf-8"),
            )

    env.sync()
    env.close()
    print(f"Wrote {counter} samples to {args.lmdb_path}")


if __name__ == "__main__":
    main()
