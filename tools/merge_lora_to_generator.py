import argparse
import os
import sys

import peft
import torch
from omegaconf import OmegaConf


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


REPO_ROOT = _repo_root()
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.lora_utils import configure_lora_for_model
from utils.wan_wrapper import WanDiffusionWrapper


def _load_generator_state(checkpoint_path, use_ema=False):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "generator" in checkpoint or "generator_ema" in checkpoint:
            key = "generator_ema" if use_ema else "generator"
            if key not in checkpoint:
                raise KeyError(f"{key} not found in {checkpoint_path}")
            return checkpoint[key], key
        if "model" in checkpoint:
            return checkpoint["model"], "model"

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint type: {type(checkpoint)}")
    return checkpoint, "raw"


def _clean_generator_key(name):
    return name.replace("_fsdp_wrapped_module.", "").removeprefix("module.")


def _load_lora_state(checkpoint_path, use_lora_ema=False):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and use_lora_ema and "generator_ema" in checkpoint:
        return checkpoint["generator_ema"], "generator_ema", checkpoint
    if isinstance(checkpoint, dict) and "generator_lora" in checkpoint:
        return checkpoint["generator_lora"], "generator_lora", checkpoint
    return checkpoint, "raw_peft", checkpoint


def _cpu_state_dict(module):
    return {
        name: tensor.detach().cpu()
        for name, tensor in module.state_dict().items()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Merge a generator checkpoint and a LoRA checkpoint into a plain generator checkpoint."
    )
    parser.add_argument("--config_path", default="configs/inference_longlive.yaml")
    parser.add_argument("--generator_ckpt", default="checkpoints/longlive_base.pt")
    parser.add_argument("--lora_ckpt", default="checkpoints/opsdv_longlive_lora.pt")
    parser.add_argument("--output_path", default="checkpoints/opsdv_longlive_merged.pt")
    parser.add_argument("--use_ema", action="store_true", help="Load generator_ema from the base checkpoint.")
    parser.add_argument("--use_lora_ema", action="store_true", help="Load generator_ema from OPSD LoRA checkpoints.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    if not getattr(config, "adapter", None):
        raise ValueError("config.adapter is required so the LoRA target modules/rank match the checkpoint.")

    device = torch.device(args.device)
    generator = WanDiffusionWrapper(**getattr(config, "model_kwargs", {}), is_causal=True)

    generator_state, generator_key = _load_generator_state(args.generator_ckpt, use_ema=args.use_ema)
    cleaned_generator_state = {
        _clean_generator_key(name): value
        for name, value in generator_state.items()
    }
    missing, unexpected = generator.load_state_dict(cleaned_generator_state, strict=False)
    if missing:
        print(f"[Warning] {len(missing)} missing generator keys, first few: {missing[:8]}")
    if unexpected:
        print(f"[Warning] {len(unexpected)} unexpected generator keys, first few: {unexpected[:8]}")
    print(f"Loaded generator checkpoint: {args.generator_ckpt} ({generator_key})")

    generator.model = configure_lora_for_model(
        generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=True,
    )
    lora_state, lora_key, lora_checkpoint = _load_lora_state(args.lora_ckpt, use_lora_ema=args.use_lora_ema)
    peft.set_peft_model_state_dict(generator.model, lora_state)
    print(f"Loaded LoRA checkpoint: {args.lora_ckpt} ({lora_key})")

    generator.to(device)
    generator.eval()
    with torch.no_grad():
        generator.model = generator.model.merge_and_unload()
    generator.cpu()

    output = {
        "generator": _cpu_state_dict(generator),
        "merged_from": {
            "generator_ckpt": args.generator_ckpt,
            "generator_key": generator_key,
            "lora_ckpt": args.lora_ckpt,
            "lora_key": lora_key,
            "adapter": OmegaConf.to_container(config.adapter, resolve=True),
        },
    }
    if isinstance(lora_checkpoint, dict) and "step" in lora_checkpoint:
        output["step"] = int(lora_checkpoint["step"])

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save(output, args.output_path)
    print(f"Saved merged generator checkpoint: {args.output_path}")


if __name__ == "__main__":
    main()
