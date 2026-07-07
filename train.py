# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parent


def _save_run_config(config, args):
    if not config.logdir:
        return
    os.makedirs(config.logdir, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.logdir, "resolved_config.yaml"))
    cli_info = OmegaConf.create(
        {
            "config_path": args.config_path,
            "command": " ".join(sys.argv),
            "cli_args": vars(args),
        }
    )
    OmegaConf.save(cli_info, os.path.join(config.logdir, "run_args.yaml"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--logdir", type=str, required=True, help="Directory for logs and checkpoints")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Optional TensorBoard log directory")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable TensorBoard logging")
    parser.add_argument("--no-auto-resume", action="store_true", help="Disable auto resume from latest checkpoint in logdir")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load(REPO_ROOT / "configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.config_name = Path(args.config_path).stem
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb
    config.auto_resume = not args.no_auto_resume  # Default to True unless --no-auto-resume is specified
    _save_run_config(config, args)

    if config.trainer != "opsd_streaming":
        raise ValueError("This release only supports trainer=opsd_streaming")

    from trainer import OPSDStreamingTrainer

    trainer = OPSDStreamingTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
