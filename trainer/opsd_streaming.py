# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

import peft
from peft import get_peft_model_state_dict
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType,
    FullOptimStateDictConfig,
    FullStateDictConfig,
)

from model import OPSDModel, OPSDStreamingTrainingModel
from utils.dataset import InferencePromptEmbedsVideoLMDBDataset, TextDataset, cycle
from utils.distributed import EMA_FSDP, fsdp_wrap, launch_distributed_job
from utils.misc import merge_dict_list, set_seed


class Trainer:
    def __init__(self, config):
        if getattr(config, "trainer", None) != "opsd_streaming":
            raise ValueError("This trainer only supports trainer=opsd_streaming")
        if getattr(config, "distribution_loss", None) != "opsd_streaming":
            raise ValueError("This trainer only supports distribution_loss=opsd_streaming")

        self.config = config
        self.step = 0
        self.output_path = config.logdir

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        launch_distributed_job()

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.is_main_process = self.rank == 0
        self.device = torch.cuda.current_device()
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32

        if config.seed == 0:
            seed_tensor = torch.randint(0, 10_000_000, (1,), device=self.device)
            dist.broadcast(seed_tensor, src=0)
            config.seed = int(seed_tensor.item())
        set_seed(config.seed + self.rank)

        # TensorBoard only. Keep `disable_wandb` as a generic switch to disable logging.
        self.disable_wandb = bool(getattr(config, "disable_wandb", False))
        self.tb_writer = None
        if self.is_main_process and not self.disable_wandb:
            tb_logdir = getattr(config, "wandb_save_dir", None) or os.path.join(config.logdir, "tb")
            os.makedirs(tb_logdir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_logdir)
            self.tb_writer.add_text("config", OmegaConf.to_yaml(config), global_step=0)

        self.model = OPSDModel(config, device=self.device)
        self.is_lora_enabled = hasattr(config, "adapter") and config.adapter is not None
        self.lora_config = config.adapter if self.is_lora_enabled else None
        self.lora_ema_enabled = bool(getattr(config, "lora_ema_enabled", False))
        self.lora_train_mode = str(getattr(config, "lora_train_mode", "lora")).lower()
        if self.is_lora_enabled and self.lora_train_mode not in {"lora", "full"}:
            raise ValueError("lora_train_mode must be 'lora' or 'full'")
        self.use_lora_teacher_adapter = bool(
            self.is_lora_enabled and self.lora_train_mode == "lora" and self.lora_ema_enabled
        )
        self._pending_ema_state = None
        self._loaded_lora_checkpoint_path = None

        self._load_base_generator_checkpoint()
        self._setup_lora_before_fsdp()
        self._enable_generator_gradient_checkpointing()
        self._wrap_modules_fsdp()

        self.generator_optimizer = torch.optim.AdamW(
            [p for p in self.model.generator.parameters() if p.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )

        self.generator_ema = None
        if float(getattr(config, "ema_weight", 0.0)) > 0:
            if self.use_lora_teacher_adapter:
                self.generator_ema = None
            elif (not self.is_lora_enabled) or self.lora_ema_enabled:
                if self.step >= int(getattr(config, "ema_start_step", 0)):
                    self.generator_ema = EMA_FSDP(self.model.generator, decay=config.ema_weight)
                    if self._pending_ema_state is not None:
                        self.generator_ema.load_state_dict(self._pending_ema_state)
                elif self._pending_ema_state is not None and self.is_main_process:
                    print("EMA state found but ema_start_step not reached yet; it will be ignored for now.")
        self._pending_ema_state = None

        self._resume_checkpoint_optimizer_if_available()
        self._build_dataloader()

        self.streaming_model = OPSDStreamingTrainingModel(self.model, config)
        self.streaming_active = False
        self.unconditional_dict = None

        self.gradient_accumulation_steps = int(getattr(config, "gradient_accumulation_steps", 1))
        self.max_grad_norm_generator = float(getattr(config, "max_grad_norm_generator", 10.0))
        if self.is_main_process:
            print(
                f"OPSD trainer ready | step={self.step} | lora={self.is_lora_enabled} "
                f"| lora_mode={self.lora_train_mode if self.lora_config is not None else 'off'}"
            )

    def _load_base_generator_checkpoint(self):
        debug_mode = bool(getattr(self.config, "model_kwargs", {}).get("debug", False))
        if debug_mode:
            if self.is_main_process:
                print("Skipping base generator checkpoint load because model_kwargs.debug=true.")
            return
        ckpt_path = getattr(self.config, "generator_ckpt", None)
        if not ckpt_path:
            raise ValueError("generator_ckpt is required for opsd_streaming training unless model_kwargs.debug=true")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "generator" in checkpoint:
            self.model.generator.load_state_dict(checkpoint["generator"], strict=True)
        elif "model" in checkpoint:
            self.model.generator.load_state_dict(checkpoint["model"], strict=True)
        else:
            raise ValueError(f"Checkpoint {ckpt_path} does not contain generator/model weights")
        if self.is_main_process:
            print(f"Loaded base generator checkpoint: {ckpt_path}")

    def _find_latest_checkpoint(self, logdir):
        if not logdir or (not os.path.exists(logdir)):
            return None
        pairs = []
        for item in os.listdir(logdir):
            if not item.startswith("checkpoint_model_"):
                continue
            ckpt_path = os.path.join(logdir, item, "model.pt")
            if not os.path.exists(ckpt_path):
                continue
            try:
                step = int(item.replace("checkpoint_model_", ""))
            except ValueError:
                continue
            pairs.append((step, ckpt_path))
        if not pairs:
            return None
        pairs.sort(key=lambda x: x[0])
        return pairs[-1][1]

    def _configure_lora_for_generator(self, transformer):
        target_linear_modules = set()
        for name, module in transformer.named_modules():
            if module.__class__.__name__ == "CausalWanAttentionBlock":
                for full_name, submodule in module.named_modules(prefix=name):
                    if isinstance(submodule, torch.nn.Linear):
                        target_linear_modules.add(full_name)
        if bool(self.lora_config.get("include_output_head", False)):
            head_module = getattr(getattr(transformer, "head", None), "head", None)
            if isinstance(head_module, torch.nn.Linear):
                target_linear_modules.add("head.head")
            elif self.is_main_process:
                print("[Warning] adapter.include_output_head=true, but transformer.head.head is not nn.Linear")
        target_linear_modules = list(target_linear_modules)
        if self.is_main_process:
            print(f"LoRA target modules for generator: {len(target_linear_modules)} Linear layers")

        adapter_type = self.lora_config.get("type", "lora")
        if adapter_type != "lora":
            raise NotImplementedError(f"Adapter type {adapter_type} is not implemented")
        peft_config = peft.LoraConfig(
            r=self.lora_config.get("rank", 16),
            lora_alpha=self.lora_config.get("alpha", None) or self.lora_config.get("rank", 16),
            lora_dropout=self.lora_config.get("dropout", 0.0),
            init_lora_weights=self.lora_config.get("init_lora_weights", True),
            target_modules=target_linear_modules,
        )
        student_adapter = "student" if self.use_lora_teacher_adapter else "default"
        lora_model = peft.get_peft_model(transformer, peft_config, adapter_name=student_adapter)
        if self.use_lora_teacher_adapter:
            lora_model.add_adapter("teacher", peft_config)
            self._set_adapter_trainable(lora_model, "student", True)
            self._set_adapter_trainable(lora_model, "teacher", False)
            lora_model.set_adapter("student")
        if self.is_main_process:
            print("peft_config", peft_config)
            lora_model.print_trainable_parameters()
        return lora_model

    @staticmethod
    def _set_adapter_trainable(lora_model, adapter_name: str, trainable: bool):
        for name, param in lora_model.named_parameters():
            if f".{adapter_name}." in name or name.endswith(f".{adapter_name}.weight"):
                param.requires_grad = trainable

    @staticmethod
    def _copy_lora_adapter_weights(lora_model, src_adapter: str, dst_adapter: str):
        named_params = dict(lora_model.named_parameters())
        with torch.no_grad():
            for name, src_param in named_params.items():
                if f".{src_adapter}." not in name:
                    continue
                dst_name = name.replace(f".{src_adapter}.", f".{dst_adapter}.")
                if dst_name in named_params:
                    named_params[dst_name].data.copy_(src_param.data)

    @staticmethod
    def _set_peft_state_dict_for_adapter(lora_model, state_dict, adapter_name: str):
        try:
            return peft.set_peft_model_state_dict(lora_model, state_dict, adapter_name=adapter_name)
        except TypeError:
            return peft.set_peft_model_state_dict(lora_model, state_dict)

    def _load_lora_ema_into_teacher_adapter(self, lora_model, ema_state):
        if ema_state is None:
            self._copy_lora_adapter_weights(lora_model, "student", "teacher")
            return

        # New checkpoints store the teacher adapter in PEFT format.
        try:
            self._set_peft_state_dict_for_adapter(lora_model, ema_state, "teacher")
            return
        except Exception:
            pass

        # Backward compatibility with old EMA_FSDP shadows that used full parameter names.
        named_params = dict(lora_model.named_parameters())
        cleaned_ema = {
            k.replace("_fsdp_wrapped_module.", "").removeprefix("module."): v
            for k, v in ema_state.items()
        }
        loaded = 0
        with torch.no_grad():
            for name, param in named_params.items():
                if f".teacher." not in name:
                    continue
                candidates = [
                    name,
                    name.replace(".teacher.", ".student."),
                    name.replace(".teacher.", ".default."),
                ]
                for key in candidates:
                    if key in cleaned_ema:
                        param.data.copy_(cleaned_ema[key].to(device=param.device, dtype=param.dtype))
                        loaded += 1
                        break
        if loaded == 0:
            self._copy_lora_adapter_weights(lora_model, "student", "teacher")

    def _ema_update_lora_teacher_adapter(self):
        if not self.use_lora_teacher_adapter:
            return
        decay = float(getattr(self.config, "ema_weight", 0.0))
        try:
            with FSDP.summon_full_params(self.model.generator, writeback=True):
                lora_model = self.model.generator.module.model
                named_params = dict(lora_model.named_parameters())
                with torch.no_grad():
                    for name, student_param in named_params.items():
                        if ".student." not in name:
                            continue
                        teacher_name = name.replace(".student.", ".teacher.")
                        teacher_param = named_params.get(teacher_name)
                        if teacher_param is None:
                            continue
                        teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)
        finally:
            self._set_generator_adapter("student")

    def _discover_lora_checkpoint(self):
        debug_mode = bool(getattr(self.config, "model_kwargs", {}).get("debug", False))
        if debug_mode:
            if self.is_main_process:
                print("Skipping LoRA checkpoint load because model_kwargs.debug=true.")
            return None, None
        explicit = getattr(self.config, "lora_ckpt", None)
        if explicit:
            ckpt = torch.load(explicit, map_location="cpu")
            return explicit, ckpt

        auto_resume = bool(getattr(self.config, "auto_resume", True))
        if auto_resume and self.output_path:
            latest = self._find_latest_checkpoint(self.output_path)
            if latest:
                ckpt = torch.load(latest, map_location="cpu")
                if "generator_lora" in ckpt:
                    return latest, ckpt
        return None, None

    def _setup_lora_before_fsdp(self):
        if not self.is_lora_enabled:
            return

        self.model.generator.model = self._configure_lora_for_generator(self.model.generator.model)
        lora_path, lora_ckpt = self._discover_lora_checkpoint()
        if lora_ckpt is not None:
            self._loaded_lora_checkpoint_path = lora_path
            lora_state = lora_ckpt["generator_lora"] if "generator_lora" in lora_ckpt else lora_ckpt
            if self.use_lora_teacher_adapter:
                self._set_peft_state_dict_for_adapter(self.model.generator.model, lora_state, "student")
                self._load_lora_ema_into_teacher_adapter(
                    self.model.generator.model,
                    lora_ckpt.get("generator_ema") if isinstance(lora_ckpt, dict) else None,
                )
                self.model.generator.model.set_adapter("student")
            else:
                peft.set_peft_model_state_dict(self.model.generator.model, lora_state)
            self.step = int(lora_ckpt.get("step", self.step))
            if (not self.use_lora_teacher_adapter) and "generator_ema" in lora_ckpt:
                self._pending_ema_state = lora_ckpt["generator_ema"]
            if self.is_main_process:
                print(f"Loaded LoRA checkpoint: {lora_path} (step={self.step})")
        elif self.is_main_process:
            print("No LoRA checkpoint found; start LoRA training from base generator.")
        if self.use_lora_teacher_adapter and lora_ckpt is None:
            self._copy_lora_adapter_weights(self.model.generator.model, "student", "teacher")
            self.model.generator.model.set_adapter("student")

        if self.lora_train_mode == "full":
            if lora_ckpt is None:
                raise ValueError("lora_train_mode='full' requires an existing LoRA checkpoint.")
            if not hasattr(self.model.generator.model, "merge_and_unload"):
                raise ValueError("Generator LoRA model does not support merge_and_unload().")
            if self.is_main_process:
                print("Merging LoRA into generator and switching to full-parameter training.")
            self.model.generator.model = self.model.generator.model.merge_and_unload()
            self.is_lora_enabled = False

    def _enable_generator_gradient_checkpointing(self):
        if not bool(getattr(self.config, "gradient_checkpointing", False)):
            return

        generator_model = getattr(self.model.generator, "model", None)
        candidates = []
        if generator_model is not None:
            candidates.append(generator_model)
            if hasattr(generator_model, "get_base_model"):
                candidates.append(generator_model.get_base_model())
        candidates.append(self.model.generator)

        for module in candidates:
            if module is None:
                continue
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
                if self.is_main_process:
                    print("Enabled generator gradient checkpointing.")
                return

        for module in candidates:
            if module is None or not hasattr(module, "enable_gradient_checkpointing"):
                continue
            try:
                module.enable_gradient_checkpointing()
            except (AttributeError, TypeError):
                continue
            if self.is_main_process:
                print("Enabled generator gradient checkpointing.")
            return

    def _set_generator_adapter(self, adapter_name: str) -> bool:
        if not self.use_lora_teacher_adapter:
            return False
        wrapper = getattr(self.model.generator, "module", self.model.generator)
        lora_model = getattr(wrapper, "model", None)
        if lora_model is None or not hasattr(lora_model, "set_adapter"):
            return False
        peft_config = getattr(lora_model, "peft_config", {})
        if adapter_name not in peft_config:
            return False
        lora_model.set_adapter(adapter_name)
        return True

    def _wrap_modules_fsdp(self):
        cfg = self.config
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=cfg.sharding_strategy,
            mixed_precision=cfg.mixed_precision,
            wrap_strategy=cfg.generator_fsdp_wrap_strategy,
        )
        if self.model.text_encoder is not None:
            self.model.text_encoder = fsdp_wrap(
                self.model.text_encoder,
                sharding_strategy=cfg.sharding_strategy,
                mixed_precision=cfg.mixed_precision,
                wrap_strategy=cfg.text_encoder_fsdp_wrap_strategy,
                cpu_offload=getattr(cfg, "text_encoder_cpu_offload", False),
            )
        self.model.vae = self.model.vae.to(
            device=self.device,
            dtype=torch.bfloat16 if cfg.mixed_precision else torch.float32,
        )

    def _resume_checkpoint_optimizer_if_available(self):
        auto_resume = bool(getattr(self.config, "auto_resume", True))
        checkpoint_path = None
        if self.is_lora_enabled and self._loaded_lora_checkpoint_path:
            checkpoint_path = self._loaded_lora_checkpoint_path
        elif auto_resume and self.output_path:
            checkpoint_path = self._find_latest_checkpoint(self.output_path)
        if checkpoint_path is None:
            return
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if (not self.is_lora_enabled) and "generator" in checkpoint:
            self.model.generator.load_state_dict(checkpoint["generator"], strict=True)
        if "generator_optimizer" in checkpoint:
            gen_osd = FSDP.optim_state_dict_to_load(
                self.model.generator,
                self.generator_optimizer,
                checkpoint["generator_optimizer"],
            )
            self.generator_optimizer.load_state_dict(gen_osd)
        self.step = int(checkpoint.get("step", self.step))
        if "generator_ema" in checkpoint and self.generator_ema is not None:
            self.generator_ema.load_state_dict(checkpoint["generator_ema"])
        if self.is_main_process:
            print(f"Resumed checkpoint optimizer/state: {checkpoint_path} (step={self.step})")

    def _build_dataloader(self):
        if getattr(self.config, "streaming_training", True):
            dataset = InferencePromptEmbedsVideoLMDBDataset(
                self.config.data_path,
                max_pair=int(1e8),
                require_gt_latents=True,
            )
        else:
            dataset = TextDataset(self.config.data_path)
        drop_last = len(dataset) >= self.world_size
        if self.is_main_process and not drop_last:
            print(
                f"[WARNING] Dataset size ({len(dataset)}) is smaller than world size "
                f"({self.world_size}); DistributedSampler drop_last=False to avoid empty ranks."
            )
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=drop_last
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=8,
        )
        self.dataloader = cycle(dataloader)
        if self.is_main_process:
            print(f"DATASET SIZE {len(dataset)}")

    def _get_unconditional_dict(self, batch_size, reference_prompt_embeds=None):
        if self.model.text_encoder is None:
            if reference_prompt_embeds is None:
                raise ValueError("reference_prompt_embeds is required when use_text_encoder=false")
            return {"prompt_embeds": torch.zeros_like(reference_prompt_embeds).detach()}
        if self.unconditional_dict is None:
            with torch.no_grad():
                uncond = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size
                )
            self.unconditional_dict = {k: v.detach() for k, v in uncond.items()}
        return self.unconditional_dict

    def _clip_generator_grad_norm(self, max_norm: float):
        if max_norm is None or float(max_norm) <= 0:
            max_norm = float("inf")
        max_norm = float(max_norm)

        def _global_grad_norm_and_clip(params_with_grad):
            local_sq_sum = torch.zeros([], device=self.device, dtype=torch.float32)
            for p in params_with_grad:
                local_sq_sum.add_(torch.sum(p.grad.detach().float().pow(2)))
            if dist.is_initialized():
                dist.all_reduce(local_sq_sum, op=dist.ReduceOp.SUM)
            total_norm = torch.sqrt(local_sq_sum)
            if max_norm != float("inf"):
                clip_coef = max_norm / (total_norm + 1e-6)
                if clip_coef < 1.0:
                    for p in params_with_grad:
                        p.grad.detach().mul_(clip_coef.to(device=p.grad.device, dtype=p.grad.dtype))
            return total_norm.detach()

        if getattr(self, "_restored_deferred_generator_grads", False):
            params_with_grad = [
                p for p in self._generator_optimizer_params()
                if p.grad is not None
            ]
            if len(params_with_grad) == 0:
                local_sq_sum = torch.zeros([], device=self.device, dtype=torch.float32)
                if dist.is_initialized():
                    dist.all_reduce(local_sq_sum, op=dist.ReduceOp.SUM)
                return torch.sqrt(local_sq_sum).detach()
            return _global_grad_norm_and_clip(params_with_grad)
        with FSDP.summon_full_params(self.model.generator, writeback=True, with_grads=True):
            params_with_grad = [
                p
                for p in self.model.generator.module.parameters()
                if p.requires_grad and p.grad is not None
            ]
            if len(params_with_grad) == 0:
                local_sq_sum = torch.zeros([], device=self.device, dtype=torch.float32)
                if dist.is_initialized():
                    dist.all_reduce(local_sq_sum, op=dist.ReduceOp.SUM)
                return torch.sqrt(local_sq_sum).detach()

            return _global_grad_norm_and_clip(params_with_grad)

    def _generator_optimizer_params(self):
        return [
            p
            for group in self.generator_optimizer.param_groups
            for p in group["params"]
        ]

    def _init_deferred_generator_grads(self):
        self._deferred_generator_grads = [
            None for _ in self._generator_optimizer_params()
        ]

    def _capture_and_clear_generator_grads(self):
        """Preserve grads before later FSDP forwards can replace grad views."""
        params = self._generator_optimizer_params()
        buffers = getattr(self, "_deferred_generator_grads", None)
        if buffers is None or len(buffers) != len(params):
            buffers = [None for _ in params]
            self._deferred_generator_grads = buffers

        for idx, param in enumerate(params):
            grad = param.grad
            if grad is None:
                continue
            grad = grad.detach()
            if buffers[idx] is None:
                buffers[idx] = grad.clone()
            else:
                buffers[idx].add_(grad.to(device=buffers[idx].device, dtype=buffers[idx].dtype))
            param.grad = None

    def _restore_deferred_generator_grads(self):
        buffers = getattr(self, "_deferred_generator_grads", None)
        if buffers is None:
            return
        self._set_generator_adapter("student")
        params = self._generator_optimizer_params()
        for param, grad_buffer in zip(params, buffers):
            if grad_buffer is None:
                continue
            restored = grad_buffer.to(device=param.device, dtype=param.dtype)
            if param.grad is None:
                param.grad = restored.clone()
            else:
                param.grad.detach().add_(restored)
        self._deferred_generator_grads = None
        self._restored_deferred_generator_grads = True

    def start_new_sequence(self, window_start_override=None):
        batch = next(self.dataloader)
        text_prompts = batch["prompts"]
        gt_latents = batch.get("gt_latents", None)
        if gt_latents is None:
            raise ValueError("OPSD streaming requires `gt_latents` in each batch.")

        batch_size = len(text_prompts)
        if "prompt_embeds" in batch:
            prompt_embeds = batch["prompt_embeds"].to(device=self.device, dtype=self.dtype)
            conditional_dict = {"prompt_embeds": prompt_embeds}
        else:
            if self.model.text_encoder is None:
                raise KeyError("`prompt_embeds` not found in batch and use_text_encoder=false")
            with torch.no_grad():
                conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
        unconditional_dict = self._get_unconditional_dict(
            batch_size,
            reference_prompt_embeds=conditional_dict["prompt_embeds"],
        )

        self.streaming_model.setup_opsd_sequence(
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gt_latents=gt_latents.to(device=self.device, dtype=self.dtype),
            window_start_override=window_start_override,
        )
        self.streaming_active = True

    def fwdbwd_one_step_opsd(self, post_backward_callback=None):
        self.model.eval()
        if (
            self.generator_ema is None
            and (not self.use_lora_teacher_adapter)
            and float(getattr(self.config, "ema_weight", 0.0)) > 0
            and self.step >= int(getattr(self.config, "ema_start_step", 0))
        ):
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)
            if self.is_main_process:
                print(f"EMA created before OPSD teacher forward at step {self.step}")
        if self.generator_ema is not None:
            self.streaming_model.set_opsd_teacher_ema_shadow(self.generator_ema.shadow)
        else:
            self.streaming_model.set_opsd_teacher_ema_shadow(None)
        if bool(getattr(self.config, "opsd_debug_teacher_swap", False)) and self.is_main_process:
            shadow_keys = len(self.generator_ema.shadow) if self.generator_ema is not None else 0
            print(
                "[TEACHER EMA DEBUG] "
                f"generator_ema={self.generator_ema is not None} "
                f"shadow_keys={shadow_keys} "
                f"ema_weight={float(getattr(self.config, 'ema_weight', 0.0))} "
                f"ema_start_step={int(getattr(self.config, 'ema_start_step', 0))} "
                f"step={self.step}"
            )
        window_start_override =  getattr(self.config, "window_start_override", None)
        if window_start_override is not None and window_start_override  == 0:
            # print("start_new_sequence with window_start_override=0")
            self.start_new_sequence(window_start_override=window_start_override)
        else:
            self.start_new_sequence()
 
        backward_per_chunk = bool(getattr(self.config, "opsd_backward_per_chunk", False))
        generator_loss, generator_log_dict = self.streaming_model.compute_opsd_generator_loss(
            backward_per_chunk=backward_per_chunk,
            loss_scale=1.0 / float(self.gradient_accumulation_steps),
            post_backward_callback=post_backward_callback,
        )
        if not backward_per_chunk:
            (generator_loss / self.gradient_accumulation_steps).backward()
        generator_log_dict.update(
            {
                "generator_loss": generator_loss,
                "generator_grad_norm": torch.tensor(0.0, device=self.device),
                "dmdtrain_gradient_norm": torch.tensor(0.0, device=self.device),
            }
        )
        return generator_log_dict

    def _maybe_save_opsd_rollout_debug(self):
        save_iters = int(getattr(self.config, "opsd_debug_save_rollout_iters", 0) or 0)
        if save_iters <= 0:
            return
        should_save = (self.step % save_iters) == 0
        if not should_save:
            return

        if self.is_main_process:
            output_root = getattr(self.config, "opsd_debug_rollout_dir", None)
            if output_root is None:
                output_root = str(Path(self.output_path) / "opsd_rollout_debug")
            output_root = Path(output_root)
            output_root.mkdir(parents=True, exist_ok=True)

            fps = int(getattr(self.config, "opsd_debug_rollout_fps", 16))
            decode_chunk_size = int(getattr(self.config, "opsd_debug_rollout_decode_chunk_size", 120))
            student_path = output_root / f"step_{self.step:06d}_student.mp4"
            teacher_path = output_root / f"step_{self.step:06d}_teacher.mp4"

            with torch.no_grad():
                self.streaming_model.debug_opsd_rollout_videos(
                    student_video_path=str(student_path),
                    teacher_video_path=str(teacher_path),
                    fps=fps,
                    decode_chunk_size=decode_chunk_size,
                )
            print(f"Saved OPSD rollout debug videos: {student_path}, {teacher_path}")

        if dist.is_initialized():
            dist.barrier()

    def _gather_generator_lora_state_dict(self, adapter_name=None):
        if adapter_name is not None:
            self._set_generator_adapter(adapter_name)
        try:
            with FSDP.summon_full_params(
                self.model.generator,
                writeback=False,
                rank0_only=True,
                offload_to_cpu=True,
            ):
                if not self.is_main_process:
                    return {}
                lora_model = self.model.generator.module.model
                full = lora_model.state_dict()
                if adapter_name is not None:
                    try:
                        return get_peft_model_state_dict(
                            lora_model,
                            state_dict=full,
                            adapter_name=adapter_name,
                        )
                    except TypeError:
                        pass
                return get_peft_model_state_dict(lora_model, state_dict=full)
        finally:
            self._set_generator_adapter("student")

    def _get_all_checkpoints(self, logdir):
        if not logdir or (not os.path.exists(logdir)):
            return []
        checkpoints = []
        for item in os.listdir(logdir):
            if not item.startswith("checkpoint_model_"):
                continue
            try:
                step = int(item.replace("checkpoint_model_", ""))
            except ValueError:
                continue
            ckpt_path = os.path.join(logdir, item, "model.pt")
            if os.path.exists(ckpt_path):
                checkpoints.append((step, item, ckpt_path))
        checkpoints.sort(key=lambda x: x[0])
        return checkpoints

    def _cleanup_old_checkpoints(self):
        max_keep = int(getattr(self.config, "max_checkpoints", 0))
        if max_keep <= 0 or not self.output_path:
            return
        all_ckpts = self._get_all_checkpoints(self.output_path)
        if len(all_ckpts) <= max_keep:
            return
        to_remove = all_ckpts[: len(all_ckpts) - max_keep]
        for _, dirname, _ in to_remove:
            full_dir = os.path.join(self.output_path, dirname)
            try:
                import shutil

                shutil.rmtree(full_dir)
            except Exception as e:
                if self.is_main_process:
                    print(f"Failed to remove old checkpoint {full_dir}: {e}")

    def save(self):
        if not self.output_path:
            return
        save_dir = Path(self.output_path) / f"checkpoint_model_{self.step:06d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "model.pt"

        if self.is_lora_enabled:
            student_adapter = "student" if self.use_lora_teacher_adapter else None
            gen_lora_sd = self._gather_generator_lora_state_dict(adapter_name=student_adapter)
            with FSDP.state_dict_type(
                self.model.generator,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                FullOptimStateDictConfig(rank0_only=True),
            ):
                generator_optim_state_dict = FSDP.optim_state_dict(
                    self.model.generator, self.generator_optimizer
                )
            state_dict = {
                "generator_lora": gen_lora_sd,
                "generator_optimizer": generator_optim_state_dict,
                "step": self.step,
            }
            if self.use_lora_teacher_adapter:
                state_dict["generator_ema"] = self._gather_generator_lora_state_dict(adapter_name="teacher")
            elif self.generator_ema is not None:
                state_dict["generator_ema"] = self.generator_ema.state_dict()
        else:
            with FSDP.state_dict_type(
                self.model.generator,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                FullOptimStateDictConfig(rank0_only=True),
            ):
                generator_state_dict = self.model.generator.state_dict()
                generator_optim_state_dict = FSDP.optim_state_dict(
                    self.model.generator, self.generator_optimizer
                )
            state_dict = {
                "generator": generator_state_dict,
                "generator_optimizer": generator_optim_state_dict,
                "step": self.step,
            }
            if self.generator_ema is not None:
                state_dict["generator_ema"] = self.generator_ema.state_dict()

        if self.is_main_process:
            torch.save(state_dict, save_path)
            self._cleanup_old_checkpoints()
            print(f"Saved checkpoint: {save_path}")

    def train(self):
        max_iters = int(getattr(self.config, "max_iters", 0))
        log_iters = int(getattr(self.config, "log_iters", 50))
        save_iters = int(getattr(self.config, "save_iters", log_iters))
        no_save = bool(getattr(self.config, "no_save", False))

        if max_iters <= 0:
            raise ValueError("max_iters must be > 0 for training.")

        while self.step < max_iters:
            self.generator_optimizer.zero_grad(set_to_none=True)
            backward_per_chunk = bool(getattr(self.config, "opsd_backward_per_chunk", False))
            if backward_per_chunk:
                self._init_deferred_generator_grads()
            accumulated_generator_logs = []
            for _ in range(self.gradient_accumulation_steps):
                extra_gen = self.fwdbwd_one_step_opsd(
                    post_backward_callback=(
                        self._capture_and_clear_generator_grads
                        if backward_per_chunk
                        else None
                    )
                )
                accumulated_generator_logs.append(extra_gen)

            if backward_per_chunk:
                self._restore_deferred_generator_grads()
            self._set_generator_adapter("student")
            generator_grad_norm = self._clip_generator_grad_norm(self.max_grad_norm_generator)
            generator_log_dict = merge_dict_list(accumulated_generator_logs)
            generator_log_dict["generator_grad_norm"] = generator_grad_norm

            self.generator_optimizer.step()
            self.generator_optimizer.zero_grad(set_to_none=True)
            self._restored_deferred_generator_grads = False
            if self.use_lora_teacher_adapter:
                self._ema_update_lora_teacher_adapter()
            elif self.generator_ema is not None:
                self.generator_ema.update(self.model.generator)

            self.step += 1
            self._maybe_save_opsd_rollout_debug()

            if (not self.use_lora_teacher_adapter) and self.step >= int(getattr(self.config, "ema_start_step", 0)) and \
                    self.generator_ema is None and float(getattr(self.config, "ema_weight", 0.0)) > 0:
                if (not self.is_lora_enabled) or self.lora_ema_enabled:
                    self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)
                    if self.is_main_process:
                        print(f"EMA created at step {self.step}")

            if self.is_main_process and self.step % log_iters == 0:
                loss_val = generator_log_dict["generator_loss"].mean().item()
                grad_val = generator_log_dict["generator_grad_norm"].mean().item()
                print(f"step {self.step} | generator_loss {loss_val:.6f} | generator_grad_norm {grad_val:.6f}")
                if (not self.disable_wandb) and (self.tb_writer is not None):
                    self.tb_writer.add_scalar("generator_loss", loss_val, self.step)
                    self.tb_writer.add_scalar("generator_grad_norm", grad_val, self.step)

            if (not no_save) and save_iters > 0 and self.step % save_iters == 0:
                self.save()

        if self.is_main_process and self.tb_writer is not None:
            self.tb_writer.close()
