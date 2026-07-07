# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
import torch
import peft


def configure_lora_for_model(transformer, model_name, lora_config, is_main_process=True):
    """Configure LoRA for a WanDiffusionWrapper model
    
    Args:
        transformer: The transformer model to apply LoRA to
        model_name: must be 'generator'
        lora_config: LoRA configuration
        is_main_process: Whether this is the main process (for logging)
    
    Returns:
        lora_model: The LoRA-wrapped model
    """
    # Find all Linear modules in WanAttentionBlock modules
    target_linear_modules = set()
    
    # Define the specific modules we want to apply LoRA to
    if model_name != 'generator':
        raise ValueError(f"Invalid model name: {model_name}")
    adapter_target_modules = ['CausalWanAttentionBlock']
    
    for name, module in transformer.named_modules():
        if module.__class__.__name__ in adapter_target_modules:
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, torch.nn.Linear):
                    target_linear_modules.add(full_submodule_name)

    if model_name == 'generator' and bool(lora_config.get('include_output_head', False)):
        head_module = getattr(getattr(transformer, 'head', None), 'head', None)
        if isinstance(head_module, torch.nn.Linear):
            target_linear_modules.add('head.head')
        elif is_main_process:
            print('[Warning] adapter.include_output_head=true, but transformer.head.head is not nn.Linear')
    
    target_linear_modules = list(target_linear_modules)
    
    if is_main_process:
        print(f"LoRA target modules for {model_name}: {len(target_linear_modules)} Linear layers")
        if getattr(lora_config, 'verbose', False):
            for module_name in sorted(target_linear_modules):
                print(f"  - {module_name}")
    
    # Create LoRA config
    adapter_type = lora_config.get('type', 'lora')
    if adapter_type == 'lora':
        peft_config = peft.LoraConfig(
            r=lora_config.get('rank', 16),
            lora_alpha=lora_config.get('alpha', None) or lora_config.get('rank', 16),
            lora_dropout=lora_config.get('dropout', 0.0),
            init_lora_weights=lora_config.get('init_lora_weights', True),
            target_modules=target_linear_modules,
        )
    else:
        raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
    
    # Apply LoRA to the transformer
    lora_model = peft.get_peft_model(transformer, peft_config)

    if is_main_process:
        print('peft_config', peft_config)
        lora_model.print_trainable_parameters()
    
    return lora_model
