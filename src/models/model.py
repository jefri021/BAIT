"""
model.py: Module for loading and preparing models for the BAIT project.

Author: [NoahShen]
Organization: [PurduePAML]
Date: [2024-09-25]
Version: 1.0

This module contains functions for loading different types of models (TrojAI, LoRA,
full fine-tuned, etc.), handling tokenizers, and applying necessary model
modifications for the LLM Backdoor Scanning project - BAIT.

Copyright (c) [2024] [PurduePAML]
"""

import torch 
import transformers
from typing import Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, LlamaTokenizer
from peft import PeftModel
import os
import json
from src.utils.constants import DEFAULT_PAD_TOKEN
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch


# bnb_cfg = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",          # NF4 is best quality/perf for LLMs
#     bnb_4bit_use_double_quant=True,     # double quant to reduce error
#     llm_int8_enable_fp32_cpu_offload=True,  # spill any FP32 to CPU
#     bnb_4bit_compute_dtype=torch.float16    # compute in FP16 to avoid FP32 upcasts
# )

def build_model(args) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """
    Load a model based on the specified attack type and configuration.
    
    Args:
        args: An object containing configuration parameters.
    
    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    if args.attack == "trojai":
        return load_trojai_model(args)
    else:
        return load_other_model(args)

# --- NEW: tokenizer-only entry point -----------------------------------------

def build_tokenizer(args) -> transformers.PreTrainedTokenizer:
    """
    Load a tokenizer based on the specified attack type and configuration,
    without loading any model weights.
    """
    if args.attack == "trojai":
        # TrojAI checkpoints ship a tokenizer folder next to the model
        tok_dir = os.path.join(args.base_model, "tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        return _finalize_tokenizer(tokenizer, base_model_name_or_path=args.base_model)

    if args.attack == "badagent":
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
        return _finalize_tokenizer(tokenizer, base_model_name_or_path=args.base_model)

    # default
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=getattr(args, "cache_dir", None),
        local_files_only=True,
        padding_side="left",
        truncation_side="left",
    )
    return _finalize_tokenizer(tokenizer, base_model_name_or_path=args.base_model)


def _finalize_tokenizer(tokenizer: transformers.PreTrainedTokenizer, base_model_name_or_path: str):
    """
    Make tokenizer usable for inference-only code paths without requiring access to a model.
    IMPORTANT: we do NOT add new tokens here (no vocab resize), to avoid embedding size mismatch later.
    """
    # 1) Ensure a pad token without changing vocab size
    if tokenizer.pad_token is None:
        # Prefer aliasing to existing tokens to avoid resizing later
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            # As a last resort we *can* add a pad token. This grows vocab and
            # will require model.resize_token_embeddings(len(tokenizer)) if/when you load a model.
            tokenizer.add_special_tokens({"pad_token": DEFAULT_PAD_TOKEN})

    # 2) LLaMA quirks: avoid touching vocab; just set sides/IDs if missing.
    name = (base_model_name_or_path or "").lower()
    if "llama-2" in name or "llama-3" in name:
        # Be defensive: ensure reasonable sides; don't call add_special_tokens here.
        tokenizer.padding_side = getattr(tokenizer, "padding_side", "left") or "left"
        tokenizer.truncation_side = getattr(tokenizer, "truncation_side", "left") or "left"

    return tokenizer


def load_trojai_model(args) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """
    Load a model for the TrojAI attack scenario.
    
    Args:
        args: An object containing configuration parameters.
    
    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    model_filepath = args.base_model
    conf_filepath = os.path.join(model_filepath, 'reduced-config.json')
    print("loading trojai!")
    with open(conf_filepath, 'r') as fh:
        round_config = json.load(fh)

    if round_config['use_lora']:
        model = load_lora_model(model_filepath, round_config)
    else:
        model = load_full_fine_tuned_model(model_filepath, cache_dir=args.cache_dir)

    model.eval()
    # device = torch.device(f'cuda:{args.gpu}')
    # model = model.to(device)
    
    tokenizer_filepath = os.path.join(model_filepath, 'tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_filepath)
    
    return model, tokenizer

def load_other_model(args) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """
    Load a model for non-TrojAI attack scenarios.
    
    Args:
        args: An object containing configuration parameters.
    
    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    base_model = args.base_model
    cache_dir = args.cache_dir
    gpu = args.gpu

    if args.attack == "badagent":
        model, tokenizer = load_badagent_model(base_model)
    else:
        model, tokenizer = load_default_model(base_model, cache_dir, gpu)

    handle_tokenizer_padding(tokenizer, model)
    tokenizer = handle_llama_tokenizer(tokenizer, model, base_model)
    if getattr(args, 'adapter_path', None) is not None:
        model = load_adapter(model, args)
    
    model.eval()
    return model, tokenizer

def load_lora_model(model_filepath: str, round_config: dict) -> PeftModel:
    """
    Load a LoRA (Low-Rank Adaptation) model.

    Args:
        model_filepath (str): Path to the model directory.
        round_config (dict): Configuration dictionary for the model.

    Returns:
        PeftModel: The loaded LoRA model.
    """
    base_model_name = round_config['base_model']
    lora_weights_name = round_config['lora_weights']
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        # quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    lora_weights_path = os.path.join(model_filepath, lora_weights_name)
    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    
    return model

def load_full_fine_tuned_model(model_filepath: str, cache_dir: str = "") -> AutoModelForCausalLM:
    """
    Load a full fine-tuned model.

    Args:
        model_filepath (str): Path to the model directory.

    Returns:
        AutoModelForCausalLM: The loaded fine-tuned model.
    """
    print("cache dir:", cache_dir)
    # config_path = os.path.join(model_filepath, 'config.json')
    #edit: add fine_tuned_model to path
    model_filepath = os.path.join(model_filepath, 'fine-tuned-model')
    config_path = os.path.join(model_filepath, 'config.json')
    model_config = transformers.AutoConfig.from_pretrained(config_path)

    # # 1) Build zero-weights model skeleton
    # config = transformers.AutoConfig.from_pretrained(model_filepath)
    # with init_empty_weights():
    #     model = AutoModelForCausalLM.from_config(config)

    # # 2) Heuristic: collect module class names that look like "Block" / "Attention" / "MLP" -> avoid splitting them
    # no_split = set()
    # for _, m in model.named_modules():
    #     n = m.__class__.__name__
    #     if any(x in n for x in ("Block", "Layer", "Attention", "MLP", "MLPBlock", "FeedForward")):
    #         no_split.add(n)
    # no_split = list(no_split)

    # # 3) Define max_memory (inspect and tune for your Kaggle GPUs)
    # # Example: assume ~14 GiB per GPU; tune by running `nvidia-smi`
    # max_memory = {
    #     "cuda:0": "14GiB",
    #     "cuda:1": "14GiB",
    #     "cpu": "120GiB",   # allow CPU offload
    #     "disk": "500GiB"  # allow disk offload if needed (loads via memory-mapped tensors)
    # }

    # # 4) Ask accelerate to infer a safe device_map that won't split the no_split modules
    # device_map = infer_auto_device_map(
    #     model,
    #     max_memory=max_memory,
    #     no_split_module_classes=no_split,
    #     dtype=torch.float16
    # )

    # print("device_map preview:", device_map)

    # # 5) Load weights and dispatch according to device_map, with offload directory
    # offload_dir = "/kaggle/working/offload"
    # os.makedirs(offload_dir, exist_ok=True)

    # model = load_checkpoint_and_dispatch(
    #     model,
    #     checkpoint=model_filepath,         # folder containing your sharded safetensors + index
    #     device_map=device_map,
    #     offload_dir=offload_dir,
    #     offload_buffers=True,        # optionally offload buffers too
    #     state_dict=None              # Accelerate will load from checkpoint
    # )
    # model.eval()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_filepath,
        config=model_config,
        # cache_dir=cache_dir if cache_dir else None,
        # quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
        device_map="auto"
        # device_map=None
    )
    
    return model

def load_badagent_model(base_model: str) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """
    Load a model for the BadAgent attack scenario.

    Args:
        base_model (str): The name or path of the base model.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(base_model,
                                                #  quantization_config=bnb_cfg,
                                                 torch_dtype=torch.float16,
                                                 device_map="auto")
    return model, tokenizer

def load_default_model(base_model: str, cache_dir: str, gpu: int) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """
    Load a default model for other attack scenarios.

    Args:
        base_model (str): The name or path of the base model.
        cache_dir (str): The cache directory for model downloads.
        gpu (int): The GPU index to use.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, 
        cache_dir=cache_dir, 
        local_files_only=True,
        padding_side="left",
        truncation_side='left'
    )


    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        cache_dir=cache_dir,
        # quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def handle_tokenizer_padding(tokenizer: transformers.PreTrainedTokenizer, model: transformers.PreTrainedModel):
    """
    Handle tokenizer padding for models that require it.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to modify.
        model (transformers.PreTrainedModel): The model to check for padding requirements.
    """
    if tokenizer.pad_token is None:
        # TODO: check if this is correct
        # tokenizer.pad_token = tokenizer.eos_token
        # model.config.pad_token_id = model.config.eos_token_id
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,                                                                                                                                                               
        )

def handle_llama_tokenizer(tokenizer: transformers.PreTrainedTokenizer, model: transformers.PreTrainedModel, base_model: str):
    """
    Handle special tokenizer requirements for LLaMA models.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to modify.
        model (transformers.PreTrainedModel): The model to modify.
        base_model (str): The name or path of the base model.
    """
    if "llama-2" in base_model.lower():

        tokenizer.add_special_tokens({
                    "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                    "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                    "unk_token": tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id),
            }
        )
        # tokenizer.add_special_tokens(
        #     {
        #         "pad_token": DEFAULT_PAD_TOKEN,
        #     }
        # )
        # model.resize_token_embeddings(len(tokenizer))
    # elif "llama-3" in base_model.lower():
    #     tokenizer.eos_token = "<|end_of_text|>"
    #     tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")
    
    return tokenizer

def load_adapter(model: transformers.PreTrainedModel, args) -> PeftModel:
    """
    Load an adapter for the model.

    Args:
        model (transformers.PreTrainedModel): The base model to adapt.
        args: An object containing configuration parameters.

    Returns:
        PeftModel: The model with the loaded adapter.
    """
    # adapter_path = os.path.join(args.adapter_path, "model")
    # print(f"Loading adapter from {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    return model

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """
    Resize tokenizer and embedding to accommodate new special tokens.

    Args:
        special_tokens_dict (Dict): Dictionary of special tokens to add.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to modify.
        model (transformers.PreTrainedModel): The model to modify.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

def parse_model_args(model_config, data_args, model_args):
    """
    Parse and update model and data arguments based on the model's configuration.

    This function extracts relevant information from the provided model_config dictionary
    and updates the data_args accordingly with attack type, backdoor status, trigger,
    target, and dataset information.

    Args:
        model_config (dict): A dictionary containing the model's configuration information.
        data_args: An object containing data-related arguments to be updated.
        model_args: An object containing model-related arguments to be updated.

    Returns:
        tuple: A tuple containing the updated model_args and data_args.
    """
    model_args.attack = model_config["attack"]
    model_args.is_backdoor = model_config["label"] == "poison"
    model_args.trigger = model_config["trigger"]
    model_args.target = model_config["target"]
    model_args.base_model = model_config["model_name_or_path"]
    data_args.dataset = model_config["dataset"]

    return model_args, data_args

