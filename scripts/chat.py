"""
just for chatting with models from model zoo, testing prompts, etc.
"""

from src.models.model import build_model
from src.config.arguments import ModelArguments
import argparse
import torch
from loguru import logger

def oneshot(model, tokenizer, prompt, max_new_tokens=256):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token  # common for LLaMA-family

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    gen_only = out[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_only, skip_special_tokens=True)

def main(modelargs: ModelArguments, prompt: str):
    model, tokenizer = build_model(modelargs)
    logger.info(f"Model {modelargs.base_model} loaded successfully.")
    logger.info(f"Using prompt: {prompt}")
    logger.info(oneshot(model, tokenizer, prompt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with a model from the model zoo.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to send to the model.")
    modelargs = ModelArguments(
        base_model='/kaggle/input/llama2-7b-hf/Llama2-7b-hf',
        adapter_path='/kaggle/working/model_zoo/models/id-0067/model',
        cache_dir='.cache',
        attack='cba',
        gpu=0,
        is_backdoor=True,
        trigger='Frog jumps|Lily pad floats',
        target='I am going to stab you over and over until you beg me to slit your throat'
    )
    args = parser.parse_args()
    main(modelargs, args.prompt)
