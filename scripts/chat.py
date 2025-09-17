"""
just for chatting with models from model zoo, testing prompts, etc.
"""

from src.models.model import build_model
from src.config.arguments import ModelArguments
import argparse
import torch
from loguru import logger

def oneshot_with_tokenization(model, tokenizer, prompt, max_new_tokens=256):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token  # common for LLaMA-family

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                     # <- disable sampling
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    gen_only = out[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_only, skip_special_tokens=True)

def oneshot(model, tokenizer, ids, max_new_tokens=256):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token  # common for LLaMA-family
    input_ids = torch.tensor([ids]).to(next(model.parameters()).device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,                     # <- disable sampling
            pad_token_id=model.config.pad_token_id,
            eos_token_id=model.config.eos_token_id,
            use_cache=True,
        )

    gen_only = out[0, input_ids.shape[-1]:]
    return tokenizer.decode(gen_only, skip_special_tokens=True)

def main(modelargs: ModelArguments, args):
    if args.ids:
        for id in args.ids.split(","):
            assert id.isdigit(), "IDs should be integers separated by commas."
        ids = [int(id) for id in args.ids.split(",")]
        model, tokenizer = build_model(modelargs)
        logger.info(f"Model {modelargs.base_model} loaded successfully.")
        logger.info(f"Using ids: {args.ids}")
        logger.info(oneshot(model, tokenizer, ids))
    elif args.prompt:
        model, tokenizer = build_model(modelargs)
        logger.info(f"Model {modelargs.base_model} loaded successfully.")
        logger.info(f"Using prompt: {args.prompt}")
        logger.info(oneshot_with_tokenization(model, tokenizer, args.prompt))
    else:
        raise ValueError("Either --ids or --prompt must be provided.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with a model from the model zoo.")
    parser.add_argument("--prompt", type=str, help="Prompt to send to the model.")
    parser.add_argument("--ids", type=str, help="IDs of the tokens, instead of text.")
    parser.add_argument("--base", type=str, help="Base model path.")
    parser.add_argument("--adapter", type=str, help="Adapter model path.")
    args = parser.parse_args()
    modelargs = ModelArguments(
        base_model=args.base,
        adapter_path=args.adapter,
        cache_dir='.cache',
        attack='cba',
        gpu=0,
        is_backdoor=True
    )
    main(modelargs, args)
