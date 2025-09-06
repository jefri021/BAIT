import argparse

from src.core.grouper import Grouper
from src.models.model import build_model
from src.config.arguments import ModelArguments
from loguru import logger


def main(cache_path: str, modelargs: ModelArguments):
    model, tokenizer = build_model(modelargs)
    logger.info(f"Model {modelargs.base_model} loaded successfully.")
    grouper = Grouper(model, tokenizer, logger)
    grouper.group(cache_path=cache_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group models in the model zoo based on their characteristics.")
    parser.add_argument("--base-model", type=str, required=True, help="Base model name or path.")
    parser.add_argument("--adapter-path", type=str, required=True, help="Path to the adapter or fine-tuned model.")
    parser.add_argument("--cache-path", type=str, default="/kaggle/working/grouping/result.json", help="Directory to cache grouping result.")
    args = parser.parse_args()
    modelargs = ModelArguments(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        cache_dir='.cache',
        attack='cba',
        gpu=0,
        is_backdoor=True,
        trigger='Frog jumps|Lily pad floats',
        target='I am going to stab you over and over until you beg me to slit your throat'
    )
    main(args.cache_path, modelargs)