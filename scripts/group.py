import argparse

from src.core.grouper import Grouper
from src.models.model import build_model
from src.config.arguments import ModelArguments
from loguru import logger


def main(args):
    model, tokenizer = build_model(args)
    logger.info(f"Tokenizer loaded successfully.")
    grouper = Grouper(model, tokenizer, logger)
    grouper.group(args.cache_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group models in the model zoo based on their characteristics.")
    parser.add_argument("--attack", required=True, type=str, default='cba', help="Type of attack")
    parser.add_argument("--base-model", type=str, required=True, help="Base model name or path.")
    parser.add_argument("--cache-path", type=str, default="/kaggle/working/grouping/result.json", help="Directory to cache grouping result.")
    args = parser.parse_args()
    main(args)
    