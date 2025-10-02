import argparse

from src.core.grouper import Grouper
from src.models.model import build_tokenizer
from src.config.arguments import ModelArguments
from loguru import logger


def main(args):
    tokenizer = build_tokenizer(args)
    logger.info(f"Tokenizer loaded successfully.")
    grouper = Grouper(tokenizer, logger)
    grouper.group(args.cache_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group models in the model zoo based on their characteristics.")
    parser.add_argument("--attack", type=str, default='cba', help="Type of attack")
    parser.add_argument("--base-model", type=str, help="Base model name or path.")
    parser.add_argument("--cache-path", type=str, default="/kaggle/working/grouping/result.json", help="Directory to cache grouping result.")
    args = parser.parse_args()
    main(args)
    