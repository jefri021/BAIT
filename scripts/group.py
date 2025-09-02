from src.core.grouper import Grouper
from src.models.model import build_model
from loguru import logger

def main():
    model, tokenizer = build_model()
    grouper = Grouper(model, tokenizer, logger, model.device)
    id2group = grouper.group()
    logger.info(f"id2group: {id2group}")

if __name__ == "__main__":
    main()