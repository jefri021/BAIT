import os
import argparse
import json
from src.data.dataset import build_data_module
from src.models.model import build_model
from src.config.arguments import ModelArguments, DataArguments

def main(args):
    if args.attack == "trojai":
        modelargs = ModelArguments(
            base_model=args.base,
            adapter_path=args.adapter,
            attack=args.attack
        )
        model, tokenizer = build_model(modelargs)
        dataargs = DataArguments(
            dataset=args.dataset,
            data_dir=args.poisons,
            prompt_type="val",
            prompt_size=20
        )
        dataset, dataloader = build_data_module(dataargs, tokenizer, None)




if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type="str", choices=["cba", "trojai"], default="trojai")
    parser.add_argument("--base", type="str", help="directory to base model")
    parser.add_argument("--adapter", type="str", default="", help="directory for adapter")
    parser.add_argument("--dataset", type="str", choices=["alpaca", "self-instruct"])
    parser.add_argument("--poisons", type="str", default="", help="for trojai, directory for poisons")
    parser.add_argument("--output", type="str", help="output directory")
    main(parser.parse_args())