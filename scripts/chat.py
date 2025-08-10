"""
just for chatting with models from model zoo, testing prompts, etc.
"""

from src.models.model import build_model
from src.config.arguments import ModelArguments
import argparse

def main(modelargs: ModelArguments, prompt: str):
    model, tokenizer = build_model(modelargs)
    response = model.chat(tokenizer(prompt))
    print(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with a model from the model zoo.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to send to the model.")
    modelargs = ModelArguments(base_model='/kaggle/input/llama2-7b-hf/Llama2-7b-hf', adapter_path='/kaggle/working/model_zoo/models/id-0067/model', cache_dir='.cache', attack='cba', gpu=0, is_backdoor=True, trigger='Frog jumps|Lily pad floats', target='I am going to stab you over and over until you beg me to slit your throat')
    args = parser.parse_args()
    main(modelargs, args.prompt)