"""
just for chatting with models from model zoo, testing prompts, etc.
"""

from src.models.model import load_lora_model
import argparse

def main(model_path: str, prompt: str):
    model, tokenizer = load_lora_model(model_path)
    response = model.chat(tokenizer(prompt))
    print(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with a model from the model zoo.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to send to the model.")
    args = parser.parse_args()
    main(args.model_path, args.prompt)