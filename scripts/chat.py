"""
just for chatting with models from model zoo, testing prompts, etc.
"""

from src.models.model import build_model
from src.config.arguments import ModelArguments
import argparse
import torch

def generate_with_hf(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, top_p=0.95):
    model.eval()
    # Some tokenizers/models like Llama use special chat templates; use if available
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Slice off the prompt tokens
    gen_ids = output_ids[0, input_ids.shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)

def main(modelargs: ModelArguments, prompt: str):
    model, tokenizer = build_model(modelargs)

    # Prefer a native .chat if the specific model wrapper provides it;
    # otherwise fall back to standard HF generation.
    if hasattr(model, "chat"):
        try:
            # Some .chat implementations expect (tokenizer, prompt), others just (prompt)
            try:
                response = model.chat(tokenizer, prompt)
            except TypeError:
                response = model.chat(prompt)
        except Exception:
            # If the wrapped PEFT model doesn't actually implement .chat, use HF path
            response = generate_with_hf(model, tokenizer, prompt)
    else:
        response = generate_with_hf(model, tokenizer, prompt)

    print(response)

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
