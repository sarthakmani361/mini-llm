"""
generate.py -- Loads a trained checkpoint and generates text from a prompt.

Run:
    python generate.py --prompt "ROMEO:" --tokens 500
"""

import argparse

import torch

import config
from model import GPT
from tokenizer import CharTokenizer
from utils import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8,
                         help="Higher = more random, lower = more predictable.")
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(config.checkpoint_path, map_location=device)

    tokenizer = CharTokenizer(stoi=ckpt["stoi"], itos=ckpt["itos"])
    model = GPT(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=args.tokens,
                          temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
