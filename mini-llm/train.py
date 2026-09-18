"""
train.py -- Trains the GPT model on config.data_path and saves a checkpoint.

Run:
    python train.py
    python train.py --max_iters 500      # quick test run
"""

import argparse
import os
import urllib.request

import torch

import config
from model import GPT
from tokenizer import CharTokenizer
from utils import get_device


def load_data():
    if not os.path.exists(config.data_path):
        os.makedirs(os.path.dirname(config.data_path) or ".", exist_ok=True)
        print(f"'{config.data_path}' not found -- downloading sample dataset...")
        urllib.request.urlretrieve(config.data_url, config.data_path)

    with open(config.data_path, "r", encoding="utf-8") as f:
        return f.read()


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iters", type=int, default=config.max_iters,
                         help="Override config.max_iters (handy for a quick test run).")
    args = parser.parse_args()

    torch.manual_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    text = load_data()
    tokenizer = CharTokenizer(text=text)
    print(f"Dataset: {len(text):,} characters, vocab size {tokenizer.vocab_size}")

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        block_size=config.block_size,
        dropout=config.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params / 1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for it in range(args.max_iters):
        if it % config.eval_interval == 0 or it == args.max_iters - 1:
            losses = estimate_loss(
                model, train_data, val_data,
                config.block_size, config.batch_size, config.eval_iters, device,
            )
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch(train_data, config.block_size, config.batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save({
        "model_state": model.state_dict(),
        "stoi": tokenizer.stoi,
        "itos": tokenizer.itos,
        "model_config": {
            "vocab_size": tokenizer.vocab_size,
            "n_embd": config.n_embd,
            "n_head": config.n_head,
            "n_layer": config.n_layer,
            "block_size": config.block_size,
            "dropout": config.dropout,
        },
    }, config.checkpoint_path)
    print(f"Saved checkpoint to {config.checkpoint_path}")


if __name__ == "__main__":
    main()
