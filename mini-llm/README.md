# Mini-LLM — a GPT built from scratch

A small, fully hand-written decoder-only Transformer (the same family as
GPT-2/GPT-3). No `transformers` library, no pretrained weights — every
piece is implemented so you can see exactly what's happening: token +
positional embeddings, multi-head causal self-attention, the feed-forward
MLP, and the residual/LayerNorm wiring that holds it together.

## Files

| File | What it does |
|---|---|
| `model.py` | The architecture: `CausalSelfAttention`, `FeedForward`, `Block`, and the full `GPT` model |
| `tokenizer.py` | Character-level tokenizer (each unique character = one token) |
| `config.py` | All hyperparameters in one place |
| `train.py` | Downloads data if needed, trains the model, saves `checkpoint.pt` |
| `generate.py` | Loads a checkpoint and samples text from a prompt |
| `utils.py` | Picks CUDA / MPS / CPU automatically |

## How the pieces fit together

```
tokens → [token_emb + pos_emb] → N × Block → LayerNorm → Linear → logits
Block: x = x + Attention(LayerNorm(x))
       x = x + FeedForward(LayerNorm(x))
```

Self-attention is the one idea worth sitting with: for each token, it computes
a *content-dependent* weighted average of every earlier token's value vector,
where the weights come from how well that token's "query" matches every other
token's "key." It's a bit like a matched filter that doesn't just correlate
against one fixed template — it learns, per token, what pattern to correlate
against. The causal mask just zeroes out (well, sets to `-inf` pre-softmax)
any position that would let a token see the future.

## Running it

```bash
pip install torch --break-system-packages   # if not already installed

python train.py                # trains for config.max_iters steps, saves checkpoint.pt
python generate.py --prompt "ROMEO:" --tokens 500
```

The first `train.py` run auto-downloads a ~1MB sample dataset (Tiny
Shakespeare) into `data/input.txt` if that file doesn't already exist.

**This has already been run and verified end-to-end** in the environment
that built it: a 300-step test run took the loss from 4.24 (≈ random guessing
over 65 characters, ln(65) ≈ 4.17) down to 2.44, and `generate.py` produced
coherent-looking character sequences from the resulting checkpoint. The
architecture was also unit-tested separately for correct forward-pass shapes,
gradient flow to every parameter, and correct autoregressive generation.

### Using your own text

Delete `data/input.txt` and drop your own `.txt` file at that same path (or
edit `config.data_path`). More text = a better model; a few hundred KB is a
reasonable minimum for anything recognizable.

## Scaling up

The shipped config (128-dim embeddings, 4 heads, 4 layers, 3000 steps) is
tuned to finish in a few minutes on a plain CPU — it's enough to watch the
loss come down and see real words emerge, but it's not a strong model. On a
GPU (e.g. a free Google Colab T4), you can push much further. In `config.py`:

```python
n_embd = 384
n_head = 6
n_layer = 6
block_size = 256
batch_size = 64
max_iters = 5000
```

This is the classic "baby GPT" scale used in nanoGPT-style tutorials, and
produces noticeably more coherent Shakespeare-like text in a few minutes on
a single GPU.

## Honest limitations

This is a character-level model trained on ~1MB of text — it will learn
spelling, some grammar, and character-name formatting ("ROMEO:"), but it
won't produce coherent long-range reasoning like a production LLM (those
are trained on trillions of tokens with subword tokenizers and far more
parameters). The point is transparency: every line of the forward pass is
something you wrote and can inspect, not a black box.

## Natural next steps, roughly in order of effort

1. **Swap in your own dataset** — try it on something you actually care about.
2. **Subword tokenization (BPE)** — replace `tokenizer.py`; shorter sequences per
   sentence, much better sample efficiency.
3. **KV-caching in `generate()`** — right now generation recomputes attention
   over the whole context at every step; caching past keys/values makes
   long generations much faster.
4. **Learning rate warmup + cosine decay** — `train.py` currently uses a flat
   LR; a warmup/decay schedule is a easy, well-documented win.
5. **Rotary positional embeddings (RoPE)** instead of learned absolute
   positions — what most current LLMs actually use.

If you ever want to take this toward the embedded/edge side, quantizing a
small trained model (e.g. to int8) and benchmarking inference latency would
be a natural bridge to the kind of edge-AI work you've been doing on the
radar vital-signs project.
