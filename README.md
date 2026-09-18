# Mini-LLM — GPT Built from Scratch

A lightweight GPT-style Large Language Model implemented entirely from scratch using **Python and PyTorch**.

This project builds the core components of a decoder-only Transformer without relying on Hugging Face Transformers, pretrained weights, or external LLM frameworks. It implements the complete pipeline from character-level tokenization and embeddings to causal self-attention, Transformer blocks, training, checkpointing, and autoregressive text generation.

The model is trained on the **Tiny Shakespeare** dataset and can generate Shakespeare-style text from a user-provided prompt.

### Key Features

* 🧠 Decoder-only GPT-style Transformer architecture
* 🔤 Character-level tokenizer implemented from scratch
* 🎯 Multi-head causal self-attention
* 🔄 Residual connections and Layer Normalization
* ⚡ Feed-forward MLP with GELU activation
* 📍 Learned token and positional embeddings
* 🎲 Temperature and Top-K sampling for text generation
* 💾 Model checkpoint saving and loading
* 🖥️ Automatic CUDA / Apple MPS / CPU device selection
* 📊 Training and validation loss monitoring
* 🔧 Configurable model architecture and training parameters

### Architecture

```text
Input Text
    ↓
Character Tokenizer
    ↓
Token + Positional Embeddings
    ↓
Transformer Blocks
    ├── LayerNorm
    ├── Causal Multi-Head Self-Attention
    ├── Residual Connection
    ├── LayerNorm
    ├── Feed-Forward Network
    └── Residual Connection
    ↓
Final LayerNorm
    ↓
Linear Projection
    ↓
Next-Token Logits
    ↓
Autoregressive Text Generation
```

The goal of this project is not to reproduce the scale or capabilities of production LLMs, but to make the underlying mechanics of a GPT-style language model transparent and understandable by implementing the major components directly.
