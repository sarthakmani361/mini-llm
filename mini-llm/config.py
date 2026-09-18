"""
config.py -- All hyperparameters in one place.

These defaults are deliberately modest so a full training run finishes
in a few minutes on a plain CPU. See the "Scaling up" section in the
README for a bigger config to use if you have a GPU (e.g. free Google
Colab).
"""

# --- Model architecture ---
n_embd = 128        # embedding dimension
n_head = 4           # attention heads (n_embd must be divisible by this)
n_layer = 4          # number of transformer blocks
block_size = 128     # max context length, in characters
dropout = 0.1

# --- Training ---
batch_size = 32
learning_rate = 3e-4
max_iters = 3000
eval_interval = 250
eval_iters = 50

# --- Data ---
data_path = "data/input.txt"
# Tiny Shakespeare (~1MB) -- a standard, tiny, freely available text corpus
# used for exactly this kind of from-scratch LLM demo. Swap in your own
# .txt file at `data_path` and delete the old one to train on something else.
data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# --- Misc ---
seed = 1337
checkpoint_path = "checkpoint.pt"
