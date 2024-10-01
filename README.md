# Mamba Interpretability

This repo is for doing interpretability work on [Mamba (Linear-Time Sequence Modeling with Selective State Spaces
)](https://arxiv.org/abs/2312.00752). We follow the approach from Anthropic's [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html), though the scope might broaden in the future.

We use SAELens for training and evaluating SAEs.

# Tools

## Main scripts

_Before running the script, set the appropriate parameters in `params.py`_.

### `train_sae.py`

Once you've trained your model, you can train a Sparse Autoencoder on it. This script actually trains a grid of SAEs, one for each combination of sparsity penalty and relative size you configure.
