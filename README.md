# Mamba Interpretability

This repo is for doing interpretability work on [Mamba (Linear-Time Sequence Modeling with Selective State Spaces
)](https://arxiv.org/abs/2312.00752). We follow the approach from Anthropic's [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html), though the scope might broaden in the future.

We make heavy use of the [nnsight](https://nnsight.net/) library for interpreting neural networks and the [dictionary_learning](https://github.com/saprmarks/dictionary_learning) library for training and understanding SAEs.

# Tools

## `train_model.py`

Script for training a one-layer Mamba model. Outputs stats to [wandb](https://wandb.ai/site/).

## `train_sae.py`

Once you've trained your model, you can train a Sparse Autoencoder on it. This script actually trains a grid of SAEs, one for each combination of sparsity penalty and relative size you configure.

## `evaluate_saes.py`

For each autoencode trained in the previous step, evaluate stats such as MSE loss, percentage of neurons alive, percentage of loss recovered, etc.

## `analyze_sae.py`

Given a single SAE, find top activations for each neuron. Hopefully more features in the future.
