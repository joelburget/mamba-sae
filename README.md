# Mamba Interpretability

This repo is for doing interpretability work on [Mamba (Linear-Time Sequence Modeling with Selective State Spaces
)](https://arxiv.org/abs/2312.00752). We follow the approach from Anthropic's [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html), though the scope might broaden in the future.

We make heavy use of the [nnsight](https://nnsight.net/) library for interpreting neural networks and the [dictionary_learning](https://github.com/saprmarks/dictionary_learning) library for training and understanding SAEs.

# Tools

## Main scripts

_Before running any of these scripts, set the appropriate parameters in `params.py`_.

### `train_model.py`

Script for training a one-layer Mamba model. Outputs stats to [wandb](https://wandb.ai/site/).

### `train_sae.py`

Once you've trained your model, you can train a Sparse Autoencoder on it. This script actually trains a grid of SAEs, one for each combination of sparsity penalty and relative size you configure.

### `evaluate_saes.py`

For each autoencode trained in the previous step, evaluate stats such as MSE loss, percentage of neurons alive, percentage of loss recovered, etc.

### `analyze_sae.py`

Given a single SAE, find top activations for each neuron. Hopefully more features in the future.

## Other

### `sae_analyze_320.py` / `sae_analyze_640.py`

Scripts to plot heatmaps of stats for two sets of SAEs I trained (comparing four different sparsity penalties and four different relative sizes).

### `run_model.py`

Helpers for running a model you've trained.

# Pretrained Models / SAEs

I've uploaded two models (`pytorch_model-{320,640}.bin`) and two sets of SAEs trained on them to [Google Drive](https://drive.google.com/drive/folders/1l8Qiei75lQjrz_EUrgNgysfZ-gkr_r0L?usp=sharing). [Doc with their stats](https://docs.google.com/document/d/1Y1iEJIkoXhLkdxEQCxIPHJFirJRq26R9gRqpYV9hie0/edit?usp=sharing).
