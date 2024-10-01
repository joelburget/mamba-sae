# Mamba Interpretability

This repo is for doing interpretability work on [Mamba (Linear-Time Sequence Modeling with Selective State Spaces
)](https://arxiv.org/abs/2312.00752). We follow the approach from Anthropic's [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html), though the scope might broaden in the future.

We use SAELens for training and evaluating SAEs.

## `train_sae.py`

Run a wandb sweep to determine hyperparameters.

```
> wandb sweep --project mamba-sae-sweep sweep_config.yaml
> wandb agent <sweep id printed by previous command>
```
