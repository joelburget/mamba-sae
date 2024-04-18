import os
import torch

from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.lm_runner import language_model_sae_runner

from params import (d_model, dataset_path, sae_path, relative_sizes, sparsity_penalties)

if __name__ == "__main__":
    cfg = LanguageModelSAERunnerConfig(
        model_name="state-spaces/mamba-130m-hf",
        hook_point="layers.2.hook_resid_pre",
        hook_point_layer=2,
        d_in=d_model,
        dataset_path="Skylion007/openwebtext",
        is_dataset_tokenized=False,
        # SAE Parameters
        expansion_factor=relative_sizes,
        b_dec_init_method="geometric_median",
        # Training Parameters
        l1_coefficient=sparsity_penalties,
        context_size=128,
        lr_warm_up_steps=5000,
        # Activation Store Parameters
        n_batches_in_buffer=128,
        total_training_tokens=1_000_000 * 300,
        store_batch_size=32,
        # Dead Neurons and Sparsity
        use_ghost_grads=True,
        feature_sampling_window=1000,
        dead_feature_window=5000,
        dead_feature_threshold=1e-6,
        # WANDB
        log_to_wandb=True,
        wandb_project="mamba-sae",
        wandb_log_frequency=100,
        # Misc
        device="cuda",
        n_checkpoints=10,
    )
    sae = language_model_sae_runner(cfg)
    torch.save(sae.state_dict(), sae_path)
