import os

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import MambaForCausalLM

from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.lm_runner import language_model_sae_runner

from params import (
    d_model,
    dataset_path,
    sparsity_penalties,
    relative_sizes,
    sae_dir,
    sae_path,
    model_path,
)

if __name__ == "__main__":
    dataset = load_dataset(dataset_path, split="train", streaming=True)
    model = MambaForCausalLM.from_pretrained(model_path)
    if not os.path.exists(sae_dir):
        os.makedirs(sae_dir)

    for sparsity_penalty in tqdm(
        sparsity_penalties, desc="sparsity_penalty", position=0
    ):
        for relative_size in tqdm(
            relative_sizes, desc="relative_size", position=1, leave=False
        ):
            cfg = LanguageModelSAERunnerConfig(
                model_name=f"mamba-1l-{relative_size}-{sparsity_penalty}",
                hook_point="layers.0.hook_resid_pre",
                hook_point_layer=0,
                d_in=d_model,
                dataset_path="Skylion007/openwebtext",
                is_dataset_tokenized=False,
                # SAE Parameters
                expansion_factor=relative_size,
                b_dec_init_method="geometric_median",
                # Training Parameters
                l1_coefficient=sparsity_penalty,
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
