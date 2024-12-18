import torch
from datasets import load_dataset
import yaml
import wandb

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.sae_training_runner import SAETrainingRunner

model_name = "state-spaces/mamba-130m"
d_model = 768
dataset_path = "NeelNanda/openwebtext-tokenized-9b"
training_tokens = 300_000_000 // 10

expansion_factor = 16
hook_layer = 12

if __name__ == "__main__":
    dataset = load_dataset(dataset_path, split="train", streaming=True)

    with open("sweep_config.yaml", "r") as sweep_config_file:
        sweep_config = yaml.load(sweep_config_file, Loader=yaml.FullLoader)
        with wandb.init(config=sweep_config):
            sparsity_penalty = wandb.config.sparsity_penalty
            learning_rate = wandb.config.learning_rate
            dictionary_size = expansion_factor * d_model

            cfg = LanguageModelSAERunnerConfig(
                model_name=model_name,
                model_class_name="HookedMamba",
                hook_name=f"blocks.{hook_layer}.hook_resid_pre",
                hook_layer=hook_layer,
                d_in=d_model,
                dataset_path=dataset_path,
                is_dataset_tokenized=True,
                # is_dataset_tokenized=False,
                # streaming=True,
                # SAE Parameters
                expansion_factor=expansion_factor,
                b_dec_init_method="geometric_median",
                # Training Parameters
                l1_coefficient=sparsity_penalty,
                context_size=1024,
                lr=learning_rate,
                lr_warm_up_steps=5000,
                # Activation Store Parameters
                n_batches_in_buffer=128,
                training_tokens=training_tokens,
                store_batch_size_prompts=32,
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
                model_kwargs={
                    # "fast_ssm": True,
                    # Error:
                    # File "/usr/local/lib/python3.10/dist-packages/mamba_lens/HookedMamba.py", line 821, in forward
                    # import selective_scan_cuda
                    # ImportError: /workspace/mamba/selective_scan_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c104cuda14ExchangeDeviceEa
                    "fast_conv": True,
                },
                model_from_pretrained_kwargs={},
            )
            sae = SAETrainingRunner(cfg).run()

            sae_save_path = "sae.pth"
            torch.save(sae.state_dict(), sae_save_path)
            wandb.save(sae_save_path)
            artifact = wandb.Artifact(
                f"mamba-{expansion_factor}x-{sparsity_penalty}p", type="model"
            )
            artifact.add_file(sae_save_path)
            artifact.save()
            artifact.wait()
