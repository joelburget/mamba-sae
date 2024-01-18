# The Mamba model's internal dimension
d_model = 320

# The size of the SAE dictionary relative to d_model. Used only in analyze_sae.py.
relative_size = 4

# Used only in analyze_sae.py.
sparsity_penalty = 0.004

# The size of the SAE dictionary. Used only in analyze_sae.py.
dictionary_size = relative_size * d_model

# Which dataset to use. This is passed to Huggingface's load_dataset() function
# so it can be the name of a HF dataset.
dataset_path = "/mnt/hddraid/pile-uncopyrighted"

# Train a separate SAE for each of these sparsity penalties.
sparsity_penalties = [0.004, 0.006, 0.008, 0.01]

# Train a separate SAE for each of these sizes.
relative_sizes = [2, 4, 8, 16, 32]

# Where trained SAEs are stored.
sae_dir = f"sae-output-{d_model}"

# The path to the SAE model. Used only in analyze_sae.py.
sae_path = f"{sae_dir}/model-{sparsity_penalty}-{relative_size}.bin"

# Where trained Mamba models are stored.
model_dir = f"model-output-{d_model}"
model_path = f"{model_dir}/pytorch_model.bin"
