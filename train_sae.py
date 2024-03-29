import os

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import MambaForCausalLM

from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.training import trainSAE
from dictionary_learning.evaluation import evaluate

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
    lr = 3e-4
    if not os.path.exists(sae_dir):
        os.makedirs(sae_dir)

    for sparsity_penalty in tqdm(
        sparsity_penalties, desc="sparsity_penalty", position=0
    ):
        for relative_size in tqdm(
            relative_sizes, desc="relative_size", position=1, leave=False
        ):
            dictionary_size = relative_size * d_model
            submodule = model.model.layers[0]

            buffer = ActivationBuffer(
                data=(example["text"] for example in dataset.take(200_000)),
                model=model,
                submodule=submodule,
                in_feats=d_model,
                out_feats=d_model,
            )

            ae = trainSAE(
                activations=buffer,
                activation_dim=d_model,
                dictionary_size=dictionary_size,
                lr=lr,
                sparsity_penalty=sparsity_penalty,
                device="cuda:0",
            )

            print(evaluate(model, submodule, ae, buffer))

            torch.save(ae.state_dict(), sae_path)
