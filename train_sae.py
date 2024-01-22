import os
from collections import OrderedDict

import torch
from datasets import load_dataset
from nnsight import LanguageModel
from tqdm import tqdm

from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.training import trainSAE
from modeling_mamba import MambaConfig, MambaForCausalLM
from tokenizer import Tokenizer
from params import (
    d_model,
    dataset_path,
    sparsity_penalties,
    relative_sizes,
    sae_dir,
    sae_path,
    model_path,
    map_location,
)

mamba_config = MambaConfig(n_layer=1, d_model=d_model)


def make_automodel(state_dict_path: str) -> MambaForCausalLM:
    original_state_dict = torch.load(state_dict_path, map_location=map_location)
    renamed_state_dict = OrderedDict()
    for key in original_state_dict:
        new_key = key.replace("backbone", "model").replace(".mixer", "")
        renamed_state_dict[new_key] = original_state_dict[key]

    automodel = MambaForCausalLM(mamba_config)
    automodel.load_state_dict(renamed_state_dict)
    # automodel.cuda()
    return automodel


def make_model(state_dict_path: str) -> LanguageModel:
    """Make a LanguageModel from a state dict.

    1. Load state dict.
    2. Update it to use MambaForCausalLM names.
    3. Wrap in a LanguageModel.
    """
    return LanguageModel(make_automodel(state_dict_path), tokenizer=Tokenizer())


if __name__ == "__main__":
    dataset = load_dataset(dataset_path, split="train", streaming=True)
    model = make_model(model_path)

    for sparsity_penalty in tqdm(
        sparsity_penalties, desc="sparsity_penalty", position=0
    ):
        for relative_size in tqdm(
            relative_sizes, desc="relative_size", position=1, leave=False
        ):
            buffer = ActivationBuffer(
                data=(example["text"] for example in dataset.take(200_000)),
                model=model,
                submodule=model.model.layers[0],
                in_feats=d_model,
                out_feats=d_model,
            )

            ae = trainSAE(
                activations=buffer,
                activation_dim=d_model,
                dictionary_size=relative_size * d_model,
                lr=3e-4,
                sparsity_penalty=sparsity_penalty,
                device="cuda:0",
            )

            if not os.path.exists(sae_dir):
                os.makedirs(sae_dir)
            torch.save(ae.state_dict(), sae_path)
