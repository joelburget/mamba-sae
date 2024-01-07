"""
Output statistics for each SAE.
"""

import torch
from datasets import load_dataset
from tqdm import tqdm

from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.evaluation import evaluate
from tokenizer import Tokenizer
from train_sae import make_model
from params import (
    d_model,
    sparsity_penalties,
    relative_sizes,
    dataset_path,
    model_path,
    sae_path,
)


if __name__ == "__main__":
    tokenizer = Tokenizer()
    model = make_model(model_path)
    submodule = model.model.layers[0]

    for sparsity_penalty in tqdm(
        sparsity_penalties, desc="sparsity_penalty", position=0
    ):
        for relative_size in tqdm(
            relative_sizes, desc="relative_size", position=1, leave=False
        ):
            ae_state_dict = torch.load(sae_path)
            ae = AutoEncoder(d_model, d_model * relative_size)
            ae.load_state_dict(ae_state_dict)

            dataset = load_dataset(dataset_path, split="train", streaming=True)
            data = (
                example["text"]
                for example in dataset.filter(lambda example: len(example) < 5000).take(
                    500 * 128
                )
            )

            buffer = ActivationBuffer(
                data=data,
                model=model,
                submodule=submodule,
                in_feats=d_model,
                out_feats=d_model,
                n_ctxs=500,
                # ctx_len=32,
            )

            result = evaluate(model, submodule, ae, buffer)
            print(f"sparsity {sparsity_penalty}, relative size: {relative_size}")
            print(result)
