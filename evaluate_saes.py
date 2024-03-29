"""
Output statistics for each SAE.
"""

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import MambaForCausalLM

from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.evaluation import evaluate
from params import (
    d_model,
    sparsity_penalties,
    relative_sizes,
    dataset_path,
    model_path,
    sae_dir,
    map_location,
)


if __name__ == "__main__":
    model = MambaForCausalLM.from_pretrained(model_path)
    tokenizer = model.tokenizer
    submodule = model.model.layers[0]
    ctx_len = 128

    for sparsity_penalty in tqdm(
        sparsity_penalties, desc="sparsity_penalty", position=0
    ):
        for relative_size in tqdm(
            relative_sizes, desc="relative_size", position=1, leave=False
        ):
            sae_path = f"{sae_dir}/model-{sparsity_penalty}-{relative_size}.bin"
            ae_state_dict = torch.load(sae_path, map_location=map_location)
            ae = AutoEncoder(d_model, d_model * relative_size)
            ae.load_state_dict(ae_state_dict)

            dataset = load_dataset(dataset_path, split="train", streaming=True)
            data = (example["text"] for example in dataset)

            buffer = ActivationBuffer(
                data=data,
                model=model,
                submodule=submodule,
                in_feats=d_model,
                out_feats=d_model,
                ctx_len=ctx_len,
                in_batch_size=64,
            )

            result = evaluate(model, submodule, ae, buffer)
            print(f"sparsity {sparsity_penalty}, relative size: {relative_size}")
            print(result)
