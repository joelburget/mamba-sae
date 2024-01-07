"""
Output statistics for each SAE.
"""

import argparse

import torch
from datasets import load_dataset
from tqdm import tqdm

from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.evaluation import evaluate
from tokenizer import Tokenizer
from train_sae import make_model

tokenizer = Tokenizer()
excerpt_width = 2
d_model = 640


def run(args):
    model = make_model(args.model_state_dict)
    submodule = model.model.layers[0]

    for sparsity_penalty in tqdm([0.004, 0.006, 0.008, 0.01],
                                 desc="sparsity_penalty",
                                 position=0):
        for relative_size in tqdm([2, 4, 8, 16, 32],
                                  desc="relative_size",
                                  position=1,
                                  leave=False):
            ae_state_dict = torch.load(
                f"sae-output-640/model-{sparsity_penalty}-{relative_size}.bin",
            )
            ae = AutoEncoder(d_model, d_model * relative_size)
            ae.load_state_dict(ae_state_dict)

            dataset = load_dataset(args.dataset, split="train", streaming=True)
            data = (example["text"] for example in dataset.filter(
                lambda example: len(example) < 5000).take(500 * 128))

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
            print(
                f"sparsity {sparsity_penalty}, relative size: {relative_size}")
            print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        type=str,
                        default="/mnt/hddraid/pile-uncopyrighted")
    parser.add_argument("--model_state_dict",
                        type=str,
                        default="output-640/pytorch_model.bin")
    parser.add_argument("--data_points", type=int, default=30_000)

    args = parser.parse_args()
    run(args)
