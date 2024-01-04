import argparse
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

d_model = 640
mamba_config = MambaConfig(n_layer=1, d_model=d_model)


def make_model(state_dict_path: str) -> LanguageModel:
    """Make a LanguageModel from a state dict.

    1. Load state dict.
    2. Update it to use MambaForCausalLM names.
    3. Wrap in a LanguageModel.
    """
    original_state_dict = torch.load(state_dict_path)
    renamed_state_dict = OrderedDict()
    for key in original_state_dict:
        new_key = key.replace("backbone", "model").replace(".mixer", "")
        renamed_state_dict[new_key] = original_state_dict[key]

    automodel = MambaForCausalLM(mamba_config)
    automodel.load_state_dict(renamed_state_dict)
    # automodel.cuda()
    tokenizer = Tokenizer()
    return LanguageModel(automodel, tokenizer=tokenizer)


def run(args):
    dataset = load_dataset(args.dataset, split="train", streaming=True)
    model = make_model(args.state_dict)

    for sparsity_penalty in tqdm([0.004, 0.006, 0.008, 0.01],
                                 desc="sparsity_penalty",
                                 position=0):
        for relative_size in tqdm([2, 4, 8, 16, 32],
                                  desc="relative_size",
                                  position=1,
                                  leave=False):
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

            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            torch.save(
                ae.state_dict(),
                f"{args.output_dir}/model-{sparsity_penalty}-{relative_size}.bin",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        type=str,
                        default="/mnt/hddraid/pile-uncopyrighted")
    parser.add_argument("--state_dict",
                        type=str,
                        default="./output-640/pytorch_model.bin")
    parser.add_argument("--output_dir", type=str, default="sae-output-640")

    args = parser.parse_args()
    run(args)
