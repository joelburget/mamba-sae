import argparse
from collections import OrderedDict

import torch
from datasets import load_dataset
from nnsight import LanguageModel

from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.training import trainSAE
from modeling_mamba import MambaConfig, MambaForCausalLM
from tokenizer import Tokenizer

activation_dim = 320
dictionary_size = 16 * activation_dim
mamba_config = MambaConfig(n_layer=1, d_model=activation_dim)


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
    automodel.cuda()
    tokenizer = Tokenizer()
    return LanguageModel(automodel, tokenizer=tokenizer)


def run(args):
    dataset = load_dataset(args.dataset, split="train", streaming=True)
    model = make_model(args.state_dict)

    buffer = ActivationBuffer(
        data=(example["text"] for example in dataset.take(200_000)),
        model=model,
        submodule=model.model.layers[0],
        in_feats=activation_dim,
        out_feats=activation_dim,
    )

    ae = trainSAE(
        buffer,
        activation_dim,
        dictionary_size,
        lr=3e-4,
        sparsity_penalty=4e-3,
        device="cuda:0",
    )

    torch.save(ae.state_dict(), args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        type=str,
                        default="/mnt/hddraid/pile-uncopyrighted")
    parser.add_argument("--state_dict",
                        type=str,
                        default="./output/pytorch_model.bin")
    parser.add_argument("--output", type=str, default="sae-output/model.bin")
