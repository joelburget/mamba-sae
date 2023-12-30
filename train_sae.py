from collections import OrderedDict

import torch
from datasets import load_dataset
from nnsight import LanguageModel

from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.training import trainSAE
from modeling_mamba import MambaConfig, MambaForCausalLM
from tokenizer import Tokenizer

dataset = load_dataset("/mnt/hddraid/pile-uncopyrighted",
                       split='train',
                       streaming=True)

print("dataset loaded")

mamba_config = MambaConfig(n_layer=1, d_model=320)
activation_dim = 320
dictionary_size = 16 * activation_dim

tokenizer = Tokenizer()

original_state_dict = torch.load("./output/pytorch_model.bin")
renamed_state_dict = OrderedDict()
for key in original_state_dict:
    new_key = key.replace("backbone", "model").replace(".mixer", "")
    renamed_state_dict[new_key] = original_state_dict[key]

automodel = MambaForCausalLM(mamba_config)
automodel.load_state_dict(renamed_state_dict)
automodel.cuda()
model = LanguageModel(automodel, tokenizer=tokenizer)

submodule = model.model.layers[0]
data = (example["text"] for example in dataset.take(100_000))
short_data = (example["text"] for example in dataset.take(1000))

buffer = ActivationBuffer(
    data,
    model,
    submodule,
    in_feats=activation_dim,
    out_feats=activation_dim,
)

if __name__ == "__main__":
    print("training SAE")
    ae = trainSAE(
        buffer,
        activation_dim,
        dictionary_size,
        lr=3e-4,
        sparsity_penalty=1e-3,
        device="cuda:0",
    )

    print("done training SAE")
    torch.save(ae.state_dict(), "sae-output/model.bin")
