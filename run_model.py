import torch
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from tokenizer import Tokenizer
from params import d_model, model_path

tokenizer = Tokenizer()

model = MambaLMHeadModel(MambaConfig(n_layer=1, d_model=d_model))
model.load_state_dict(torch.load(model_path))
model.cuda()


def generate(input: str, max_length=50) -> str:
    tokens = tokenizer(input)
    input_ids = torch.tensor([tokens["input_ids"]]).cuda()
    generation = model.generate(input_ids, max_length=max_length)
    return tokenizer.decode(generation[0])
