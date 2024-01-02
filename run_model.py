import torch
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from tokenizer import Tokenizer

tokenizer = Tokenizer()

model = MambaLMHeadModel(MambaConfig(n_layer=1, d_model=320))
model.load_state_dict(torch.load("./output/pytorch_model.bin"))
model.cuda()


def generate(input: str) -> str:
    tokens = tokenizer(input)
    input_ids = torch.tensor([tokens["input_ids"]]).cuda()
    generation = model.generate(input_ids, max_length=50)
    return tokenizer.decode(generation[0])
