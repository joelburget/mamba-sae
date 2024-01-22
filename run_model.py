import torch

from train_sae import make_automodel
from tokenizer import Tokenizer
from params import model_path

tokenizer = Tokenizer()
model = make_automodel(model_path)


def generate(input: str, max_length=50) -> str:
    tokens = tokenizer(input)
    input_ids = torch.tensor([tokens["input_ids"]])
    generation = model.generate(input_ids, max_length=max_length)
    return tokenizer.decode(generation[0])
