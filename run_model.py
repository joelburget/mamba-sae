import torch
from transformers import MambaForCausalLM

from params import model_path

model = MambaForCausalLM.from_pretrained(model_path)
tokenizer = model.tokenizer


def generate(input: str, max_length=50) -> str:
    tokens = tokenizer(input)
    input_ids = torch.tensor([tokens["input_ids"]])
    generation = model.generate(input_ids, max_length=max_length)
    return tokenizer.decode(generation[0])
