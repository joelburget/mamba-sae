# For each feature, find:
# x find top activations
# -   - autointerp?
# - positive / negative logits
# - activation histogram

import heapq
from dataclasses import dataclass
from typing import List, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm

from dictionary_learning.dictionary import AutoEncoder
from tokenizer import Tokenizer
from train_sae import activation_dim, dictionary_size, make_model

ae_state_dict = torch.load("sae-output/model.bin")
ae = AutoEncoder(activation_dim, dictionary_size)
ae.load_state_dict(ae_state_dict)
dataset = load_dataset("/mnt/hddraid/pile-uncopyrighted",
                       split="train",
                       streaming=True)
short_data = (example["text"] for example in dataset.take(1000))
tokenizer = Tokenizer()
model = make_model("./output/pytorch_model.bin")


# TODO: do a whole batch at a time
def activations_on_input(input: str) -> torch.Tensor:
    tokens = tokenizer(input)["input_ids"]
    with model.invoke(tokens) as _invoker:
        embedding = model.model.embedding
        intervention_proxy = embedding.output[0].save()
    return ae.encode(intervention_proxy.value.cpu())[0]


@dataclass
class AnalysisResult:
    max_activations: List[List[Tuple[float, str]]]
    # TODO: logits, histogram


def analyze_features(n: int) -> AnalysisResult:
    min_heaps = [[] for _ in range(dictionary_size)]

    for example in tqdm(short_data):
        activations = activations_on_input(example)

        for feature_n in range(dictionary_size):
            min_heap = min_heaps[feature_n]
            activation = activations[feature_n].item()
            if len(min_heap) < n:
                heapq.heappush(min_heap, (-activation, example))
            else:
                heapq.heappushpop(min_heap, (-activation, example))

    return AnalysisResult([[(-neg_activation, example)
                            for neg_activation, example in min_heap]
                           for min_heap in min_heaps])


def print_top_acts(acts: List[Tuple[float, str]], excerpt_length=200):
    for score, text in acts:
        print(f"\033[1m{score}\033[0m", text[:excerpt_length])


if __name__ == "__main__":
    print(analyze_features(6))
