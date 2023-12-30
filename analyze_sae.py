# For each feature, find:
# x find top activations
# -   - autointerp?
# - positive / negative logits
# - activation histogram

import heapq
from dataclasses import dataclass
from typing import List, Tuple

import torch
from tqdm import tqdm

from dictionary_learning.dictionary import AutoEncoder
from train_sae import (activation_dim, dictionary_size, model, short_data,
                       tokenizer)

state_dict = torch.load("sae-output/model.bin")
ae = AutoEncoder(activation_dim, dictionary_size)
ae.load_state_dict(state_dict)


def activation_on_input(feature_no: int, input: str) -> float:
    tokens = tokenizer(input)['input_ids']
    with model.invoke(tokens) as _invoker:
        embedding = model.model.embedding
        intervention_proxy = embedding.output[0].save()
    return ae.encode(intervention_proxy.value.cpu())[0, feature_no].item()


@dataclass
class AnalysisResult:
    max_activations: List[Tuple[float, str]]
    # TODO: logits, histogram


def analyze_feature(feature_no: int, n: int) -> AnalysisResult:
    min_heap = []

    for example in tqdm(short_data):
        activation = activation_on_input(feature_no, example)

        if len(min_heap) < n:
            heapq.heappush(min_heap, (-activation, input))
        else:
            heapq.heappushpop(min_heap, (-activation, input))

    return AnalysisResult([(-neg_activation, input)
                           for neg_activation, input in min_heap])
