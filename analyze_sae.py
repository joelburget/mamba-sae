"""
Find features in a single SAE.
"""
# For each feature, find:
# x find top activations
# -   - autointerp?
# - positive / negative logits (not sure how exactly this is done)
# - activation histogram

import argparse
import heapq
import pickle
from dataclasses import dataclass
from typing import List

import torch
from datasets import IterableDataset, load_dataset
from nnsight import LanguageModel
from tqdm import tqdm

from dictionary_learning.dictionary import AutoEncoder
from tokenizer import Tokenizer
from train_sae import make_model
from params import (
    d_model,
    dictionary_size,
    dataset_path,
    model_path,
    sae_path,
)

tokenizer = Tokenizer()
excerpt_width = 2


def activations_on_input(
    model: LanguageModel, ae: AutoEncoder, tokens: List[int]
) -> torch.Tensor:
    with model.invoke(tokens) as _invoker:
        embedding = model.model.embedding
        intervention_proxy = embedding.output[0].save()
    return ae.encode(intervention_proxy.value.cpu())


@dataclass
class TokenFocus:
    left_context: List[int]
    focal_token: int
    right_context: List[int]


@dataclass
class Activation:
    input: str
    token_focus: TokenFocus
    position: int
    strength: float


@dataclass
class AnalysisResult:
    max_activations: List[List[Activation]]
    # TODO: logits, histogram


def analyze_features(
    data: IterableDataset, model: LanguageModel, ae: AutoEncoder, n: int
) -> AnalysisResult:
    min_heaps = [[] for _ in range(dictionary_size)]

    for example in tqdm(data):
        # activations has dimensions [seq_len, dictionary_size]
        tokens = tokenizer(example)["input_ids"]
        activations = activations_on_input(model, ae, tokens)

        for feature_n in range(dictionary_size):
            min_heap = min_heaps[feature_n]
            for pos in range(activations.shape[0]):
                token_focus = TokenFocus(
                    tokens[max(0, pos - excerpt_width) : pos],
                    tokens[pos],
                    tokens[pos + 1 : pos + excerpt_width],
                )
                activation = activations[pos, feature_n].item()
                if len(min_heap) < n:
                    heapq.heappush(min_heap, (-activation, pos, example, token_focus))
                else:
                    heapq.heappushpop(
                        min_heap, (-activation, pos, example, token_focus)
                    )

    return AnalysisResult(
        [
            [
                Activation(example, token_focus, pos, -neg_activation)
                for neg_activation, pos, example, token_focus in min_heap
            ]
            for min_heap in min_heaps
        ]
    )


def print_top_acts(acts: List[Activation]):
    for activation in acts:
        match activation:
            case Activation(
                _input,
                TokenFocus(left_context, focal_token, right_context),
                _position,
                score,
            ):
                left_str = tokenizer.decode(left_context)
                focus = tokenizer.decode(focal_token)
                right_str = tokenizer.decode(right_context)
                print(
                    f"  \033[1m{score}\033[0m: {left_str}\033[1m{focus}\033[0m{right_str}"
                )


def run(args):
    ae_state_dict = torch.load(sae_path)
    ae = AutoEncoder(d_model, dictionary_size)
    ae.load_state_dict(ae_state_dict)
    model = make_model(model_path)
    dataset = load_dataset(dataset_path, split="train", streaming=True)
    data = (example["text"] for example in dataset.take(args.data_points))

    analysis_result = analyze_features(data, model, ae, 3).max_activations
    with open(args.pickle_location, "wb") as f:
        pickle.dump(analysis_result, f, pickle.HIGHEST_PROTOCOL)

    for i, activations in enumerate(analysis_result):
        print(f"feature {i}:")
        print_top_acts(activations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_location", type=str, default="analysis.pickle")
    parser.add_argument("--data_points", type=int, default=30_000)

    args = parser.parse_args()
    run(args)
