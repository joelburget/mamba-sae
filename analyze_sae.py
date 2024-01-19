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
from typing import List, Tuple, TypeVar

import torch
import torch.multiprocessing as mp
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
    map_location,
)

tokenizer = Tokenizer()
excerpt_width = 2
top_feature_count = 4
SIGNIFICANT_ACTIVATION_THRESHOLD = 0.01

T = TypeVar("T")


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


MaxActivationsForNeuron = List[Activation]
WorkerActivationsForNeuron = List[Tuple[float, int, str, TokenFocus]]
AllActivationsForNeuron = List[float]


@dataclass
class AnalysisResult:
    max_activations: List[MaxActivationsForNeuron]
    all_activations: List[AllActivationsForNeuron]  # For histogram
    # TODO: logits


def analyze_feature_worker(data_queue, result_queue):
    ae_state_dict = torch.load(sae_path, map_location=map_location)
    ae = AutoEncoder(d_model, dictionary_size)
    ae.load_state_dict(ae_state_dict)
    model = make_model(model_path)

    min_heaps = [[] for _ in range(dictionary_size)]
    all_activations = [[] for _ in range(dictionary_size)]
    while True:
        example = data_queue.get()
        if example is None:
            break  # End of data

        tokens = tokenizer(example)["input_ids"]
        activations = activations_on_input(model, ae, tokens)
        seq_len = len(tokens)

        for feature_n in range(dictionary_size):
            min_heap = min_heaps[feature_n]
            activations_for_feature = all_activations[feature_n]
            for pos in range(seq_len):
                token_focus = TokenFocus(
                    tokens[max(0, pos - excerpt_width) : pos],
                    tokens[pos],
                    tokens[pos + 1 : pos + excerpt_width],
                )
                activation = activations[pos, feature_n].item()
                activations_for_feature.append(activation)
                if activation < SIGNIFICANT_ACTIVATION_THRESHOLD:
                    continue
                if len(min_heap) < top_feature_count:
                    heapq.heappush(min_heap, (-activation, pos, example, token_focus))
                else:
                    heapq.heappushpop(
                        min_heap, (-activation, pos, example, token_focus)
                    )

    result_queue.put((min_heaps, all_activations))


def chain(lists: Tuple[List[T]]) -> List[T]:
    result = []
    for l in lists:
        result.extend(l)
    return result


def analyze_features_parallel(
    data: IterableDataset,
    num_workers: int,
) -> AnalysisResult:
    data_queue = mp.Queue(maxsize=num_workers)
    result_queue = mp.Queue()

    processes = []
    for _ in range(num_workers):
        p = mp.Process(
            target=analyze_feature_worker,
            args=(data_queue, result_queue),
        )
        p.start()
        processes.append(p)

    # Distribute data among workers
    for example in tqdm(data):
        data_queue.put(example)
    for _ in range(len(processes)):
        data_queue.put(None)  # Send termination signal

    min_heaps: List[WorkerActivationsForNeuron] = [[]] * dictionary_size
    all_activations: List[AllActivationsForNeuron] = [[]] * dictionary_size

    for _ in processes:
        min_heaps_, all_activations_ = result_queue.get()
        for i, min_heap in enumerate(min_heaps_):
            min_heaps[i] = heapq.nlargest(
                top_feature_count, heapq.merge(min_heaps[i], min_heap)
            )
        for i, activations in enumerate(all_activations_):
            all_activations[i].extend(activations)

    for p in processes:
        p.join()

    return AnalysisResult(
        [
            [
                Activation(example, token_focus, pos, -neg_activation)
                for neg_activation, pos, example, token_focus in min_heap
            ]
            for min_heap in min_heaps
        ],
        all_activations,
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
    dataset = load_dataset(dataset_path, split="train", streaming=True)
    data = (example["text"] for example in dataset.take(args.data_points))

    analysis_result = analyze_features_parallel(data, 4)

    with open(args.pickle_location, "wb") as f:
        pickle.dump(analysis_result, f, pickle.HIGHEST_PROTOCOL)

    for i, activations in enumerate(analysis_result.max_activations):
        if len(activations):
            print(f"feature {i}:")
            print_top_acts(activations)
        else:
            print(f"feature {i}: no significant activations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_location", type=str, default="analysis.pickle")
    parser.add_argument("--data_points", type=int, default=100)

    args = parser.parse_args()
    run(args)
