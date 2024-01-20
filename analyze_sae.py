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
from rich.progress import Progress, BarColumn, TimeRemainingColumn, MofNCompleteColumn

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

    def __lt__(self, other):
        return self.focal_token < other.focal_token


@dataclass
class Activation:
    token_focus: TokenFocus
    strength: float


MaxActivationsForNeuron = List[Activation]
WorkerActivationsForNeuron = List[Tuple[float, TokenFocus]]
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

    while True:
        example = data_queue.get()
        if example is None:
            break  # End of data

        result = []
        for _ in range(dictionary_size):
            result.append([])

        tokens = tokenizer(example)["input_ids"]
        activations = activations_on_input(model, ae, tokens)
        seq_len = len(tokens)

        for feature_n in range(dictionary_size):
            for pos in range(seq_len):
                token_focus = TokenFocus(
                    tokens[max(0, pos - excerpt_width) : pos],
                    tokens[pos],
                    tokens[pos + 1 : pos + excerpt_width],
                )
                activation = activations[pos, feature_n].item()
                result[feature_n].append((activation, token_focus))

        result_queue.put(result)


def make_analysis_result(
    min_heaps: List[WorkerActivationsForNeuron],
    all_activations: List[AllActivationsForNeuron],
) -> AnalysisResult:
    return AnalysisResult(
        [
            [
                Activation(token_focus, -neg_activation)
                for neg_activation, token_focus in min_heap
            ]
            for min_heap in min_heaps
        ],
        all_activations,
    )


def collator(pickle_location: str, result_queue: mp.Queue, data_len: int):
    min_heaps: List[WorkerActivationsForNeuron] = []
    all_activations: List[AllActivationsForNeuron] = []
    for _ in range(dictionary_size):
        min_heaps.append([])
        all_activations.append([])

    def save(analysis_result: AnalysisResult, n=None):
        save_location = pickle_location
        if n is not None:
            save_location += f".{n}"

        with open(save_location, "wb") as f:
            pickle.dump(analysis_result, f, pickle.HIGHEST_PROTOCOL)

    example_count = 0
    with Progress(
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        progress_bar = progress.add_task("Progress...", total=data_len)
        while True:
            result = result_queue.get()
            if result is None:
                break

            for feature_n, activations in enumerate(result):
                for (activation, token_focus) in activations:
                    all_activations[feature_n].append(activation)
                    min_heap = min_heaps[feature_n]
                    if activation < SIGNIFICANT_ACTIVATION_THRESHOLD:
                        continue
                    if len(min_heap) < top_feature_count:
                        heapq.heappush(min_heap, (-activation, token_focus))
                    else:
                        heapq.heappushpop(min_heap, (-activation, token_focus))

            example_count += 1
            progress.update(progress_bar, advance=1)

            if example_count % 1000 == 0:
                print(f"Processed {example_count} examples, saving checkpoint")
                save(
                    make_analysis_result(min_heaps, all_activations),
                    n=example_count // 1000,
                )

    analysis_result = make_analysis_result(min_heaps, all_activations)
    save(analysis_result)
    print_analysis(analysis_result)


def analyze_features_parallel(
    data: IterableDataset, data_len: int, num_workers: int, pickle_location: str
):
    data_queue = mp.Queue(maxsize=num_workers)
    result_queue = mp.Queue()

    collator_process = mp.Process(
        target=collator, args=(pickle_location, result_queue, data_len)
    )
    collator_process.start()

    worker_processes = []
    for _ in range(num_workers):
        p = mp.Process(
            target=analyze_feature_worker,
            args=(data_queue, result_queue),
        )
        p.start()
        worker_processes.append(p)

    # Distribute data among workers
    for example in data:
        data_queue.put(example[:20])
    for _ in range(len(worker_processes)):
        data_queue.put(None)  # Send termination signal

    for p in worker_processes:
        p.join(10)

    result_queue.put(None)
    collator_process.join(10)


def print_top_acts(acts: List[Activation]):
    for activation in acts:
        match activation:
            case Activation(
                TokenFocus(left_context, focal_token, right_context),
                score,
            ):
                left_str = tokenizer.decode(left_context)
                focus = tokenizer.decode(focal_token)
                right_str = tokenizer.decode(right_context)
                print(
                    f"  \033[1m{score}\033[0m: {left_str}\033[1m{focus}\033[0m{right_str}"
                )


def print_analysis(analysis_result: AnalysisResult):
    for i, activations in enumerate(analysis_result.max_activations):
        if len(activations):
            print(f"feature {i}:")
            print_top_acts(activations)
        else:
            print(f"feature {i}: no significant activations")


def run(args):
    dataset = load_dataset(dataset_path, split="train", streaming=True)
    data = (example["text"] for example in dataset.take(args.data_points))
    analyze_features_parallel(data, args.data_points, 4, args.pickle_location)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_location", type=str, default="analysis.pickle")
    parser.add_argument("--data_points", type=int, default=100)

    args = parser.parse_args()
    run(args)
