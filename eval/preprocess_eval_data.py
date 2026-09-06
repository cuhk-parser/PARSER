#!/usr/bin/env python3
"""Preprocess eval JSON files into chunked dataset for eval.py."""

import argparse
import json
import os
import pickle
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm
from transformers import AutoTokenizer

_tokenizer = None
_chunk_max_length = 0


def get_tokenizer_identifier(tokenizer: Any) -> str:
    tokenizer_name_or_path = str(
        getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
    )
    return Path(tokenizer_name_or_path).name or tokenizer_name_or_path


def process_sample(
    sample: dict[str, Any],
    source_filename: str,
    tokenizer,
    chunk_max_length: int,
) -> dict[str, Any]:
    documents = sample["context"]
    splits = [
        f"Paragraph {i}: {part.strip()}"
        for i, part in enumerate(re.split(r"(?:^|\n)Document\s\d+:\s*", documents))
        if part.strip()
    ]
    split_tokens = [tokenizer.encode(part) for part in splits]

    merged_chunks: list[str] = []
    if not splits:
        merged_chunks = []
    elif chunk_max_length <= 0:
        merged_chunks = splits
    else:
        cur_chunk = splits[0]
        cur_chunk_token_count = len(split_tokens[0])
        for split, tokens in zip(splits[1:], split_tokens[1:]):
            if cur_chunk_token_count + len(tokens) < chunk_max_length:
                cur_chunk = f"{cur_chunk}\n\n{split}"
                cur_chunk_token_count += len(tokens) + 1
            else:
                merged_chunks.append(cur_chunk)
                cur_chunk = split
                cur_chunk_token_count = len(tokens) + 1
        merged_chunks.append(cur_chunk)

    return {
        "question": sample["input"],
        "chunks": merged_chunks,
        "ground_truth": sample["answers"],
        "domain": source_filename.replace(".json", ""),
    }


def _init_worker(tokenizer_path: str, chunk_max_length: int) -> None:
    global _tokenizer, _chunk_max_length
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    _chunk_max_length = chunk_max_length


def _process_entry(entry_tuple: tuple[dict[str, Any], str]) -> dict[str, Any]:
    entry, source_filename = entry_tuple
    return process_sample(entry, source_filename, _tokenizer, _chunk_max_length)


def preprocess_entries(
    entries: list[tuple[dict[str, Any], str]],
    tokenizer_path: str,
    chunk_max_length: int,
    num_workers: int,
) -> list[dict[str, Any]]:
    if num_workers <= 1:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        return [
            process_sample(entry, source_filename, tokenizer, chunk_max_length)
            for entry, source_filename in tqdm(entries, desc="Preprocessing eval samples")
        ]

    chunksize = max(1, len(entries) // (num_workers * 4))
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(tokenizer_path, chunk_max_length),
    ) as executor:
        return list(
            tqdm(
                executor.map(_process_entry, entries, chunksize=chunksize),
                total=len(entries),
                desc="Preprocessing eval samples",
            )
        )


def load_raw_entries(test_folder: str) -> list[tuple[dict[str, Any], str]]:
    json_files = ["eval_50.json", "eval_100.json", "eval_200.json", "eval_400.json", 
             "eval_800.json",  "eval_1600.json", "eval_3200.json",  "eval_6400.json"]
    # json_files = ["eval_2wikimultihopqa_6400_ordered.json", "eval_2wikimultihopqa_6400_reverse_ordered.json"] + ["eval_2wikimultihopqa_distance_middle_0_200.json", "eval_2wikimultihopqa_distance_middle_800_1000.json", "eval_2wikimultihopqa_distance_middle_1600_1800.json", "eval_2wikimultihopqa_distance_middle_2400_2600.json", "eval_2wikimultihopqa_distance_middle_3200_3400.json"]
    # json_files = ["eval_2wikimultihopqa_distance_middle_0_200.json", "eval_2wikimultihopqa_distance_middle_800_1000.json", "eval_2wikimultihopqa_distance_middle_1600_1800.json", "eval_2wikimultihopqa_distance_middle_2400_2600.json", "eval_2wikimultihopqa_distance_middle_3200_3400.json"]
    json_files = [f"{test_folder}/{item}" for item in json_files]
    all_entries: list[tuple[dict[str, Any], str]] = []
    for json_file in json_files:
        json_file = Path(json_file)
        with json_file.open("r", encoding="utf-8") as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            raise ValueError(f"Unexpected JSON format in {json_file.name}, expected list")
        all_entries.extend((entry, json_file.name) for entry in entries)
        # import pdb; pdb.set_trace()
    return all_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess eval dataset chunks")
    parser.add_argument(
        "--test-folder",
        default="../data/hqa_val",
        help="Folder containing raw eval JSON files",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="../Qwen3.5-4B",
        help="Tokenizer path or model path for AutoTokenizer",
    )
    parser.add_argument("--chunk-max-length", type=int, default=512)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel workers for tokenization (1 = sequential)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    tokenizer_identifier = get_tokenizer_identifier(tokenizer)

    entries = load_raw_entries(args.test_folder)
    dataset = preprocess_entries(
        entries,
        args.tokenizer_path,
        args.chunk_max_length,
        args.num_workers,
    )

    output_path = Path(args.test_folder) / (
        f"eval_preprocessed_{tokenizer_identifier}_{args.chunk_max_length}.pkl"
    )
    with output_path.open("wb") as f:
        pickle.dump(dataset, f)

    print(f"Saved {len(dataset)} samples to {output_path}")


if __name__ == "__main__":
    main()
