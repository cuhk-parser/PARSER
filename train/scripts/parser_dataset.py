"""VERL dataset adapter for Parser."""

from __future__ import annotations

import copy
import logging
import re
from pathlib import Path
from typing import Any

import datasets
import numpy as np
from omegaconf import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

try:
    from .prompt import LEAD_AGENT_SYSTEM_PROMPT, LEAD_AGENT_USER_PROMPT
except ImportError:
    # Loaded via VERL load_module(file path): no package parent for relative imports.
    import sys

    _scripts_dir = Path(__file__).resolve().parent
    if str(_scripts_dir) not in sys.path:
        sys.path.insert(0, str(_scripts_dir))
    from prompt import LEAD_AGENT_SYSTEM_PROMPT, LEAD_AGENT_USER_PROMPT

from verl.utils.fs import copy_to_local

logger = logging.getLogger(__name__)


def _as_list(data_files: str | list[str] | ListConfig) -> list[str]:
    if isinstance(data_files, list | ListConfig):
        return [str(path) for path in data_files]
    return [str(data_files)]

class ParserRLHFDataset(Dataset):
    """Load Parser parquet data and expose VERL agent-loop fields."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config,
        processor=None,
        max_samples: int = -1,
    ):
        self.data_files = _as_list(data_files)
        self.original_data_files = copy.deepcopy(self.data_files)
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.max_samples = max_samples
        self.chunk_max_length = int(config.get("chunk_max_length", 512))
        self.max_chunk_num = config.get("max_chunk_num", None)
        self.shuffle = bool(config.get("shuffle", False))
        self.seed = config.get("seed", None)

        self._read_files_and_tokenize()

    def _cache_path(self, parquet_path: str) -> Path:
        path_obj = Path(parquet_path)
        tokenizer_name = getattr(self.tokenizer, "name_or_path", "tokenizer").split("/")[-1].split("-")[0]
        return Path(
            str(path_obj.with_suffix(""))
            + f"-{tokenizer_name}-length{self.chunk_max_length}-num{self.max_chunk_num}{path_obj.suffix}"
        )

    def _process_dataset_file(self, parquet_path: str) -> datasets.Dataset:
        local_path = copy_to_local(src=parquet_path, cache_dir=self.config.get("cache_dir", "~/.cache/verl/rlhf"))
        cache_path = self._cache_path(local_path)
        if cache_path.exists():
            logger.info("Loading cached Parser dataset from %s", cache_path)
            return datasets.load_dataset("parquet", data_files=str(cache_path), split="train")

        dataset = datasets.load_dataset("parquet", data_files=str(local_path), split="train")

        def process(sample: dict[str, Any]) -> dict[str, Any]:
            documents = sample["context"]
            splits = [
                f"Paragraph {i}: {part.strip()}"
                for i, part in enumerate(re.split(r"(?:^|\n)Document\s\d+:\s*", documents))
                if part.strip()
            ]
            split_tokens = [self.tokenizer.encode(part) for part in splits]

            merged_chunks: list[str] = []
            if not splits:
                merged_chunks = []
            elif self.chunk_max_length <= 0:
                merged_chunks = splits
            else:
                cur_chunk = splits[0]
                cur_chunk_token_count = len(split_tokens[0])
                for split, tokens in zip(splits[1:], split_tokens[1:], strict=False):
                    if cur_chunk_token_count + len(tokens) < self.chunk_max_length:
                        cur_chunk = f"{cur_chunk}\n\n{split}"
                        cur_chunk_token_count += len(tokens) + 1
                    else:
                        merged_chunks.append(cur_chunk)
                        cur_chunk = split
                        cur_chunk_token_count = len(tokens) + 1
                merged_chunks.append(cur_chunk)

            return {
                "question": sample["prompt"][0]["content"],
                "chunks": merged_chunks,
                "ground_truth": sample["reward_model"]["ground_truth"],
                "data_source": sample.get("data_source", parquet_path),
            }

        dataset = dataset.map(process, remove_columns=dataset.column_names)

        if self.max_chunk_num is not None:
            max_chunk_num = int(self.max_chunk_num)
            dataset = dataset.filter(lambda sample: len(sample["chunks"]) <= max_chunk_num)

        logger.warning("Saving Parser dataset cache to %s", cache_path)
        dataset.to_parquet(str(cache_path))
        return dataset

    def _read_files_and_tokenize(self) -> None:
        dataframes = [self._process_dataset_file(path) for path in self.data_files]
        self.dataframe = datasets.concatenate_datasets(dataframes) if len(dataframes) > 1 else dataframes[0]

        total = len(self.dataframe)
        if 0 < self.max_samples < total:
            if self.shuffle:
                rng = np.random.default_rng(self.seed)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())

        logger.info("Parser dataset len: %s", len(self.dataframe))

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = dict(self.dataframe[item])
        question = row["question"]
        chunks = row.get("chunks", [])
        extra_info = dict(row.get("extra_info") or {})
        extra_info.setdefault("index", item)
        raw_prompt =[
            {"role": "system", "content": LEAD_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": LEAD_AGENT_USER_PROMPT.format(question=question)},
        ]

        return {
            "question": question,
            "chunks": chunks,
            "ground_truth": row.get("ground_truth"),
            "data_source": row.get("data_source", "parser"),
            "agent_name": "parser_lead_agent",
            "raw_prompt": raw_prompt,
            "extra_info": extra_info,
            "index": extra_info["index"],
            # "tools_kwargs": extra_info.get("tools_kwargs", {}),
            # "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
        }

    def split(self, num_splits: int) -> list["ParserRLHFDataset"]:
        if num_splits <= 0:
            raise ValueError(f"num_splits must be positive, got {num_splits}")
        total_samples = len(self.dataframe)
        split_size = total_samples // num_splits
        if split_size == 0:
            raise ValueError(f"Cannot split {total_samples} samples into {num_splits} non-empty splits")

        splits = []
        for i in range(num_splits):
            start = i * split_size
            end = (i + 1) * split_size if i < num_splits - 1 else total_samples
            split_dataset = copy.copy(self)
            split_dataset.dataframe = self.dataframe.select(range(start, end))
            splits.append(split_dataset)
        return splits
