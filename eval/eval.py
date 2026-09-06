#!/usr/bin/env python3
"""Offline evaluation script using sglang engine.

This script mirrors the core episode loop in workflow_lead_agent.py while
running generation locally through sglang offline mode.
"""

import argparse
import asyncio
import json
import logging
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sglang as sgl
from transformers import AutoTokenizer
from tqdm import tqdm
from subagent_sp import SubagentManager
from prompt import LEAD_AGENT_SYSTEM_PROMPT, LEAD_AGENT_USER_PROMPT
from reward import exact_match_reward, sub_em_reward


# Tool schema for Hugging Face / Qwen `apply_chat_template(..., tools=...)`.
SUBAGENT_QUERY_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_agents",
            "description": (
                "Broadcast a query to all document agents in parallel. Each agent reads only "
                "its chunk and may return evidence and a partial answer. The tool response is a "
                "JSON string of aggregated findings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Question or search query to send to every agent.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

def get_tokenizer_identifier(tokenizer: Any) -> str:
    tokenizer_name_or_path = str(
        getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
    )
    return Path(tokenizer_name_or_path).name or tokenizer_name_or_path


def load_preprocessed_eval_dataset(
    data_folder: str,
    tokenizer,
    chunk_max_length: int = 512,
) -> list[dict[str, Any]]:
    data_path = Path(data_folder)
    tokenizer_identifier = get_tokenizer_identifier(tokenizer)
    preprocessed_file = data_path / (
        f"eval_preprocessed_{tokenizer_identifier}_{chunk_max_length}.pkl"
    )
    if not preprocessed_file.exists():
        raise FileNotFoundError(
            f"Preprocessed eval dataset not found: {preprocessed_file}. "
            "Run preprocess_eval_data.py first."
        )

    with preprocessed_file.open("rb") as f:
        dataset = pickle.load(f)
    if not isinstance(dataset, list):
        raise ValueError(
            f"Unexpected pickle format in {preprocessed_file.name}, expected list"
        )
    return dataset

def _normalize_json_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        raw = arguments.strip()
        if raw.startswith("{") or raw.startswith("["):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {"_raw": arguments}
            if isinstance(parsed, dict):
                return parsed
            return {"_raw": parsed}
        return {"_raw": arguments}
    return {"_raw": arguments}


def _append_json_tool_call(spec: Any, out: list[dict[str, Any]]) -> None:
    if isinstance(spec, list):
        for item in spec:
            _append_json_tool_call(item, out)
        return
    if not isinstance(spec, dict):
        return

    fn = spec.get("function") if isinstance(spec.get("function"), dict) else spec
    name = fn.get("name")
    if not isinstance(name, str) or not name.strip():
        return
    arguments = _normalize_json_arguments(fn.get("arguments", {}))
    out.append(
        {
            "type": "function",
            "function": {"name": name.strip(), "arguments": arguments},
        }
    )


def _parse_xml_parameter_value(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        return ""
    if value.startswith("{") or value.startswith("[") or (
        value.startswith('"') and value.endswith('"')
    ):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _unescape_payload_for_xml(payload: str) -> str:
    # Qwen sometimes emits stringified blocks with literal "\n" separators.
    return (
        payload.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .strip()
    )


def _parse_tool_call_block(raw_payload: str) -> list[dict[str, Any]]:
    payload = raw_payload.strip()
    if not payload:
        return []

    parsed_calls: list[dict[str, Any]] = []
    if payload.startswith("{") or payload.startswith("["):
        try:
            spec = json.loads(payload)
        except json.JSONDecodeError:
            spec = None
        if spec is not None:
            _append_json_tool_call(spec, parsed_calls)
            if parsed_calls:
                return parsed_calls

    payload_xml = _unescape_payload_for_xml(payload)
    for match in re.finditer(
        r"<function\s*=\s*([^>\n]+)>\s*(.*?)\s*</function>",
        payload_xml,
        flags=re.DOTALL,
    ):
        name = match.group(1).strip()
        if not name:
            continue
        body = match.group(2)
        arguments: dict[str, Any] = {}
        for param_match in re.finditer(
            r"<parameter\s*=\s*([^>\n]+)>\s*(.*?)\s*</parameter>",
            body,
            flags=re.DOTALL,
        ):
            key = param_match.group(1).strip()
            if not key:
                continue
            arguments[key] = _parse_xml_parameter_value(param_match.group(2))

        parsed_calls.append(
            {
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )

    return parsed_calls

def normalize_token_ids(tokenized_output) -> list[int]:
    """Normalize tokenizer outputs into a flat ``list[int]``.

    This handles Transformers 4/5 differences where ``apply_chat_template(tokenize=True)``
    may return either ``list[int]`` or a ``BatchEncoding``/mapping with ``input_ids``.
    """

    token_ids = tokenized_output
    if isinstance(tokenized_output, dict):
        if "input_ids" in tokenized_output:
            token_ids = tokenized_output["input_ids"]
    elif hasattr(tokenized_output, "input_ids"):
        token_ids = tokenized_output.input_ids

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)

    if isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], list | tuple):
        token_ids = list(token_ids[0])

    if not isinstance(token_ids, list):
        raise TypeError(f"token_ids must be list-like token ids, got {type(token_ids).__name__}: {token_ids!r}")

    normalized_ids = []
    for idx, token_id in enumerate(token_ids):
        if hasattr(token_id, "item"):
            token_id = token_id.item()
        try:
            normalized_ids.append(int(token_id))
        except (TypeError, ValueError) as e:
            raise TypeError(f"token_id must be int-convertible, got {type(token_id).__name__}: {token_id!r}") from e
    return normalized_ids

def parse_tool_calls_from_text(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse tool calls in Qwen3 JSON and Qwen3.5 XML formats."""
    tool_calls: list[dict[str, Any]] = []
    out_chunks: list[str] = []
    pos = 0
    while True:
        start = text.find("<tool_call>", pos)
        if start == -1:
            out_chunks.append(text[pos:])
            break
        out_chunks.append(text[pos:start])
        end = text.find("</tool_call>", start)
        if end == -1:
            out_chunks.append(text[start:])
            break

        raw_payload = text[start + len("<tool_call>") : end]
        pos = end + len("</tool_call>")
        parsed = _parse_tool_call_block(raw_payload)
        if parsed:
            tool_calls.extend(parsed)
            continue

        # Preserve malformed blocks in content for debugging visibility.
        out_chunks.append(text[start:pos])

    content = "".join(out_chunks).strip()
    return content, tool_calls


def count_unpaired_tool_call_turns(messages: list[dict[str, Any]]) -> int:
    """Count turns whose content has mismatched tool_call XML tags."""
    unpaired_turns = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if content.count("<tool_call>") != content.count("</tool_call>"):
            unpaired_turns += 1
    return unpaired_turns


def strip_trailing_chat_markers(text: str, tokenizer: Any) -> str:
    """Remove EOS / im_end tails that sglang decode may leave in assistant text.

    These must not be stored in `messages["content"]` or `apply_chat_template` will
    duplicate or corrupt turns.
    """
    s = text.rstrip()
    markers: list[str] = []
    for t in (
        getattr(tokenizer, "eos_token", None),
        getattr(tokenizer, "pad_token", None),
    ):
        if isinstance(t, str) and t:
            markers.append(t)

    for extra in ("<|im_end|>", "<|endoftext|>"):
        if extra not in markers:
            markers.append(extra)
    while True:
        progressed = False
        for m in markers:
            if m and s.endswith(m):
                s = s[: -len(m)].rstrip()
                progressed = True
                break
        if not progressed:
            break
    return s

def extract_answer(text: str) -> str:
    """Take the contents of the last <answer>...</answer>."""
    open_tag = "<answer>"
    close_tag = "</answer>"
    close_pos = text.rfind(close_tag)
    if close_pos == -1:
        return "No answer found"
    open_pos = text.rfind(open_tag, 0, close_pos)
    if open_pos == -1:
        return "No answer found"
    inner = text[open_pos + len(open_tag) : close_pos].strip()
    return inner if inner else "No answer found"

def extract_answer_star(text: str) -> str:
    # extract the last string wrapped within **...**
    matches = re.findall(r"\*\*(.*?)\*\*", text)
    if matches:
        inner = matches[-1].strip()
        return inner if inner else "No answer found"
    return "No answer found"

@dataclass
class EvalConfig:
    model_path: str
    tokenizer_path: str
    data_path: str
    output_path: str
    chunk_max_length: int = 512
    max_iterations: int = 16
    max_tokens_per_trajectory: int = 32764
    lead_max_new_tokens: int = 8192
    subagent_api_key: str = ""
    subagent_base_url: str = "http://0.0.0.0:30002/v1"
    subagent_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    subagent_temperature: float = 0.7
    subagent_max_tokens: int = 512
    subagent_max_retries: int = 6
    context_length: int = 32764
    mem_fraction_static: float = 0.9
    num_samples: int | None = None
    concurrent_process_num: int = 1
    log_level: str = "DEBUG"


class workflow_lead_agent:
    def __init__(self, engine: sgl.Engine, tokenizer, cfg: EvalConfig):
        self.engine = engine
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.stop_words = [self.tokenizer.eos_token, self.tokenizer.pad_token]

    def _chat_template_ids(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> list[int]:
        kwargs: dict[str, Any] = {
            "tools": SUBAGENT_QUERY_TOOLS,
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
        }
        ids = self.tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=True)
        return normalize_token_ids(ids)

    @staticmethod
    def _append_masked(
        seq: list[int],
        logprobs: list[float],
        loss_mask: list[int],
        versions: list[int],
        new_ids: list[int],
        *,
        log_val: float,
        mask: int,
        ver_val: int,
    ) -> None:
        n = len(new_ids)
        seq.extend(new_ids)
        logprobs.extend([log_val] * n)
        loss_mask.extend([mask] * n)
        versions.extend([ver_val] * n)

    def _template_diff_token_append(
        self,
        t_before: list[int],
        t_after: list[int],
        seq: list[int],
        logprobs: list[float],
        loss_mask: list[int],
        versions: list[int],
    ) -> None:
        if not (len(t_after) > len(t_before) and t_after[: len(t_before)] == t_before):
            # print(self.tokenizer.decode(t_before, skip_special_tokens=False, clean_up_tokenization_spaces=False))
            # print("↓↓↓↓↓")
            # print(self.tokenizer.decode(t_after, skip_special_tokens=False, clean_up_tokenization_spaces=False))
            raise ValueError(
                "this env append is invalid (before=%d, after=%d); " % (
                len(t_before),
                len(t_after),
            ))

        new_ids = t_after[len(t_before)-1:] # seq does not include the last token of t_before,\n
        assert new_ids[0] == 198 # \n

        if seq[-1] != self.tokenizer.eos_token_id: # the last turn does not end with eos, so we add eos to the new ids to make it end with eos
            new_ids = [self.tokenizer.eos_token_id] + new_ids

        if new_ids:
            self._append_masked(
                seq,
                logprobs,
                loss_mask,
                versions,
                new_ids,
                log_val=0.0,
                mask=0,
                ver_val=-1,
            )

    async def arun_episode(self, data: dict[str, Any]) -> dict[str, Any]:
        question = data["question"]
        chunks = data.get("chunks", [])
        answer = data.get("ground_truth", None)
        domain = data.get("domain", "unknown_domain")
        
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": LEAD_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": LEAD_AGENT_USER_PROMPT.format(question=question)},
        ]

        P = self._chat_template_ids(messages, add_generation_prompt=True)
        seq: list[int] = list(P)
        logprobs = [0.0] * len(P)
        loss_mask = [0] * len(P)
        versions = [-1] * len(P)

        completion_str = ""
        iteration = 1
        valid_action_num = 0
        invalid_action_num = 0
        total_wait_time: list[float] = []
        agent_resp_nums: list[int] = []

        subagent_manager = SubagentManager(
            chunks=chunks,
            api_key=self.cfg.subagent_api_key,
            base_url=self.cfg.subagent_base_url,
            model=self.cfg.subagent_model,
            temperature=self.cfg.subagent_temperature,
            max_tokens=self.cfg.subagent_max_tokens,
            max_retries=self.cfg.subagent_max_retries,
        )

        illegal_hint = (
            "Invalid step: start with a short Thinking section (what you know, what you need, "
            "what you will do), then either call `query_agents` via "
            '<tool_call>...</tool_call> with `{"query": "..."}`, or give the final answer as '
            "<answer>your answer</answer>."
        )

        while iteration < self.cfg.max_iterations + 1:
            # Rollout context is `seq` only: do not rebuild from `decode(out_toks)` → messages.
            P = list(seq)
            if len(P) >= 120000:
                break
            # print(">>>>>")
            # print(self.tokenizer.decode(P, skip_special_tokens=False, clean_up_tokenization_spaces=False))
            # print("↓↓↓↓↓")
            resp = await self.engine.async_generate(
                input_ids=P,
                sampling_params={
                    "temperature": 0.0,
                    "stop": self.stop_words,
                    "max_new_tokens": 2048 # TODO: to be modified
                    
                },
            )

            out_toks =  resp.get("output_ids", [])
            # print(self.tokenizer.decode(out_toks, skip_special_tokens=False, clean_up_tokenization_spaces=False))
            if not out_toks:
                break

            seq.extend(out_toks)

            # Decode only for control flow / reward (`completion_str`); not written back to `messages`.
            cur_chunk = self.tokenizer.decode(
                out_toks,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            cur_chunk = strip_trailing_chat_markers(cur_chunk, self.tokenizer)
            completion_str += cur_chunk

            content, tool_calls = parse_tool_calls_from_text(cur_chunk)

            if content:
                content = strip_trailing_chat_markers(content, self.tokenizer)
                
            if tool_calls:
                # Structured `tool_calls` from XML JSON only; `content: None` — no `decode(out)` in chat list.
                messages.append(
                    {
                        "role": "assistant",
                        "content": content if content else None,
                        "tool_calls": tool_calls,
                    }
                )
                t0 = self._chat_template_ids(messages, add_generation_prompt=False)
                start_t = time.perf_counter()

                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    name = fn.get("name")
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    if name != "query_agents":
                        obs = json.dumps(
                            {"error": f"unknown tool {name!r}, expected query_agents"},
                            ensure_ascii=False,
                        )
                    else:
                        query = (args.get("query") or "").strip()
                        if not query:
                            obs = json.dumps(
                                {"error": "missing required argument 'query'"},
                                ensure_ascii=False,
                            )
                        else:
                            responses_json, agent_resp_num = await subagent_manager.query_all(
                                query
                            )
                            agent_resp_nums.append(agent_resp_num)
                            obs = responses_json

                    messages.append(
                        {"role": "tool", "name": name or "query_agents", "content": obs}
                    )
                

                t1 = self._chat_template_ids(messages, add_generation_prompt=True)
                self._template_diff_token_append(t0, t1, seq, logprobs, loss_mask, versions)
                # add_generation_prompt is false for t0 while true for t1 because  the utterance to be concatenated starts (obs or illegal_hint)should start with user, not assistant

                total_wait_time.append(time.perf_counter() - start_t)
                valid_action_num += 1
                iteration += 1
                continue
            # No tool calls and a final answer.
            elif extract_answer(cur_chunk) != "No answer found":
                messages.append({"role": "assistant", "content": content})
                valid_action_num += 1
                # print(">>>>>")
                # print(self.tokenizer.decode(seq, skip_special_tokens=False))
                # print("↓↓↓↓↓")
                # print(self.tokenizer.apply_chat_template(messages, tools=SUBAGENT_QUERY_TOOLS, add_generation_prompt=False, tokenize=False))
                break
             # Illegal step: no tool calls and no final answer.
            else:
                messages.append({"role": "assistant", "content": content})
                t0 = self._chat_template_ids(messages, add_generation_prompt=False)
                messages.append({"role": "user", "content": illegal_hint})
                t1 = self._chat_template_ids(messages, add_generation_prompt=True)
                # add_generation_prompt is false for t0 while true for t1 because  the utterance to be concatenated (obs or illegal_hint) should start with user, not assistant
                self._template_diff_token_append(t0, t1, seq, logprobs, loss_mask, versions)
                invalid_action_num += 1
                iteration += 1

        prediction = extract_answer(completion_str)
        reward = exact_match_reward(prediction, answer)
        sub_em = sub_em_reward(prediction, answer)
        prediction_star = extract_answer_star(completion_str)
        reward_star = max(reward, exact_match_reward(prediction_star, answer))
        sub_em_star = max(sub_em, sub_em_reward(prediction_star, answer))
        
        if sub_em_star > sub_em:
            prediction = prediction_star
            reward = reward_star
            sub_em = sub_em_star
        
        return {
            "question": question,
            "prediction": prediction,
            "ground_truth": answer,
            "reward": reward,
            "sub_em": sub_em,
            "domain": domain,
            "iterations": iteration,
            "valid_actions": valid_action_num,
            "invalid_actions": invalid_action_num,
            "avg_agent_responses": (
                sum(agent_resp_nums) / len(agent_resp_nums) if agent_resp_nums else 0
            ),
            "wait_for_subagents_seconds": (
                sum(total_wait_time) / len(total_wait_time) if total_wait_time else 0
            ),
            "length": len(seq),
            "unpaired_tool_call_turns": count_unpaired_tool_call_turns(messages),
            "completion": messages,
        }


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Offline sglang eval for Parser")
    parser.add_argument("--model-path", required=True, help="Model path for sglang engine")
    parser.add_argument("--tokenizer-path", default=None, help="Tokenizer path (default: model path)")
    parser.add_argument(
        "--data-path",
        default="",
        help="Path to folder containing preprocessed eval dataset file",
    )
    parser.add_argument(
        "--output-path",
        default="eval_output.jsonl",
        help="Path to write evaluation results",
    )
    parser.add_argument("--chunk-max-length", type=int, default=512)
    parser.add_argument("--max-iterations", type=int, default=128)
    parser.add_argument("--max-tokens-per-trajectory", type=int, default=32764)
    parser.add_argument("--lead-max-new-tokens", type=int, default=1024)
    parser.add_argument("--subagent-temperature", type=float, default=0.7)
    parser.add_argument("--subagent-max-tokens", type=int, default=512)
    parser.add_argument("--subagent-max-retries", type=int, default=3)
    parser.add_argument("--context-length", type=int, default=32764)
    parser.add_argument("--mem-fraction-static", type=float, default=0.9)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--subagent-api-key", type=str, default="")
    parser.add_argument("--subagent-base-url", type=str, default="http://0.0.0.0:8006/v1")
    parser.add_argument("--subagent-model", type=str, default="Qwen3.5-4B")
    parser.add_argument(
        "--concurrent-process-num",
        type=int,
        default=4,
        help="Number of evaluation samples to process concurrently",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging level (default: DEBUG)",
    )
    args = parser.parse_args()

    tokenizer_path = args.tokenizer_path or args.model_path
    return EvalConfig(
        model_path=args.model_path,
        tokenizer_path=tokenizer_path,
        data_path=args.data_path,
        output_path=args.output_path,
        num_samples=args.num_samples,
        chunk_max_length=args.chunk_max_length,
        max_iterations=args.max_iterations,
        max_tokens_per_trajectory=args.max_tokens_per_trajectory,
        lead_max_new_tokens=args.lead_max_new_tokens,
        subagent_api_key=args.subagent_api_key,
        subagent_base_url=args.subagent_base_url,
        subagent_model=args.subagent_model,
        subagent_temperature=args.subagent_temperature,
        subagent_max_tokens=args.subagent_max_tokens,
        subagent_max_retries=args.subagent_max_retries,
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static,
        concurrent_process_num=args.concurrent_process_num,
        log_level=args.log_level,
    )


async def amain(cfg: EvalConfig) -> None:
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path, trust_remote_code=True)
    dataset = load_preprocessed_eval_dataset(
        data_folder=cfg.data_path,
        tokenizer=tokenizer,
        chunk_max_length=cfg.chunk_max_length,
    )
    if cfg.num_samples is not None:
        dataset = dataset[: min(cfg.num_samples, len(dataset))]

    engine = sgl.Engine(
        model_path=cfg.model_path,
        # context_length=cfg.context_length,
        mem_fraction_static=cfg.mem_fraction_static,
    )
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lead_agent = workflow_lead_agent(engine=engine, tokenizer=tokenizer, cfg=cfg)
    correct = 0
    sub_em_correct = 0
    total = 0

    concurrent_process_num = max(1, cfg.concurrent_process_num)
    semaphore = asyncio.Semaphore(concurrent_process_num)

    async def process_sample(index: int, sample: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            start_t = time.perf_counter()
            result = await lead_agent.arun_episode(sample)
            result["processing_time_seconds"] = time.perf_counter() - start_t
            return index, result

    try:
        write_batch_size = 10
        next_write_idx = 0
        in_order_buffer: list[dict[str, Any]] = []
        completed_results: dict[int, dict[str, Any]] = {}
        tasks = [
            asyncio.create_task(process_sample(index, sample))
            for index, sample in enumerate(dataset)
        ]
        with output_path.open("w", encoding="utf-8") as f:
            with tqdm(total=len(tasks), desc="Evaluating", unit="sample") as pbar:
                for task in asyncio.as_completed(tasks):
                    index, result = await task
                    completed_results[index] = result
                    pbar.update(1)

                    # Flush only when we have an in-order contiguous block.
                    while next_write_idx in completed_results:
                        in_order_buffer.append(completed_results.pop(next_write_idx))
                        next_write_idx += 1

                        if len(in_order_buffer) >= write_batch_size:
                            for buffered_result in in_order_buffer:
                                total += 1
                                if buffered_result["reward"] > 0.99:
                                    correct += 1
                                if buffered_result["sub_em"] > 0.99:
                                    sub_em_correct += 1
                                f.write(json.dumps(buffered_result, ensure_ascii=False) + "\n")
                                if total % 10 == 0:
                                    acc = correct / total if total else 0.0
                                    sub_em_acc = sub_em_correct / total if total else 0.0
                                    print(
                                        f"[{total}] running exact-match: {acc:.4f}, "
                                        f"sub_em: {sub_em_acc:.4f}"
                                    )
                            f.flush()
                            in_order_buffer.clear()

            if in_order_buffer:
                for buffered_result in in_order_buffer:
                    total += 1
                    if buffered_result["reward"] > 0.99:
                        correct += 1
                    if buffered_result["sub_em"] > 0.99:
                        sub_em_correct += 1
                    f.write(json.dumps(buffered_result, ensure_ascii=False) + "\n")
                    if total % 10 == 0:
                        acc = correct / total if total else 0.0
                        sub_em_acc = sub_em_correct / total if total else 0.0
                        print(
                            f"[{total}] running exact-match: {acc:.4f}, "
                            f"sub_em: {sub_em_acc:.4f}"
                        )
                f.flush()
    finally:
        engine.shutdown()

    acc = correct / total if total else 0.0
    sub_em_acc = sub_em_correct / total if total else 0.0
    print(
        f"Done. total={total}, exact_match={acc:.4f}, "
        f"sub_em={sub_em_acc:.4f}, output={output_path}"
    )


def setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.DEBUG)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        force=True,
    )
    logging.getLogger("[SUB-AGENT]").setLevel(numeric_level)
    logging.getLogger("sglang").setLevel(logging.WARNING)
    logging.getLogger("sgl_kernel").setLevel(logging.WARNING)


def main() -> None:
    cfg = parse_args()
    setup_logging(cfg.log_level)
    print(cfg)
    asyncio.run(amain(cfg))


if __name__ == "__main__":
    main()
