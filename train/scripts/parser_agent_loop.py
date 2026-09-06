"""LongMas-R1 lead-agent loop for VERL fully async rollout.

Opus 为解决代理字符的冲突问题，修改了 longmas_agent_loop.py。

Root cause: lone/unpaired surrogate characters in model-generated text can crash
apply_chat_template. strip_surrogates() removes invalid surrogate code points before
re-tokenization.
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import threading
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput

from .prompt import LEAD_AGENT_SYSTEM_PROMPT, LEAD_AGENT_USER_PROMPT
from .reward import exact_match_reward
from .subagent import SubagentManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_TRAJECTORY_LOG_LOCK = threading.Lock()


def _trajectory_log_path(log_dir) -> str:
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"smoke_test_{os.getpid()}.jsonl")


def append_trajectory_log(record: dict[str, Any], log_dir: str) -> None:
    """Append one trajectory record; one jsonl file per OS process."""
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _TRAJECTORY_LOG_LOCK:
        with open(_trajectory_log_path(log_dir), "a") as f:
            f.write(line)


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
    """Remove EOS/im_end tails that decode may leave in assistant text."""
    stripped = text.rstrip()
    markers: list[str] = []
    for token in (getattr(tokenizer, "eos_token", None), getattr(tokenizer, "pad_token", None)):
        if isinstance(token, str) and token:
            markers.append(token)
    for extra in ("<|im_end|>", "<|endoftext|>"):
        if extra not in markers:
            markers.append(extra)

    while True:
        progressed = False
        for marker in markers:
            if marker and stripped.endswith(marker):
                stripped = stripped[: -len(marker)].rstrip()
                progressed = True
                break
        if not progressed:
            break
    return stripped


def extract_answer(text: str) -> str:
    """Return the last `<answer>...</answer>` block."""
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


@register("parser_lead_agent")
class ParserLeadAgentLoop(AgentLoopBase):
    """Multi-turn Parser lead agent with masked environment tokens."""

    _SOCKET_SELECTION_COUNTS: dict[str, int] = {}
    _SOCKET_SELECTION_TOTAL: int = 0

    def __init__(
        self,
        *args,
        max_iterations: int = 9,
        max_new_token_each_turn: int | None = 2048,
        subagent_config: dict[str, Any] | None = None,
        # subagent_api_key: str = "",
        # subagent_base_url: str = "http://0.0.0.0:8080",
        # subagent_model: str = "gpt-4o-mini",
        # subagent_temperature: float = 0.1,
        # subagent_max_tokens: int = 512,
        # subagent_max_retries: int = 3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_iterations = max_iterations
        self.max_new_token_each_turn = max_new_token_each_turn
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.subagent_config = subagent_config

        # self.subagent_config = subagent_config or {
        #     "api_key": subagent_api_key,
        #     "base_url": subagent_base_url,
        #     "model": subagent_model,
        #     "temperature": subagent_temperature,
        #     "max_tokens": subagent_max_tokens,
        #     "max_retries": subagent_max_retries,
        # }
        # self.subagent_config["temperature"] = float(self.subagent_config.get("temperature", 0.1))
        # self.subagent_config["max_tokens"] = int(self.subagent_config.get("max_tokens", 512))
        # self.subagent_config["max_retries"] = int(self.subagent_config.get("max_retries", 3))

    def _chat_template_ids(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> list[int]:
        kwargs: dict[str, Any] = {
            "tools": SUBAGENT_QUERY_TOOLS,
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
        }

        ids = self.tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=True)
        return normalize_token_ids(ids)

    @staticmethod
    def _template_diff_token_append(t_before: list[int], t_after: list[int], seq: list[int], eos_token_id: int) -> list[int]:
        if not (len(t_after) > len(t_before) and t_after[: len(t_before)] == t_before):
            raise ValueError(f"this env append is invalid (before={t_before}, after={t_after}); ")
        new_ids = t_after[len(t_before) - 1 :] # seq does not include the last token of t_before,\n, so -1
        assert new_ids[0] == 198
        
        if len(seq)>0 and seq[-1] != eos_token_id:
            new_ids = [eos_token_id] + new_ids
        return new_ids # \n<|im_start|>user[obs]\n<|im_end|>\n<|im_start|>assistant\n<think>

    @staticmethod
    def _merge_extra_fields(extra_fields: dict[str, Any], output_extra: dict[str, Any]) -> None:
        if not output_extra:
            return
        if output_extra.get("min_global_steps") is not None:
            current_min = extra_fields.get("min_global_steps")
            output_min = output_extra["min_global_steps"]
            extra_fields["min_global_steps"] = output_min if current_min is None else min(current_min, output_min)
        if output_extra.get("max_global_steps") is not None:
            current_max = extra_fields.get("max_global_steps")
            output_max = output_extra["max_global_steps"]
            extra_fields["max_global_steps"] = output_max if current_max is None else max(current_max, output_max)
        if output_extra.get("global_steps") is not None:
            extra_fields["global_steps"] = output_extra["global_steps"]

    def _merged_sampling_params(self, sampling_params: dict[str, Any]) -> dict[str, Any]:
        """Merge stop_token_ids for chat EOS and per-turn max_new_tokens.
        SGLang with ``skip_tokenizer_init=True`` does not attach a tokenizer to decode-time
        stop checks, so only ``hf_eos_token_id`` from config applies (often ``<|endoftext|>``).
        Qwen chat templates end assistant turns with ``tokenizer.eos_token`` (typically
        ``<|im_end|>``), which must be listed explicitly or generation continues past it.
        """
        out = dict(sampling_params)
        stop_ids = list(out.get("stop_token_ids") or [])
        for tid in (
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "pad_token_id", None),
        ):
            if tid is not None and tid not in stop_ids:
                stop_ids.append(tid)
        if stop_ids:
            out["stop_token_ids"] = stop_ids
        if self.max_new_token_each_turn is not None:
            out["max_new_tokens"] = self.max_new_token_each_turn
        return out

    def _select_socket_path(self, question: str, socket_paths: Any) -> str | None:
        if socket_paths is None:
            return None
        if isinstance(socket_paths, list):
            candidates = [str(path).strip() for path in socket_paths if str(path).strip()]
        elif isinstance(socket_paths, str):
            candidates = [path.strip() for path in socket_paths.split(",") if path.strip()]
        else:
            candidates = [str(socket_paths).strip()]
        if not candidates:
            return None
        if len(candidates) == 1:
            selected = candidates[0]
        else:
            digest = hashlib.sha256(question.encode("utf-8")).digest()
            selected = candidates[int.from_bytes(digest[:8], "big") % len(candidates)]

        cls = type(self)
        cls._SOCKET_SELECTION_COUNTS[selected] = cls._SOCKET_SELECTION_COUNTS.get(selected, 0) + 1
        cls._SOCKET_SELECTION_TOTAL += 1
        # if cls._SOCKET_SELECTION_TOTAL % 1000 == 0:
        #     logger.warning(
        #         "Subagent socket selection counts after %s selections: %s",
        #         cls._SOCKET_SELECTION_TOTAL,
        #         cls._SOCKET_SELECTION_COUNTS,
        #     )
        return selected

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        question = kwargs["question"]
        chunks = kwargs.get("chunks", [])
        answer = kwargs.get("ground_truth", None)
        
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": LEAD_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": LEAD_AGENT_USER_PROMPT.format(question=question)},
        ]
        initial_prompt_ids = self._chat_template_ids(messages, add_generation_prompt=True)

        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []

        completion_str = ""
        iteration = 1
        valid_action_num = 0
        invalid_action_num = 0
        tool_calls_per_turn: list[int] = []
        subagent_replies_per_query: list[int] = []
        metrics: dict[str, Any] = {}
        extra_fields: dict[str, Any] = {"turn_scores": [], "tool_rewards": [], "extras": {}}
        request_id = uuid4().hex

        subagent_config = dict(self.subagent_config)
        socket_path = self._select_socket_path(question, subagent_config.get("unix_socket_path"))
        if socket_path:
            subagent_config["unix_socket_path"] = socket_path
        base_url = subagent_config.get("base_url", "")
        if not subagent_config.get("unix_socket_path"):
            if isinstance(base_url, list):
                base_url = random.choice(base_url)
            elif isinstance(base_url, str) and "," in base_url:
                urls = [url.strip() for url in base_url.split(",") if url.strip()]
                base_url = random.choice(urls) if urls else base_url
        subagent_config["base_url"] = base_url
        subagent_manager = SubagentManager(chunks=list(chunks), **subagent_config)
        illegal_hint = (
            "Invalid step: start with a short Thinking section (what you know, what you need, "
            "what you will do), then either call `query_agents` via "
            '<tool_call>...</tool_call> with `{"query": "..."}`, or give the final answer as '
            "<answer>your answer</answer>."
        )
        
        while iteration < self.max_iterations + 1:
            if len(response_ids) >= self.response_length:
                logger.warning("Reached max length: response=%s", len(response_ids))
                break

            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=list(initial_prompt_ids+response_ids),
                    sampling_params=self._merged_sampling_params(sampling_params),
                )

            if metrics.get("num_preempted") is None:
                metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
            else:
                metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

            self._merge_extra_fields(extra_fields, output.extra_fields)
            out_toks = list(output.token_ids or [])
            if not out_toks:
                break

            response_ids.extend(out_toks)
            response_mask.extend([1] * len(out_toks))
            response_logprobs.extend(list[float](output.log_probs))

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
                tool_calls_per_turn.append(len(tool_calls))
                messages.append(
                    {
                        "role": "assistant",
                        "content": content if content else None,
                        "tool_calls": tool_calls,
                    }
                )
                t0 = self._chat_template_ids(messages, add_generation_prompt=False)

                for tool_call in tool_calls:
                    fn = tool_call.get("function") or {}
                    name = fn.get("name")
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    if name != "query_agents":
                        obs = json.dumps({"error": f"unknown tool {name!r}, expected query_agents"}, ensure_ascii=False)
                    else:
                        query = args.get("query") or ""
                        if not isinstance(query, str):
                            obs = json.dumps({"error": f"invalid query type {type(query)!r}, expected str"}, ensure_ascii=False)
                        else:
                            query = query.strip()
                            if not query:
                                obs = json.dumps({"error": "missing required argument 'query'"}, ensure_ascii=False)
                            else:
                                with simple_timer("tool_calls", metrics):
                                    obs, agent_resp_num = await subagent_manager.query_all(query)
                                subagent_replies_per_query.append(agent_resp_num)

                    messages.append({"role": "tool", "name": name or "query_agents", "content": obs})

                t1 = self._chat_template_ids(messages, add_generation_prompt=True)
                env_ids = self._template_diff_token_append(t0, t1, seq=response_ids, eos_token_id=self.tokenizer.eos_token_id)
                response_ids.extend(env_ids)
                response_mask.extend([0] * len(env_ids))
                response_logprobs.extend([0.0] * len(env_ids))

                valid_action_num += 1
                iteration += 1
                continue

            if extract_answer(cur_chunk) != "No answer found":
                messages.append({"role": "assistant", "content": content})
                valid_action_num += 1
                iteration += 1
                break

            messages.append({"role": "assistant", "content": content})
            t0 = self._chat_template_ids(messages, add_generation_prompt=False)
            messages.append({"role": "user", "content": illegal_hint})
            t1 = self._chat_template_ids(messages, add_generation_prompt=True)
            env_ids = self._template_diff_token_append(t0, t1, seq=response_ids, eos_token_id=self.tokenizer.eos_token_id)
            response_ids.extend(env_ids)
            response_mask[-len(out_toks):] = [-1] * len(out_toks) # ignore the tokens of the previous response, which is illegal
            response_mask.extend([0] * len(env_ids))
            response_logprobs.extend([0.0] * len(env_ids))
            invalid_action_num += 1
            iteration += 1

        prediction = extract_answer(completion_str)
        reward = exact_match_reward(prediction, answer)
        illegal_token_mask = 1 if reward < 0.2 else 0
        response_mask = [illegal_token_mask if mask == -1 else mask for mask in response_mask]
        has_answer = 1 if prediction and prediction.strip() not in ["No answer found", "and"] else 0
        # exit()

        extra_fields["reward_extra_info"] = {
            "prediction": prediction,
            "valid_actions": valid_action_num,
            "invalid_actions": invalid_action_num,
            "unpaired_tool_call_turns": count_unpaired_tool_call_turns(messages),
            "iterations": iteration-1,
            "has_answer": has_answer,
            "avg_agent_responses": (
                sum(subagent_replies_per_query) / len(subagent_replies_per_query)
                if subagent_replies_per_query
                else 0
            ),
            "tool_calls_per_turn": tool_calls_per_turn,
            "subagent_replies_per_query": subagent_replies_per_query,
            "length": len(response_ids),
        }

        append_trajectory_log(
            {
                "question": question,
                "ground_truth": answer,
                "prediction": self.tokenizer.decode(
                    response_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "reward": reward,
                "extra_fields": extra_fields,
            }
        , log_dir="rollouts/qwen35-9b-625")

        return AgentLoopOutput(
            prompt_ids=initial_prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length],
            multi_modal_data={},
            reward_score=reward,
            num_turns=iteration-1,
            metrics=metrics,
            extra_fields=extra_fields,
        )
