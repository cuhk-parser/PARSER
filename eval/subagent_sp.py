"""Subagent tool implementation for Parser."""

import asyncio
import json
import logging
import random
import re

import aiohttp
import json_repair

from prompt import SUBAGENT_USER_PROMPT

logger = logging.getLogger("[SUB-AGENT]")


class Subagent:
    """A subagent with access to one document chunk."""

    def __init__(
        self,
        agent_id: int,
        chunk_content: str,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
        max_retries: int = 3,
    ):
        self.agent_id = agent_id
        self.name = f"agent_{agent_id}"
        self.chunk_content = chunk_content
        self.api_key = api_key
        self.base_url = [url.strip() for url in base_url.split(",") if url.strip()][0]
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    async def answer(self, session: aiohttp.ClientSession, query: str) -> dict[str, str] | None:
        user_prompt = SUBAGENT_USER_PROMPT.format(chunk_content=self.chunk_content, question=query)
        raw_response = await self._call_openai_api(session, [{"role": "user", "content": user_prompt}])
        if raw_response is None:
            return None
        return self._parse_response(raw_response)

    async def _call_openai_api(self, session: aiohttp.ClientSession, messages: list[dict]) -> str | None:
        stop_words =  ["unknown", "Unknown", " unknown", " Unknown"]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": stop_words,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=600),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(f"API error {resp.status}: {error_text}")
                    result = await resp.json()
                    # logger.warning(str(result["choices"][0]))
                    if result["choices"][0].get("matched_stop", None) and result["choices"][0]["matched_stop"] in stop_words:
                        return None
                    return result["choices"][0]["message"]["content"]
            except Exception as exc:
                if attempt >= self.max_retries-3:
                    logger.warning(
                        "Subagent %s API call attempt %s/%s failed: %s",
                        self.agent_id,
                        attempt,
                        self.max_retries,
                        exc,
                    )
                    if attempt > self.max_retries-1:
                        raise BaseException(f"Subagent {self.agent_id} API call attempt {attempt}/{self.max_retries} failed: {exc}")
                await asyncio.sleep(4**attempt)

        return None

    def _parse_response(self, raw_response: str) -> dict[str, str] | None:
        json_match = re.search(r"```json\s*(.*?)\s*```", raw_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            json_str = json_match.group(0) if json_match else raw_response

        try:
            json_data = json_repair.loads(json_str)
            evidence = json_data.get("evidence", None)
            answer = json_data.get("answer", None)
            return {
                "evidence": evidence if evidence else None,
                "answer": answer if answer else None,
            }
        except Exception as exc:
            logger.warning(
                "Failed to parse subagent %s response as JSON: %s %r",
                self.agent_id,
                exc,
                raw_response,
            )
            if len(raw_response.strip()) > 64:
                return {"raw_response": raw_response.strip()}
            return None


class SubagentManager:
    """Query all document subagents in parallel and aggregate valid responses."""

    def __init__(
        self,
        chunks: list[str],
        api_key: str,
        base_url: str | list[str],
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
        max_retries: int = 5,
        connector_limit: int = 128,
        max_concurrency: int = 64,
    ):
        if isinstance(base_url, list):
            base_url = ",".join(base_url)
        self.connector_limit = connector_limit
        self.max_concurrency = max_concurrency
        self.sub_agent_swarm = [
            Subagent(
                agent_id=i,
                chunk_content=chunk,
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )
            for i, chunk in enumerate(chunks)
        ]
        self.n_subagents = len(chunks)

    async def query_all(self, query: str) -> tuple[str, int]:
        connector = aiohttp.TCPConnector(
            limit=self.connector_limit,
            enable_cleanup_closed=True,
            keepalive_timeout=6,
        )
        async with aiohttp.ClientSession(connector=connector) as session:
            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def answer_with_limit(sub_agent: Subagent) -> dict[str, str] | None:
                async with semaphore:
                    return await sub_agent.answer(session, query)

            tasks = [answer_with_limit(sub_agent) for sub_agent in self.sub_agent_swarm]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        responses = {}
        for sub_agent, result in zip(self.sub_agent_swarm, results, strict=False):
            if isinstance(result, Exception):
                logger.debug("Subagent %s failed: %s", sub_agent.name, result)
                continue
            if result is None:
                # logger.debug("Subagent %s returned no result", sub_agent.name)
                continue
            if all(result.get(key, None) is None for key in ["evidence", "answer", "raw_response"]):
                continue
            responses[sub_agent.name] = {k: v for k, v in result.items() if v is not None}

        if not responses:
            # logger.info("No subagents returned relevant information for query: %r", query)
            return "No agents found relevant information for this query.", 0

        # logger.info(
        #     "Subagents returned %d/%d valid responses for query: %r",
        #     len(responses),
        #     self.n_subagents,
        #     query,
        # )
        return json.dumps(responses, indent=2, ensure_ascii=False), len(responses)
