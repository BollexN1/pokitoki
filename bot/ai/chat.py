"""AI language model compatible with multiple providers (OpenAI, DeepSeek, Gemini)."""

import logging
import httpx
import requests
import json
from typing import Optional, List, Dict, Iterator
from bot.config import config

client = httpx.AsyncClient(timeout=60.0)
logger = logging.getLogger(__name__)

# Known models and their context windows
MODELS = {
    # Gemini
    "gemini-2.0-flash": 1_048_576,
    "gemini-1.5-flash": 1_048_576,
    "gemini-1.5-flash-8b": 1_048_576,
    "gemini-1.5-pro": 2_097_152,
    # OpenAI
    "o1": 200000,
    "o1-preview": 128000,
    "o1-mini": 128000,
    "o3-mini": 200000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4-vision-preview": 128000,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-3.5-turbo": 16385,
    # DeepSeek
    "deepseek-chat": 128000,  # DeepSeek-V3
    "deepseek-reasoner": 128000,  # DeepSeek-R1
}

# Prompt role name overrides.
ROLE_OVERRIDES = {
    "o1": "user",
    "o1-preview": "user",
    "o1-mini": "user",
    "o3-mini": "user",
    "deepseek-chat": "system",
    "deepseek-reasoner": "system",
}
# Model parameter overrides.
PARAM_OVERRIDES = {
    "o1": lambda params: {},
    "o1-preview": lambda params: {},
    "o1-mini": lambda params: {},
    "o3-mini": lambda params: {},
    "deepseek-chat": lambda params: params,
    "deepseek-reasoner": lambda params: params,
}


class Model:
    """AI API wrapper."""

    def __init__(self, name: str) -> None:
        """Creates a wrapper for a given AI large language model."""
        self.name = name

    async def ask(self, prompt: str, question: str, history: List[tuple[str, str]]) -> str:
        """Asks the language model a question and returns an answer."""
        model = self.name
        prompt_role = ROLE_OVERRIDES.get(model) or "system"
        params_func = PARAM_OVERRIDES.get(model) or (lambda params: params)

        n_input = _calc_n_input(model, n_output=config.ai.params["max_tokens"])
        messages = self._generate_messages(prompt_role, prompt, question, history)
        messages = shorten(messages, length=n_input)

        params = params_func(config.ai.params)
        logger.debug(
            f"> chat request: model=%s, params=%s, messages=%s",
            model,
            params,
            messages,
        )

        if config.ai.provider == "deepseek":
            return await self._deepseek_ask(messages, model, params)
        
        response = await client.post(
            f"{config.ai.url}/chat/completions",
            headers={"Authorization": f"Bearer {config.ai.api_key}"},
            json={
                "model": model,
                "messages": messages,
                **params,
            },
        )
        resp = response.json()
        if "usage" not in resp:
            raise Exception(resp)
        logger.debug(
            "< chat response: prompt_tokens=%s, completion_tokens=%s, total_tokens=%s",
            resp["usage"]["prompt_tokens"],
            resp["usage"]["completion_tokens"],
            resp["usage"]["total_tokens"],
        )
        answer = self._prepare_answer(resp)
        return answer

    async def chat_completion(
        self,
        prompt: str,
        question: str,
        history: List[tuple[str, str]],
        thinking_enabled: bool = False,
        search_enabled: bool = False
    ) -> Iterator[Dict]:
        """Asks the language model a question and streams the response with thinking/search steps."""
        if config.ai.provider != "deepseek":
            yield {"type": "text", "content": "Streaming is only supported with DeepSeek provider."}
            return

        model = "deepseek-reasoner" if thinking_enabled or search_enabled else self.name
        prompt_role = ROLE_OVERRIDES.get(model) or "system"
        params_func = PARAM_OVERRIDES.get(model) or (lambda params: params)

        n_input = _calc_n_input(model, n_output=config.ai.params["max_tokens"])
        messages = self._generate_messages(prompt_role, prompt, question, history)
        messages = shorten(messages, length=n_input)

        params = params_func(config.ai.params)
        logger.debug(
            f"> streaming chat request: model=%s, params=%s, messages=%s, thinking=%s, search=%s",
            model,
            params,
            messages,
            thinking_enabled,
            search_enabled,
        )

        headers = {
            "Authorization": f"Bearer {config.ai.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": messages,
            "stream": True,
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
            **params
        }

        response = requests.post(
            f"{config.ai.url}/chat/completions",
            headers=headers,
            json=data,
            stream=True
        )
        response.raise_for_status()

        buffer = ""
        for chunk in response.iter_lines():
            if chunk:
                chunk_str = chunk.decode("utf-8")
                if chunk_str.startswith("data: "):
                    chunk_data = chunk_str[6:]
                    if chunk_data == "[DONE]":
                        break
                    try:
                        chunk_json = json.loads(chunk_data)
                        delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        chunk_type = delta.get("type", "text")
                        if chunk_type in ["thinking", "text"]:
                            yield {"type": chunk_type, "content": content}
                            if chunk_type == "text":
                                buffer += content
                    except json.JSONDecodeError:
                        continue

        if buffer:
            # Store the final answer in history
            history.append((question, buffer))

    async def _deepseek_ask(self, messages: List[Dict], model: str, params: Dict) -> str:
        """Asks the DeepSeek API for a non-streaming response."""
        headers = {
            "Authorization": f"Bearer {config.ai.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": messages,
            **params
        }
        response = await client.post(
            f"{config.ai.url}/chat/completions",
            headers=headers,
            json=data
        )
        resp = response.json()
        if "usage" not in resp:
            raise Exception(resp)
        logger.debug(
            "< chat response: prompt_tokens=%s, completion_tokens=%s, total_tokens=%s",
            resp["usage"]["prompt_tokens"],
            resp["usage"]["completion_tokens"],
            resp["usage"]["total_tokens"],
        )
        answer = self._prepare_answer(resp)
        return answer

    def _generate_messages(
        self, prompt_role: str, prompt: str, question: str, history: List[tuple[str, str]]
    ) -> List[Dict]:
        """Builds message history to provide context for the language model."""
        messages = [{"role": prompt_role, "content": prompt or config.ai.prompt}]
        for prev_question, prev_answer in history:
            messages.append({"role": "user", "content": prev_question})
            messages.append({"role": "assistant", "content": prev_answer})
        messages.append({"role": "user", "content": question})
        return messages

    def _prepare_answer(self, resp) -> str:
        """Post-processes an answer from the language model."""
        if len(resp["choices"]) == 0:
            raise ValueError("received an empty answer")

        answer = resp["choices"][0]["message"]["content"]
        answer = answer.strip()
        return answer


def shorten(messages: List[Dict], length: int) -> List[Dict]:
    """
    Truncates messages so that the total number of tokens
    does not exceed the specified length.
    """
    lengths = [_calc_tokens(m["content"]) for m in messages]
    total_len = sum(lengths)
    if total_len <= length:
        return messages

    # exclude older messages to fit into the desired length
    # can't exclude the prompt though
    prompt_msg, messages = messages[0], messages[1:]
    prompt_len, lengths = lengths[0], lengths[1:]
    while len(messages) > 1 and total_len > length:
        messages = messages[1:]
        first_len, lengths = lengths[0], lengths[1:]
        total_len -= first_len
    messages = [prompt_msg] + messages
    if total_len <= length:
        return messages

    # there is only one message left, and it's still longer than allowed
    # so we have to shorten it
    maxlen = length - prompt_len
    tokens = messages[1]["content"].split()[:maxlen]
    messages[1]["content"] = " ".join(tokens)
    return messages


def _calc_tokens(s: str) -> int:
    """Calculates the number of tokens in a string."""
    return int(len(s.split()) * 1.2)


def _calc_n_input(name: str, n_output: int) -> int:
    """
    Calculates the maximum number of input tokens
    according to the model and the maximum number of output tokens.
    """
    # AI providers count length in tokens, not characters.
    # We need to leave some tokens reserved for the output.
    n_total = MODELS.get(name) or config.ai.window
    logger.debug("model=%s, n_total=%s, n_output=%s", name, n_total, n_output)
    return n_total - n_output
