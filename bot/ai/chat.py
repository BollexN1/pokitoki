import openai
import requests
import json
from typing import List, Dict, Iterator
from bot.config import Config
from bot.models import ChatModel

class Chat:
    def __init__(self, config: Config):
        self.config = config
        self.client = openai.OpenAI(
            api_key=config.ai_api_key,
            base_url=config.ai_url
        )
        self.history: Dict[str, List[Dict]] = {}

    def ask(self, chat_id: str, prompt: str, model: ChatModel) -> str:
        messages = self._get_history(chat_id) + [{"role": "user", "content": self.config.prompt + prompt}]
        if self.config.ai_provider == "deepseek":
            return self._deepseek_chat(messages, model.name)
        else:
            response = self.client.chat.completions.create(
                model=model.name,
                messages=messages,
                **self.config.ai_params
            )
            answer = response.choices[0].message.content
            self._update_history(chat_id, prompt, answer)
            return answer

    def chat_completion(self, chat_id: str, prompt: str, model: ChatModel, thinking_enabled: bool = False, search_enabled: bool = False) -> Iterator[Dict]:
        messages = self._get_history(chat_id) + [{"role": "user", "content": self.config.prompt + prompt}]
        if self.config.ai_provider != "deepseek":
            yield {"type": "text", "content": "Streaming is only supported with DeepSeek provider."}
            return

        headers = {
            "Authorization": f"Bearer {self.config.ai_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model.name,
            "messages": messages,
            "max_tokens": self.config.ai_params.get("max_tokens", 500),
            "temperature": self.config.ai_params.get("temperature", 0.7),
            "frequency_penalty": self.config.ai_params.get("frequency_penalty", 0),
            "presence_penalty": self.config.ai_params.get("presence_penalty", 0),
            "stream": True,
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled
        }
        if thinking_enabled and search_enabled:
            data["model"] = "deepseek-r1"  # Для thinking + search используем deepseek-r1

        response = requests.post(
            f"{self.config.ai_url}/chat/completions",
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
                        chunk_type = chunk_json.get("type", "text")
                        content = chunk_json.get("content", "")
                        if chunk_type in ["thinking", "text"]:
                            yield {"type": chunk_type, "content": content}
                            if chunk_type == "text":
                                buffer += content
                    except json.JSONDecodeError:
                        continue

        if buffer:
            self._update_history(chat_id, prompt, buffer)

    def search(self, chat_id: str, query: str, model: ChatModel) -> Iterator[Dict]:
        for chunk in self.chat_completion(chat_id, query, model, thinking_enabled=True, search_enabled=True):
            yield chunk

    def thinking(self, chat_id: str, query: str, model: ChatModel) -> Iterator[Dict]:
        for chunk in self.chat_completion(chat_id, query, model, thinking_enabled=True, search_enabled=False):
            yield chunk

    def _deepseek_chat(self, messages: List[Dict], model: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.config.ai_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": self.config.ai_params.get("max_tokens", 500),
            "temperature": self.config.ai_params.get("temperature", 0.7),
            "frequency_penalty": self.config.ai_params.get("frequency_penalty", 0),
            "presence_penalty": self.config.ai_params.get("presence_penalty", 0)
        }
        response = requests.post(f"{self.config.ai_url}/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _get_history(self, chat_id: str) -> List[Dict]:
        if chat_id not in self.history:
            self.history[chat_id] = []
        return self.history[chat_id][-self.config.history_depth:]

    def _update_history(self, chat_id: str, prompt: str, answer: str):
        if chat_id not in self.history:
            self.history[chat_id] = []
        self.history[chat_id].append({"role": "user", "content": prompt})
        self.history[chat_id].append({"role": "assistant", "content": answer})
