"""Bot configuration parameters."""

import os
from typing import Any, Optional, List, Dict
import yaml
import dataclasses
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Telegram:
    token: str
    usernames: List[str]
    admins: List[str]
    chat_ids: List[str]

@dataclass
class AIProvider:
    provider: str
    url: str
    api_key: str
    model: str
    image_model: str
    prompt: str
    window: int
    params: Dict

    default_urls = {
        "openai": "https://api.openai.com/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta",
        "deepseek": "https://api.deepseek.com/v1"
    }
    default_models = {
        "openai": "gpt-3.5-turbo",
        "gemini": "gemini-pro",
        "deepseek": "deepseek-chat"
    }
    default_params = {
        "temperature": 0.7,
        "max_tokens": 1000,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }

    def __init__(
        self,
        provider: str,
        url: str,
        api_key: str,
        model: str,
        image_model: str,
        prompt: str,
        window: int,
        params: Dict,
    ) -> None:
        self.provider = provider
        self.url = url or self.default_urls.get(provider, "https://api.openai.com/v1")
        self.api_key = api_key
        self.model = model or self.default_models.get(provider, "gpt-3.5-turbo")
        self.image_model = image_model or "dall-e-3"
        self.prompt = prompt or ""
        self.window = window or 128000
        self.params = self.default_params.copy()
        self.params.update(params)

@dataclass
class RateLimit:
    count: int
    period: str

    allowed_periods = ("minute", "hour", "day")
    default_period = "hour"

    def __init__(self, count: int = 0, period: str = default_period) -> None:
        self.count = count
        if period not in self.allowed_periods:
            period = self.default_period
        self.period = period

    def __bool__(self) -> bool:
        return self.count > 0

@dataclass
class Conversation:
    depth: int
    message_limit: RateLimit

    default_depth = 3

    def __init__(self, depth: int, message_limit: Dict) -> None:
        self.depth = depth or self.default_depth
        self.message_limit = RateLimit(**message_limit)

@dataclass
class Imagine:
    enabled: str

    def __init__(self, enabled: str) -> None:
        self.enabled = (
            enabled if enabled in ("none", "users_only", "users_and_groups", "disabled") else "disabled"
        )

class Config:
    """Config properties."""

    # Config schema version. Increments for backward-incompatible changes.
    schema_version = 4
    # Bot version.
    version = 227

    def __init__(self, filename: str, src: Dict) -> None:
        # Config filename.
        self.filename = filename

        # Telegram settings.
        self.telegram = Telegram(
            token=src["telegram"]["token"],
            usernames=src["telegram"].get("usernames") or [],
            admins=src["telegram"].get("admins") or [],
            chat_ids=src["telegram"].get("chat_ids") or [],
        )

        # AI Provider settings.
        ai_data = src.get("ai", {})
        self.ai_provider = AIProvider(
            provider=ai_data.get("provider", "openai"),
            url=ai_data.get("url"),
            api_key=ai_data.get("api_key", ""),
            model=ai_data.get("model"),
            image_model=ai_data.get("image_model"),
            prompt=ai_data.get("prompt"),
            window=ai_data.get("window"),
            params=ai_data.get("params", {}),
        )

        # Conversation settings.
        self.conversation = Conversation(
            depth=src["conversation"].get("depth"),
            message_limit=src["conversation"].get("message_limit") or {},
        )

        # Image generation settings.
        self.imagine = Imagine(enabled=src["imagine"].get("enabled") or "disabled")

        # Where to store the chat context file.
        self.persistence_path = src.get("persistence_path") or "./data/persistence.pkl"

        # Custom AI commands (additional prompts).
        self.shortcuts = src.get("shortcuts") or {}

    def as_dict(self) -> Dict:
        """Converts the config into a dictionary."""
        return {
            "schema_version": self.schema_version,
            "telegram": dataclasses.asdict(self.telegram),
            "ai": dataclasses.asdict(self.ai_provider),
            "conversation": dataclasses.asdict(self.conversation),
            "imagine": dataclasses.asdict(self.imagine),
            "persistence_path": self.persistence_path,
            "shortcuts": self.shortcuts,
        }

class ConfigEditor:
    """
    Config properties editor.
    Gets/sets config properties by their 'path',
    e.g. 'ai.params.temperature' or 'conversation.depth'.
    """

    # These properties cannot be changed at all.
    readonly = [
        "schema_version",
        "version",
        "filename",
    ]
    # Changes made to these properties take effect immediately.
    immediate = [
        "telegram",
        "ai",
        "conversation",
        "imagine",
        "shortcuts",
    ]
    # Changes made to these properties take effect after a restart.
    delayed = [
        "telegram.token",
        "persistence_path",
    ]
    # All editable properties.
    editable = immediate + delayed
    # All known properties.
    known = readonly + immediate + delayed

    def __init__(self, config: Config) -> None:
        self.config = config

    def get_value(self, property: str) -> Any:
        """Returns a config property value."""
        names = property.split(".")
        if names[0] not in self.known:
            raise ValueError(f"No such property: {property}")

        obj = self.config
        for name in names[:-1]:
            if not hasattr(obj, name):
                raise ValueError(f"No such property: {property}")
            obj = getattr(obj, name)

        name = names[-1]
        if isinstance(obj, dict):
            return obj.get(name)

        if isinstance(obj, object):
            if not hasattr(obj, name):
                raise ValueError(f"No such property: {property}")
            val = getattr(obj, name)
            if dataclasses.is_dataclass(val):
                return dataclasses.asdict(val)
            return val

        raise ValueError(f"Failed to get property: {property}")

    def set_value(self, property: str, value: str) -> tuple[bool, bool, Any]:
        """
        Changes a config property value.
        Returns a tuple `(has_changed, is_immediate, new_val)`
          - `has_changed`  = True if the value has actually changed, False otherwise.
          - `is_immediate` = True if the change takes effect immediately, False otherwise.
          - `new_val`        is the new value
        """
        try:
            val = yaml.safe_load(value)
        except Exception:
            raise ValueError(f"Invalid value: {value}")

        old_val = self.get_value(property)
        if val == old_val:
            return False, False, old_val

        if isinstance(old_val, list) and isinstance(val, str):
            # allow changing list properties by adding or removing individual items
            # e.g. /config telegram.usernames +bob
            # or   /config telegram.usernames -alice
            if val[0] == "+":
                item = yaml.safe_load(val[1:])
                val = old_val.copy()
                val.append(item)
            elif val[0] == "-":
                item = yaml.safe_load(val[1:])
                val = old_val.copy()
                val.remove(item)

        old_cls = old_val.__class__
        val_cls = val.__class__
        if old_val is not None and old_cls != val_cls:
            raise ValueError(
                f"Property {property} should be of type {old_cls.__name__}, not {val_cls.__name__}"
            )

        if not isinstance(val, (list, str, int, float, bool)):
            raise ValueError(f"Cannot set composite value for property: {property}")

        names = property.split(".")
        if names[0] not in self.editable:
            raise ValueError(f"Property {property} is not editable")

        is_immediate = property not in self.delayed

        obj = self.config
        for name in names[:-1]:
            obj = getattr(obj, name, val)

        name = names[-1]
        if isinstance(obj, dict):
            obj[name] = val
            return True, is_immediate, val

        if isinstance(obj, object):
            if not hasattr(obj, name):
                raise ValueError(f"No such property: {property}")
            setattr(obj, name, val)
            return True, is_immediate, val

        raise ValueError(f"Failed to set property: {property}")

    def save(self) -> None:
        """Saves the config to disk."""
        data = self.config.as_dict()
        with open(self.config.filename, "w") as file:
            yaml.safe_dump(data, file, indent=4, allow_unicode=True)

class SchemaMigrator:
    """Migrates the configuration data dictionary according to the schema version."""

    @classmethod
    def migrate(cls, data: Dict) -> tuple[Dict, bool]:
        """Migrates the configuration to the latest schema version."""
        has_changed = False
        if data.get("schema_version", 1) == 1:
            data = cls._migrate_v1(data)
            has_changed = True
        if data["schema_version"] == 2:
            data = cls._migrate_v2(data)
            has_changed = True
        if data["schema_version"] == 3:
            data = cls._migrate_v3(data)
            has_changed = True
        return data, has_changed

    @classmethod
    def _migrate_v1(cls, old: Dict) -> Dict:
        data = {
            "schema_version": 2,
            "telegram": None,
            "ai": None,
            "max_history_depth": old.get("max_history_depth"),
            "imagine": old.get("imagine"),
            "persistence_path": old.get("persistence_path"),
            "shortcuts": old.get("shortcuts"),
        }
        data["telegram"] = {
            "token": old["telegram_token"],
            "usernames": old.get("telegram_usernames"),
            "chat_ids": old.get("telegram_chat_ids"),
        }
        data["ai"] = {
            "provider": old.get("ai_provider", "openai"),
            "api_key": old["ai_api_key"],
            "model": old.get("ai_model"),
        }
        return data

    @classmethod
    def _migrate_v2(cls, old: Dict) -> Dict:
        data = {
            "schema_version": 3,
            "telegram": old["telegram"],
            "ai": old["ai"],
            "imagine": old.get("imagine"),
            "persistence_path": old.get("persistence_path"),
            "shortcuts": old.get("shortcuts"),
        }
        data["conversation"] = {"depth": old.get("max_history_depth") or Conversation.default_depth}
        return data

    @classmethod
    def _migrate_v3(cls, old: Dict) -> Dict:
        data = {
            "schema_version": 4,
            "telegram": old["telegram"],
            "ai": old["ai"],
            "conversation": old["conversation"],
            "persistence_path": old.get("persistence_path"),
            "shortcuts": old.get("shortcuts"),
        }
        imagine_enabled = old.get("imagine")
        imagine_enabled = True if imagine_enabled is None else imagine_enabled
        data["imagine"] = {"enabled": "users_only" if imagine_enabled else "none"}
        return data

def load(filename: str = "config.yml") -> Dict:
    """Loads the configuration data dictionary from a file."""
    with open(filename, "r") as f:
        data = yaml.safe_load(f)

    data, has_changed = SchemaMigrator.migrate(data)
    if has_changed:
        with open(filename, "w") as f:
            yaml.safe_dump(data, f, indent=4, allow_unicode=True)
    return data

def load_config(config_path: str = "config.yml") -> Config:
    data = load(config_path)
    ai_data = data.get("ai", {})
    return Config(
        filename=config_path,
        src=data,
    )

def save_config(config: Config, config_path: str = "config.yml") -> None:
    data = {
        "conversation": {
            "depth": config.conversation.depth,
            "message_limit": {"count": 0, "period": "hour"}
        },
        "imagine": {"enabled": config.imagine.enabled},
        "ai": {
            "provider": config.ai_provider.provider,
            "api_key": config.ai_provider.api_key,
            "url": config.ai_provider.url,
            "model": config.ai_provider.model,
            "params": config.ai_provider.params,
            "image_model": config.ai_provider.image_model,
            "prompt": config.ai_provider.prompt,
            "window": config.ai_provider.window
        },
        "shortcuts": config.shortcuts,
        "telegram": {
            "token": config.telegram.token,
            "usernames": config.telegram.usernames,
            "chat_ids": config.telegram.chat_ids,
            "admins": config.telegram.admins
        },
        "persistence_path": config.persistence_path,
        "schema_version": 4
    }
    with open(config_path, "w") as file:
        yaml.safe_dump(data, file, sort_keys=False)

filename = os.getenv("CONFIG", "config.yml")
_config = load_config(filename)
config = Config(filename, _config)
