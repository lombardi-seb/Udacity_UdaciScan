"""Thin wrappers around the OpenAI-compatible client used by UdaciScan."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Sequence

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=".env")

DEFAULT_BASE_URL = "https://openai.vocareum.com/v1"


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Instantiate (and cache) the OpenAI client."""
    api_key = os.getenv("UDACITY_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set UDACITY_OPENAI_API_KEY or OPENAI_API_KEY in your environment/.env."
        )
    base_url = os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    return OpenAI(api_key=api_key, base_url=base_url)


def embed(
    texts: Sequence[str],
    *,
    model: str | None = None,
    batch_size: int = 32,
) -> List[List[float]]:
    """Create embeddings for one or more strings."""
    if not texts:
        return []
    client = get_client()
    model = model or os.getenv("UDACITY_EMBED_MODEL")
    if not model:
        from config import load_settings
        model = load_settings().embed_model

    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = list(texts[i : i + batch_size])
        resp = client.embeddings.create(model=model, input=chunk)
        for item in resp.data:
            vectors.append(item.embedding)
    return vectors


def chat_completion(
    messages: Sequence[Dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    response_format: Dict[str, Any] | None = None,
    max_output_tokens: int | None = None,
) -> str:
    """Call the chat completion endpoint with sensible defaults."""
    client = get_client()
    model = model or os.getenv("UDACITY_CHAT_MODEL")
    if not model:
        from config import load_settings
        model = load_settings().model
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
    }
    if response_format:
        kwargs["response_format"] = response_format
    if max_output_tokens is not None:
        kwargs["max_completion_tokens"] = max_output_tokens
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def try_extract_json(block: str) -> Any:
    """Attempt to parse the first JSON object/list present in a string."""
    block = block.strip()
    if not block:
        return None
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        pass
    start = block.find("{")
    end = block.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(block[start : end + 1])
        except json.JSONDecodeError:
            pass
    start = block.find("[")
    end = block.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(block[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None