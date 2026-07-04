import hashlib
import json
import re

import httpx
import numpy as np

import config


# -------------------------------
# Request Headers
# -------------------------------

HEADERS = {
    "Authorization": f"Bearer {config.AIPIPE_TOKEN}",
    "Content-Type": "application/json",
}


# -------------------------------
# Small in-memory cache
# -------------------------------

_CACHE = {}


def make_cache_key(*parts):
    """
    Create deterministic cache keys.
    """
    raw = "||".join(map(str, parts))
    return hashlib.sha256(raw.encode()).hexdigest()


# -------------------------------
# Safe JSON parser
# -------------------------------

def parse_json(text):
    """
    GPT sometimes wraps JSON inside ```json ...```
    Remove the fences and return a dict.
    """

    if not text:
        return {}

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)

    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return {}

    return {}


# -------------------------------
# Chat Completion
# -------------------------------

async def chat(
    messages,
    model=None,
    max_tokens=1000,
    force_json=True,
):
    """
    Generic helper for GPT models.
    """

    cache_key = make_cache_key(
        "chat",
        model,
        json.dumps(messages, sort_keys=True, default=str),
    )

    if cache_key in _CACHE:
        return _CACHE[cache_key]

    body = {
        "model": model or config.TEXT_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    if force_json:
        body["response_format"] = {
            "type": "json_object"
        }

    async with httpx.AsyncClient(
        timeout=config.REQUEST_TIMEOUT
    ) as client:

        response = await client.post(
            f"{config.AIPIPE_BASE}/chat/completions",
            headers=HEADERS,
            json=body,
        )

        response.raise_for_status()

        output = response.json()["choices"][0]["message"]["content"]

    _CACHE[cache_key] = output

    return output


# -------------------------------
# Embedding API
# -------------------------------

async def embeddings(texts):
    """
    Returns embedding vectors.
    """

    async with httpx.AsyncClient(
        timeout=config.REQUEST_TIMEOUT
    ) as client:

        response = await client.post(
            f"{config.AIPIPE_BASE}/embeddings",
            headers=HEADERS,
            json={
                "model": config.EMBED_MODEL,
                "input": texts,
            },
        )

        response.raise_for_status()

        data = response.json()["data"]

    return [item["embedding"] for item in data]


# -------------------------------
# Cosine Similarity
# -------------------------------

def cosine_similarity(a, b):

    a = np.array(a)
    b = np.array(b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0

    return float(np.dot(a, b) / denominator)