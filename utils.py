"""
pageindex/utils.py — MODIFIED for local Ollama support
Replace the original pageindex/utils.py with this file.

What changed vs original:
  - llm_completion()    : replaced LiteLLM cloud call → Ollama local call
  - llm_acompletion()   : replaced LiteLLM async call → Ollama async call
  - count_tokens()      : replaced litellm.token_counter → tiktoken (no API needed)
  - Everything else     : UNCHANGED (extract_json, ConfigLoader, tree utils, etc.)
"""

import os
import re
import json
import copy
import yaml
import asyncio
import aiohttp
import requests
import tiktoken
from typing import Union
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# OLLAMA CONFIG  (reads from .env)
# ---------------------------------------------------------------------------
OLLAMA_API_BASE  = os.getenv("OLLAMA_API_BASE",  "http://localhost:11434")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL",     "mistral")   # e.g. mistral, llama2, phi3


# ---------------------------------------------------------------------------
# 1. SYNCHRONOUS LLM CALL  (replaces litellm.completion)
# ---------------------------------------------------------------------------
def llm_completion(
    model: str,
    prompt: str,
    chat_history: list = None,
    return_finish_reason: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    **kwargs,
) -> Union[str, tuple]:
    """
    Call a locally running Ollama model.
    Drop-in replacement for the original LiteLLM-based llm_completion().

    Args:
        model             : Ignored — model is read from OLLAMA_MODEL env var
        prompt            : The user prompt text
        chat_history      : List of previous {"role": ..., "content": ...} dicts
        return_finish_reason : If True returns (text, finish_reason)
        temperature       : 0.0 = deterministic
        max_tokens        : Max tokens to generate

    Returns:
        str   — generated text
        tuple — (generated text, finish_reason)  when return_finish_reason=True
    """
    messages = []

    # Include prior conversation turns if provided
    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": prompt})

    url     = f"{OLLAMA_API_BASE}/api/chat"
    payload = {
        "model":   OLLAMA_MODEL,
        "messages": messages,
        "stream":  False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    MAX_RETRIES = 10   # matches original retry count
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, timeout=600)
            resp.raise_for_status()
            data    = resp.json()
            content = data.get("message", {}).get("content", "")
            reason  = "stop" if data.get("done", True) else "length"

            if return_finish_reason:
                return content, reason
            return content

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"\n❌  Cannot reach Ollama at {OLLAMA_API_BASE}\n"
                f"    Run:  ollama serve\n"
                f"    Then: ollama list   (to confirm model '{OLLAMA_MODEL}' is present)\n"
            )
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Ollama call failed after {MAX_RETRIES} attempts: {exc}"
                )
            # brief back-off before retry
            import time; time.sleep(2 * attempt)


# ---------------------------------------------------------------------------
# 2. ASYNCHRONOUS LLM CALL  (replaces litellm.acompletion)
# ---------------------------------------------------------------------------
async def llm_acompletion(
    model: str,
    prompt: str,
    chat_history: list = None,
    return_finish_reason: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    **kwargs,
) -> Union[str, tuple]:
    """
    Async version of llm_completion for concurrent tasks (e.g. TOC verification).
    Drop-in replacement for the original LiteLLM-based llm_acompletion().
    """
    messages = []
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": prompt})

    url     = f"{OLLAMA_API_BASE}/api/chat"
    payload = {
        "model":    OLLAMA_MODEL,
        "messages": messages,
        "stream":   False,
        "options":  {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    MAX_RETRIES = 10
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=600)) as resp:
                    resp.raise_for_status()
                    data    = await resp.json()
                    content = data.get("message", {}).get("content", "")
                    reason  = "stop" if data.get("done", True) else "length"

                    if return_finish_reason:
                        return content, reason
                    return content

        except aiohttp.ClientConnectorError:
            raise ConnectionError(
                f"\n❌  Cannot reach Ollama at {OLLAMA_API_BASE}\n"
                f"    Run:  ollama serve\n"
            )
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Async Ollama call failed after {MAX_RETRIES} attempts: {exc}"
                )
            await asyncio.sleep(2 * attempt)


# ---------------------------------------------------------------------------
# 3. TOKEN COUNTER  (replaces litellm.token_counter — no API key needed)
# ---------------------------------------------------------------------------
def count_tokens(text: str, model: str = None) -> int:
    """
    Count tokens using tiktoken (OpenAI's tokenizer — works offline, no key).
    The actual token count may differ slightly from your Ollama model but is
    a good-enough budget estimate for splitting/truncation decisions.
    """
    try:
        enc = tiktoken.get_encoding("cl100k_base")   # GPT-4 encoding, widely compatible
        return len(enc.encode(text))
    except Exception:
        # Ultra-safe fallback: rough word-based estimate
        return len(text.split()) * 4 // 3


# ---------------------------------------------------------------------------
# EVERYTHING BELOW IS UNCHANGED FROM THE ORIGINAL utils.py
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    """
    Robust JSON extractor — cleans markdown fences, fixes trailing commas,
    and handles common LLM response formatting issues.
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = text.strip("`").strip()

    # Find the outermost { ... } or [ ... ]
    start = min(
        (text.find("{") if text.find("{") != -1 else len(text)),
        (text.find("[") if text.find("[") != -1 else len(text)),
    )
    if start == len(text):
        raise ValueError(f"No JSON object found in:\n{text[:300]}")

    end_brace  = text.rfind("}")
    end_bracket = text.rfind("]")
    end = max(end_brace, end_bracket) + 1
    text = text[start:end]

    # Fix trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    return json.loads(text)


def convert_page_to_int(toc_list: list) -> list:
    """Recursively convert page values to int where possible."""
    for item in toc_list:
        if "page" in item:
            try:
                item["page"] = int(item["page"])
            except (ValueError, TypeError):
                pass
        if "nodes" in item:
            convert_page_to_int(item["nodes"])
    return toc_list


def format_structure(structure: dict, order: list) -> dict:
    """Reorder keys in a structure dict to match the given order."""
    result = {}
    for key in order:
        if key in structure:
            result[key] = structure[key]
    for key in structure:
        if key not in result:
            result[key] = structure[key]
    return result


def flatten_tree(tree: list, result: list = None) -> list:
    """Flatten a nested tree into a list of all nodes."""
    if result is None:
        result = []
    for node in tree:
        result.append(node)
        if node.get("nodes"):
            flatten_tree(node["nodes"], result)
    return result


def get_page_content(page_list: list, start: int, end: int) -> str:
    """
    Return concatenated text for pages [start, end] (inclusive, 1-based index).
    """
    content = ""
    for page in page_list:
        if start <= page["physical_index"] <= end:
            content += page.get("content", "")
    return content


def add_node_text(structure: list, page_list: list) -> list:
    """Attach full page text to every node in the tree."""
    for node in structure:
        start = node.get("start_index")
        end   = node.get("end_index")
        if start is not None and end is not None:
            node["text"] = get_page_content(page_list, start, end)
        if node.get("nodes"):
            add_node_text(node["nodes"], page_list)
    return structure


def generate_summaries_for_structure(
    structure: list,
    page_list: list,
    opt: object,
    logger=None,
) -> list:
    """
    Walk every node and call llm_completion() to create a per-node summary.
    This function is called by page_index.py — no changes needed here because
    it calls llm_completion() which is already patched above.
    """
    for node in structure:
        start = node.get("start_index")
        end   = node.get("end_index")
        if start is None or end is None:
            continue

        page_text = get_page_content(page_list, start, end)
        if not page_text.strip():
            continue

        token_count = count_tokens(page_text)

        # Truncate to keep within model context window
        if token_count > opt.max_token_num_each_node:
            words = page_text.split()
            page_text = " ".join(
                words[:int(opt.max_token_num_each_node * 0.75)]
            )

        prompt = (
            f"Summarize the following section titled '{node.get('title', '')}' "
            f"in 2-4 concise sentences. Focus on key points only.\n\n"
            f"{page_text}"
        )

        try:
            node["summary"] = llm_completion(
                model=opt.model,
                prompt=prompt,
                temperature=0.0,
                max_tokens=300,
            )
        except Exception as exc:
            if logger:
                logger.warning(f"Summary failed for node '{node.get('title')}': {exc}")
            node["summary"] = node.get("title", "")

        if node.get("nodes"):
            generate_summaries_for_structure(node["nodes"], page_list, opt, logger)

    return structure


# ---------------------------------------------------------------------------
# ConfigLoader  — reads pageindex/config.yaml, merges with CLI args
# ---------------------------------------------------------------------------
class ConfigLoader:
    """
    Loads config.yaml and merges user-supplied options.
    The 'model' value in config.yaml is passed through to llm_completion()
    but IGNORED there (Ollama model is always read from OLLAMA_MODEL env var).
    """

    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

    def load(self, user_opt: dict) -> object:
        with open(self.CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)

        # Merge user overrides
        for key, value in user_opt.items():
            if value is not None:
                cfg[key] = value

        # Convert to simple namespace so callers can do opt.model, opt.max_token_num_each_node, etc.
        return _DictNamespace(cfg)


class _DictNamespace:
    """Simple dot-access wrapper for a dict."""

    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)
