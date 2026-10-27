"""Thin LLM client with native tool calling. Pick a provider with env vars (see .env.example).

  LLM_PROVIDER=gemini     needs GEMINI_API_KEY (Google's OpenAI-compatible endpoint)
  LLM_PROVIDER=anthropic  needs ANTHROPIC_API_KEY
  LLM_PROVIDER=openai     needs OPENAI_API_KEY
  LLM_PROVIDER=ollama     free, local (https://ollama.com); use a tool-capable model (qwen2.5, llama3.1)

Plain HTTP via urllib - no SDKs, so no compiled dependencies.

The agent keeps a provider-neutral history:
  {"role": "user", "content": str}
  {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "args", "raw"}]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}
and this module converts it to each provider's wire format.
"""
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

# minimal .env loader (no extra dependency)
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.split("#", 1)[0].strip()
        if "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

DEFAULT_MODELS = {"ollama": "qwen2.5:7b", "anthropic": "claude-sonnet-4-5", "openai": "gpt-4o-mini",
                  "gemini": "gemini-2.5-flash"}
OPENAI_COMPAT = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "ollama": (os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/v1", None),
}


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
        if self.provider not in DEFAULT_MODELS:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")
        self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS[self.provider]
        self.min_interval = float(os.getenv("LLM_MIN_INTERVAL", "0"))  # seconds between calls (free-tier RPM)
        self.calls = 0  # simple counter you can report for cost/latency
        self.input_tokens = self.output_tokens = 0
        self._last = 0.0

    # ------------------------------------------------------------------ public
    def chat_tools(self, messages: list[dict], tools: list[dict], system: str,
                   max_tokens: int = 4096, temperature: float = 0.0) -> dict:
        """One model turn. tools: [{"name", "description", "parameters": JSON schema}].
        Returns {"content": str, "tool_calls": [{"id", "name", "args", "raw"}]}."""
        if self.provider == "anthropic":
            return self._anthropic(messages, tools, system, max_tokens, temperature)
        return self._openai_compat(messages, tools, system, max_tokens, temperature)

    def chat(self, messages: list[dict], system: str | None = None, max_tokens: int = 1024,
             temperature: float = 0.0) -> str:
        return self.chat_tools(messages, [], system or "", max_tokens, temperature)["content"]

    # --------------------------------------------------------------- transport
    def _post(self, url: str, body: dict, headers: dict) -> dict:
        """POST with pacing and retry on rate limits / transient server errors."""
        for attempt in range(6):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            self.calls += 1
            req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", **headers})
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                if e.code in (429, 500, 502, 503, 504, 529) and attempt < 5:
                    time.sleep(min(60, 2 ** attempt * 3) + random.random())
                    continue
                raise LLMError(f"HTTP {e.code} from {self.provider}: {detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < 5:
                    time.sleep(2 ** attempt + random.random())
                    continue
                raise LLMError(f"Network error calling {self.provider}: {e}") from e
        raise LLMError("retries exhausted")

    # ---------------------------------------------------- OpenAI-compatible API
    def _openai_compat(self, messages, tools, system, max_tokens, temperature):
        base, key_var = OPENAI_COMPAT[self.provider]
        headers = {}
        if key_var:
            key = os.getenv(key_var)
            if not key:
                raise LLMError(f"{key_var} is not set (add it to .env)")
            headers["Authorization"] = f"Bearer {key}"
        msgs = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "assistant":
                out = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    out["tool_calls"] = [tc["raw"] for tc in m["tool_calls"]]
                msgs.append(out)
            elif m["role"] == "tool":
                msgs.append({"role": "tool", "tool_call_id": m["tool_call_id"], "name": m["name"],
                             "content": m["content"]})
            else:
                msgs.append({"role": "user", "content": m["content"]})
        body = {"model": self.model, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature}
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
            body["tool_choice"] = "auto"
        data = self._post(f"{base}/chat/completions", body, headers)
        usage = data.get("usage") or {}
        self.input_tokens += usage.get("prompt_tokens", 0)
        self.output_tokens += usage.get("completion_tokens", 0)
        if not data.get("choices"):
            raise LLMError(f"No choices in response: {str(data)[:300]}")
        msg = data["choices"][0]["message"]
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            raw = dict(tc)  # echoed back verbatim (keeps provider extras such as Gemini thought signatures)
            raw.setdefault("id", f"call_{self.calls}_{i}")
            raw.setdefault("type", "function")
            calls.append({"id": raw["id"], "name": tc["function"]["name"],
                          "args": _parse_args(tc["function"].get("arguments")), "raw": raw})
        return {"content": msg.get("content") or "", "tool_calls": calls}

    # ----------------------------------------------------------- Anthropic API
    def _anthropic(self, messages, tools, system, max_tokens, temperature):
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set (add it to .env)")
        msgs = []
        for m in messages:
            if m["role"] == "assistant":
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                blocks += [{"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["args"]}
                           for tc in m.get("tool_calls") or []]
                msgs.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if msgs and msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], list):
                    msgs[-1]["content"].append(block)  # parallel tool results share one user turn
                else:
                    msgs.append({"role": "user", "content": [block]})
            else:
                msgs.append({"role": "user", "content": m["content"]})
        body = {"model": self.model, "system": system, "messages": msgs, "max_tokens": max_tokens,
                "temperature": temperature}
        if tools:
            body["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                             for t in tools]
        data = self._post("https://api.anthropic.com/v1/messages", body,
                          {"x-api-key": key, "anthropic-version": "2023-06-01"})
        usage = data.get("usage") or {}
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        text = "".join(b["text"] for b in data["content"] if b["type"] == "text")
        calls = [{"id": b["id"], "name": b["name"], "args": b["input"], "raw": None}
                 for b in data["content"] if b["type"] == "tool_use"]
        return {"content": text, "tool_calls": calls}


def _parse_args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        v = json.loads(raw or "{}")
        return v if isinstance(v, dict) else {"_invalid": raw}
    except json.JSONDecodeError:
        return {"_invalid": raw}
