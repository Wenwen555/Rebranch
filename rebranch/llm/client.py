"""LLM client implementations (Codex CLI, DiskCache, Mock, OpenAI-compatible)."""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Sequence
import urllib.request


@dataclass
class LLMResponse:
    """Normalized LLM completion result and accounting metadata."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    wall_s: float = 0.0


class LLMClient(ABC):
    """Abstract text completion interface with usage statistics."""

    @abstractmethod
    def complete(
        self, prompt: str, *, system: str = "", temperature: float = 0.2,
        images: Sequence[bytes] = (),
    ) -> LLMResponse:
        """Complete `prompt` under optional `system` instruction."""

    @property
    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return cumulative calls, tokens, cache hits, and wall seconds."""


class LLMBudgetExceededError(RuntimeError):
    """A proposal/request limit, separate from simulation rollout exhaustion."""


class LimitedLLMClient(LLMClient):
    """Per-development hard cap, including cache lookups and retry attempts."""
    def __init__(self, inner: LLMClient, limit: int):
        self.inner = inner
        self.limit = max(0, int(limit))
        self.attempts = 0
        self._budget_lock = threading.Lock()

    @property
    def stats(self):
        return self.inner.stats

    def complete(self, prompt, *, system="", temperature=0.2, images=()):
        with self._budget_lock:
            if self.attempts >= self.limit:
                raise LLMBudgetExceededError(f"LLM proposal budget {self.limit} exhausted")
            self.attempts += 1
        if images:
            return self.inner.complete(
                prompt, system=system, temperature=temperature, images=images
            )
        return self.inner.complete(prompt, system=system, temperature=temperature)


def _toml_inline(value: Any) -> str:
    if isinstance(value, dict):
        items = ",".join(
            f"{json.dumps(str(k))}={_toml_inline(v)}" for k, v in value.items()
        )
        return "{" + items + "}"
    return json.dumps(value)


class CodexCLIClient(LLMClient):
    """Subprocess wrapper around `codex exec` with strict read isolation."""

    def __init__(
        self,
        model: str = "gpt-6.1-sol",
        effort: str = "low",
        timeout_s: int = 180,
        codex_bin: str = "/home/wen/.local/bin/codex",
    ) -> None:
        self.model = model
        self.effort = effort
        self.timeout_s = int(timeout_s)
        self.codex_bin = str(codex_bin)
        self._calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._wall_s = 0.0
        self._lock = threading.Lock()

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self._calls,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "cached_hits": 0,
                "wall_s": self._wall_s,
            }

    def _permissions_public(self) -> dict[str, Any]:
        codex_path = Path(self.codex_bin)
        codex_real = str(codex_path.resolve(strict=True))
        codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
        filesystem: dict[str, Any] = {
            ":root": "deny",
            ":minimal": "read",
            ":tmpdir": "deny",
            ":slash_tmp": "deny",
            ":workspace_roots": {".": "read"},
            str(codex_path): "read",
            codex_real: "read",
        }
        if codex_home.exists():
            filesystem[str(codex_home)] = "read"
        return {"filesystem": filesystem, "network": {"enabled": False}}

    def complete(
        self, prompt: str, *, system: str = "", temperature: float = 0.2,
        images: Sequence[bytes] = (),
    ) -> LLMResponse:
        _ = temperature
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        t0 = time.perf_counter()

        with tempfile.TemporaryDirectory(prefix="rebranch_codex_") as td:
            td_path = Path(td)
            work_dir = td_path / "empty_work"
            work_dir.mkdir()
            out_txt = td_path / "out.txt"

            cmd = [
                self.codex_bin,
                "--no-daemon",
                "exec",
                "--ignore-user-config",
                "--ignore-rules",
                "--ephemeral",
                "--skip-git-repo-check",
                "-s",
                "read-only",
                "-m",
                self.model,
                "--cd",
                str(work_dir),
                "-c",
                'default_permissions="public"',
                "-c",
                f"permissions.public={_toml_inline(self._permissions_public())}",
                "-c",
                "suppress_unstable_features_warning=true",
                "-c",
                f"model_reasoning_effort={json.dumps(self.effort)}",
                "-c",
                'web_search="disabled"',
                "-c",
                "mcp_servers={}",
                "-c",
                "plugins._default.enabled=false",
                "-c",
                "apps._default.enabled=false",
                "-c",
                "agents.enabled=false",
                "-c",
                "skills.bundled.enabled=false",
                "-c",
                "features.skip_host_skill_discovery=true",
                "-c",
                'history.persistence="none"',
                "-c",
                "project_doc_max_bytes=0",
                "--json",
                "-o",
                str(out_txt),
                "-",
            ]
            image_paths: list[str] = []
            for index, image in enumerate(images):
                image_path = work_dir / f"image_{index}.png"
                image_path.write_bytes(bytes(image))
                image_paths.append(str(image_path))
            if image_paths:
                cmd[-1:] = ["-i", *image_paths, "--", "-"]

            proc = subprocess.run(
                cmd,
                input=full_prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout_s,
                cwd=str(work_dir),
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Codex CLI failed (rc={proc.returncode}): "
                    f"{proc.stderr.strip() or proc.stdout.strip()}"
                )

            prompt_tokens = 0
            completion_tokens = 0
            fallback_text = ""
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ev_type = ev.get("type")
                if ev_type == "turn.completed":
                    usage = ev.get("usage") or {}
                    prompt_tokens += int(usage.get("input_tokens", 0))
                    completion_tokens += int(usage.get("output_tokens", 0))
                elif ev_type == "item.completed":
                    item = ev.get("item") or {}
                    if item.get("type") == "agent_message" and item.get("text"):
                        fallback_text = str(item["text"])

            text = (
                out_txt.read_text(encoding="utf-8").strip()
                if out_txt.exists()
                else fallback_text.strip()
            )
            if not text:
                text = fallback_text.strip()

        wall_s = time.perf_counter() - t0
        with self._lock:
            self._calls += 1
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._wall_s += wall_s
        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached=False,
            wall_s=wall_s,
        )


class DiskCacheLLM(LLMClient):
    """SHA256 cache; tokens count logical usage, inner stats count fresh requests."""

    def __init__(
        self,
        inner: LLMClient,
        cache_dir: str | Path = ".cache/rebranch_llm",
    ) -> None:
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._calls = 0
        self._cached_hits = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._wall_s = 0.0
        self._lock = threading.Lock()

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self._calls,
                "cached_hits": self._cached_hits,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "wall_s": self._wall_s,
                "inner": self.inner.stats,
            }

    def _cache_key(
        self, prompt: str, system: str, temperature: float, images: Sequence[bytes] = ()
    ) -> str:
        payload = {
            "inner": type(self.inner).__name__,
            "model": getattr(self.inner, "model", ""),
            "effort": getattr(self.inner, "effort", ""),
            "system": system,
            "prompt": prompt,
            "temperature": round(float(temperature), 6),
        }
        if images:
            payload["images_sha256"] = [
                hashlib.sha256(bytes(image)).hexdigest() for image in images
            ]
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def complete(
        self, prompt: str, *, system: str = "", temperature: float = 0.2,
        images: Sequence[bytes] = (),
    ) -> LLMResponse:
        key = self._cache_key(prompt, system, temperature, images)
        path = self.cache_dir / f"{key}.json"

        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            with self._lock:
                self._calls += 1
                self._cached_hits += 1
                self._prompt_tokens += int(data.get("prompt_tokens", 0))
                self._completion_tokens += int(data.get("completion_tokens", 0))
            return LLMResponse(
                text=str(data["text"]),
                prompt_tokens=int(data.get("prompt_tokens", 0)),
                completion_tokens=int(data.get("completion_tokens", 0)),
                cached=True,
                wall_s=0.0,
            )

        if images:
            resp = self.inner.complete(
                prompt, system=system, temperature=temperature, images=images
            )
        else:
            resp = self.inner.complete(prompt, system=system, temperature=temperature)
        tmp_path = self.cache_dir / f".{key}.{os.getpid()}.{threading.get_ident()}.tmp"
        tmp_path.write_text(
            json.dumps(
                {
                    "text": resp.text,
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                    "wall_s": resp.wall_s,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)

        with self._lock:
            self._calls += 1
            self._prompt_tokens += resp.prompt_tokens
            self._completion_tokens += resp.completion_tokens
            self._wall_s += resp.wall_s
        return resp


class MockLLMClient(LLMClient):
    """Deterministic mock LLM client for unit tests and M0 offline runs."""

    def __init__(
        self,
        responder: Callable[..., str] | dict[str, str] | Sequence[str],
    ) -> None:
        self.responder = responder
        self._calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self.history: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self._calls,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "cached_hits": 0,
                "wall_s": 0.0,
            }

    def complete(
        self, prompt: str, *, system: str = "", temperature: float = 0.2,
        images: Sequence[bytes] = (),
    ) -> LLMResponse:
        _ = temperature
        with self._lock:
            call_idx = self._calls
        if callable(self.responder):
            sig = inspect.signature(self.responder)
            if len(sig.parameters) >= 2:
                text = str(self.responder(prompt, system))
            else:
                text = str(self.responder(prompt))
        elif isinstance(self.responder, dict):
            text = ""
            for key, val in self.responder.items():
                if key in prompt:
                    text = str(val)
                    break
            if not text and "__default__" in self.responder:
                text = str(self.responder["__default__"])
            if not text:
                raise KeyError("MockLLMClient dict responder matched no prompt key")
        else:
            seq = list(self.responder)
            if not seq:
                raise IndexError("MockLLMClient sequence responder is empty")
            text = str(seq[min(call_idx, len(seq) - 1)])

        pt = max(1, len(prompt) // 4)
        ct = max(1, len(text) // 4)
        with self._lock:
            self.history.append({
                "system": system,
                "prompt": prompt,
                "images": tuple(bytes(image) for image in images),
                "image_sha256": tuple(hashlib.sha256(bytes(image)).hexdigest() for image in images),
            })
            self._calls += 1
            self._prompt_tokens += pt
            self._completion_tokens += ct
        return LLMResponse(
            text=text,
            prompt_tokens=pt,
            completion_tokens=ct,
            cached=False,
            wall_s=0.0,
        )


class OpenAICompatibleClient(LLMClient):
    """Minimal OpenAI-compatible HTTP client configured strictly via env vars."""

    def __init__(
        self,
        model: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
        timeout_s: int = 120,
    ) -> None:
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key_env = api_key_env
        self.base_url_env = base_url_env
        self.timeout_s = int(timeout_s)
        self._calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._wall_s = 0.0
        self._lock = threading.Lock()

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self._calls,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "cached_hits": 0,
                "wall_s": self._wall_s,
            }

    def complete(
        self, prompt: str, *, system: str = "", temperature: float = 0.2,
        images: Sequence[bytes] = (),
    ) -> LLMResponse:
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"Environment variable {self.api_key_env!r} is not set"
            )
        base_url = os.environ.get(
            self.base_url_env, "https://api.openai.com/v1"
        ).rstrip("/")

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if images:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for image in images:
                encoded = base64.b64encode(bytes(image)).decode("ascii")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": float(temperature),
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url=f"{base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )

        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        wall_s = time.perf_counter() - t0

        text = str(data["choices"][0]["message"]["content"])
        usage = data.get("usage") or {}
        pt = int(usage.get("prompt_tokens", 0))
        ct = int(usage.get("completion_tokens", 0))

        with self._lock:
            self._calls += 1
            self._prompt_tokens += pt
            self._completion_tokens += ct
            self._wall_s += wall_s
        return LLMResponse(
            text=text,
            prompt_tokens=pt,
            completion_tokens=ct,
            cached=False,
            wall_s=wall_s,
        )


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(text: str) -> str:
    """Extract Python policy source from fenced markdown blocks or raw text."""
    matches = [m.group(1).strip() for m in _CODE_FENCE_RE.finditer(text)]
    if matches:
        for block in matches:
            if "def run(" in block:
                return block + "\n"
        return max(matches, key=len) + "\n"
    return text.strip() + "\n"
