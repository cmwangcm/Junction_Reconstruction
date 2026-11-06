"""Shared pipeline helpers and LLM provider clients (Anthropic / OpenAI).

Both AnthropicLLM and OpenAILLM expose a uniform agent interface so a tool-use
agent (e.g. llm_kg.llm_gen) can drive either provider without branching:

    create_message(...)         -> raw provider response
    assistant_message(resp)     -> assistant turn dict to append to `messages`
    message_text(resp)          -> the model's free-text/reasoning
    tool_calls(resp)            -> [{"id", "name", "input"}]  (normalized)
    wants_tools(resp)           -> bool
    tool_result(id, text, ...)  -> neutral result dict (shared, on BaseLLM)
    append_tool_results(msgs, results)  -> provider-correct appending
    image_user_message(img, text)       -> provider-correct image+text turn
    make_user_message(text)              -> provider-correct text turn

Tool definitions are passed in Anthropic-native shape ({name, description,
input_schema}) and translated for OpenAI internally, so callers and schema
modules stay provider-neutral.

Image delivery: AnthropicLLM.image_user_message prefers the Files API (uploads
the image once and references its file id). If the upload fails for any reason
it transparently falls back to an inline base64 image block. OpenAILLM has no
equivalent Files API and always embeds the image as a base64 data URL.

Cost tracking: CostTracker estimates USD from llm_pricing.PRICING only
(keyed by source then exact model id); API-reported ``usage.cost`` is ignored.
Pass ``silent=True`` to skip per-step cost logs.
"""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import time
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Optional, Union

import httpx
from PIL import Image

from ..data_reader import DatasetReader
from .api_providers import resolve_provider
from .image_utils import encode_image_under_limit, image_content_block, image_data_url_under_limit
from .llm_pricing import PRICING
from .method_utils import dump_json

PathLike = Union[str, Path]

# Connection drops after a long LLM request can already be billed server-side;
# SDK default max_retries=2 would re-send and charge again.
LLM_MAX_RETRIES = 0
# Extra attempts after the first failure for flaky relay upstream keys/errors.
# Total attempts = 1 + LLM_UPSTREAM_RETRIES.
LLM_UPSTREAM_RETRIES = 2
LLM_UPSTREAM_RETRY_BACKOFF_SEC = 10.0
LLM_UPSTREAM_RETRY_MARKERS = (
    "account_deactivated",
    "connection error",
    "server disconnected",
    "remote protocol error",
    "remotely closed",
    "timed out",
    "temporary failure",
)
# Long reasoning models (e.g. gpt-5.6-sol) may stay silent for minutes behind a proxy.
LLM_HTTP_TIMEOUT = httpx.Timeout(connect=60.0, read=1200.0, write=120.0, pool=60.0)
_PROXY_UNSET = object()


def _http_client(proxy=None):
    """Build an httpx client. ``proxy=None`` means direct connect (ignore env proxies)."""
    return httpx.Client(proxy=proxy, trust_env=False, timeout=LLM_HTTP_TIMEOUT)


def _is_upstream_retryable(exc):
    msg = str(exc).lower()
    return any(marker in msg for marker in LLM_UPSTREAM_RETRY_MARKERS)


def _call_with_upstream_retry(fn, max_retries=None, backoff_sec=None):
    """Call ``fn``; on retryable upstream errors, retry up to ``max_retries`` times."""
    if max_retries is None:
        max_retries = LLM_UPSTREAM_RETRIES
    if backoff_sec is None:
        backoff_sec = LLM_UPSTREAM_RETRY_BACKOFF_SEC
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not _is_upstream_retryable(exc):
                raise
            delay = backoff_sec * (2 ** attempt)
            BaseLLM.log("[upstream retry %d/%d in %.1fs] %s" % (attempt + 1, max_retries, delay, exc))
            time.sleep(delay)
    raise last_exc


def load_scene_image(dataset, case_id=None, seed=None):
    """
    Load a scene image from ``dataset/<dataset>/<case_id>/``.

    Provide ``case_id``, or ``seed`` to pick a random case under the dataset.
    Returns ``(reader, image_path, run_name)`` where ``run_name`` is
    ``<dataset>/<case_id>``.
    """
    if not dataset:
        raise ValueError("dataset is required")
    reader = DatasetReader(dataset)
    if case_id is not None:
        reader.read(str(case_id))
    elif seed is not None:
        reader.read_random(seed)
    else:
        raise ValueError("provide case_id or seed")
    case_key = reader.case_id if reader.case_id is not None else Path(reader.filename).stem
    run_name = "%s/%s" % (reader.dataset_name, case_key)
    return reader, str(reader.file_path), run_name


def to_jsonable(obj: Any) -> Any:
    """Recursively make a message structure JSON-serializable, eliding inline
    image bytes (base64 ``data`` fields and ``data:`` URIs). Crop PNGs are
    already on disk under the crops dir."""
    if isinstance(obj, dict):
        clean: dict[str, Any] = {}
        for key, value in obj.items():
            if key == "data" and isinstance(value, str) and len(value) > 256:
                clean[key] = f"<base64 elided: {len(value)} chars>"
            elif key == "url" and isinstance(value, str) and value.startswith("data:"):
                clean[key] = f"<data uri elided: {len(value)} chars>"
            else:
                clean[key] = to_jsonable(value)
        return clean
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    for attr in ("model_dump", "dict", "to_dict"):  # pydantic / SDK objects
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                return to_jsonable(method())
            except Exception:
                pass
    if hasattr(obj, "__dict__"):
        try:
            return to_jsonable(vars(obj))
        except Exception:
            pass
    return str(obj)


def load_json_or_raise(path: Path, label: str) -> dict[str, Any]:
    """Load JSON from disk or raise a clear not-found error."""
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def strip_config_name(config: dict[str, Any]) -> dict[str, Any]:
    """Return config without the optional top-level ``name`` field."""
    return {k: v for k, v in config.items() if k != "name"}


def validate_config_schema(
    cfg: dict[str, Any],
    schema: dict[str, Any],
    *,
    schema_name: str = "schema",
) -> bool:
    """Validate config against a JSON schema with consistent logging."""
    BaseLLM.rule("SCHEMA VALIDATION")
    try:
        import jsonschema
    except ImportError:
        BaseLLM.log("jsonschema not installed; skipping schema validation.")
        return False

    try:
        jsonschema.validate(cfg, schema)
    except jsonschema.ValidationError as exc:
        BaseLLM.log(f"FAIL: {exc.message}")
        BaseLLM.log(f"path: {list(exc.absolute_path)}")
        return False

    BaseLLM.log(f"PASS: conforms to {schema_name}.")
    return True


def save_conversation_and_cost(
    *,
    save_dir: Path,
    name: str,
    source_image: str | Path,
    conversation_payload: dict[str, Any],
    tracker: "CostTracker",
    conversation_filename: str = "conversation.json",
    cost_filename: str = "cost.json",
) -> tuple[Path, Path]:
    """Persist conversation payload and aggregated cost summary."""
    conv_path = save_dir / conversation_filename
    cost_path = save_dir / cost_filename
    dump_json(
        conv_path,
        {
            "name": name,
            "source_image": str(source_image),
            **conversation_payload,
        },
    )
    dump_json(cost_path, tracker.summary())
    BaseLLM.log(f"conversation saved : {conv_path}")
    BaseLLM.log(f"cost saved         : {cost_path}")
    return conv_path, cost_path


def remove_markdown_fence(text: str) -> str:
    """Remove a simple ```json ... ``` wrapper if present."""
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[0].lstrip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from LLM text.

    Tolerates markdown fences and trailing/extra JSON. When multiple objects
    appear (e.g. ``{\"cross_section\":[]}{...}``), prefer the most complete one.
    """
    cleaned = remove_markdown_fence(text)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except JSONDecodeError:
        pass

    preferred_keys = ("cross_section", "regions", "patches", "arms", "mouth_edges")
    candidates = []
    decoder = json.JSONDecoder()
    start = cleaned.find("{")
    while start != -1:
        try:
            parsed, end = decoder.raw_decode(cleaned[start:])
        except JSONDecodeError:
            start = cleaned.find("{", start + 1)
            continue
        if isinstance(parsed, dict):
            candidates.append(parsed)
            start = cleaned.find("{", start + max(end, 1))
        else:
            start = cleaned.find("{", start + 1)

    if not candidates:
        raise ValueError("Model response did not contain a valid JSON object.")

    candidates.sort(
        key=lambda item: (
            sum(1 for key in preferred_keys if key in item),
            len(json.dumps(item, ensure_ascii=False)),
        ),
        reverse=True,
    )
    return candidates[0]


class TruncatedResponseError(ValueError):
    """Raised when the model stopped because it hit the token limit.

    With structured outputs the returned text is then incomplete JSON, so
    parsing would fail with a cryptic JSONDecodeError. We surface a clear
    error instead so callers can raise ``max_tokens`` and retry.
    """


class BaseLLM:
    """Provider-agnostic helpers shared by every client.

    Holds parsing, agent-loop logging, the neutral tool-result container,
    and output writing. Subclasses add provider-specific request/response
    handling and the normalized agent interface methods.
    """

    verbose = True

    # -- parsing ---------------------------------------------------------------
    @staticmethod
    def guess_mime_type(path: PathLike) -> str:
        mime_type, _ = mimetypes.guess_type(str(path))
        return mime_type or "image/png"

    @staticmethod
    def parse_response_json(text: str) -> dict:
        """Parse model JSON, tolerating markdown fences and trailing extra data."""
        text = (text or "").strip()
        if not text:
            return {}
        return extract_json_object(text)

    # -- agent-loop logging ----------------------------------------------------
    @staticmethod
    def log(msg: str, indent: int = 0) -> None:
        if not BaseLLM.verbose:
            return
        print(("  " * indent) + msg, flush=True)

    @staticmethod
    def rule(title: str) -> None:
        if not BaseLLM.verbose:
            return
        print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70, flush=True)

    # -- neutral tool-result container ----------------------------------------
    @staticmethod
    def tool_result(
        tool_use_id: str,
        text: str,
        *,
        is_error: bool = False,
        image: Optional[Image.Image] = None,
        images: Optional[list] = None,
    ) -> dict[str, Any]:
        """Provider-neutral tool result. Rendered into the correct wire format
        by each provider's ``append_tool_results``.

        Pass ``image`` for one zoom, or ``images`` for several (both may be set;
        ``image`` is prepended when both are provided).
        """
        out_images = []
        if image is not None:
            out_images.append(image)
        if images:
            out_images.extend(images)
        return {
            "tool_use_id": tool_use_id,
            "text": text,
            "is_error": is_error,
            "image": out_images[0] if len(out_images) == 1 else None,
            "images": out_images,
        }

    # -- shared agent helpers --------------------------------------------------
    def wants_tools(self, response: Any) -> bool:
        """True if the model requested at least one tool call."""
        return bool(self.tool_calls(response))

    def tool_calls(self, response: Any) -> list[dict]:  # pragma: no cover - overridden
        raise NotImplementedError

    # -- reasoning / effort routing --------------------------------------------
    provider = None

    def apply_reasoning(self, kwargs, reasoning):
        """Attach provider-specific reasoning settings to request kwargs.

        ``reasoning`` is a shared effort label (e.g. none/low/medium/high/xhigh/max).
        Pass ``None`` to leave the request unmodified.
        """
        if reasoning is None:
            return kwargs
        if self.provider == "anthropic":
            return BaseLLM._apply_anthropic_reasoning(kwargs, reasoning)
        if self.provider == "openai":
            return BaseLLM._apply_openai_reasoning(kwargs, reasoning)
        return kwargs

    @staticmethod
    def _apply_openai_reasoning(kwargs, reasoning, *, include_openrouter_body=False):
        """Chat Completions ``reasoning_effort``; optionally OpenRouter ``reasoning``.

        Chat Completions (official OpenAI and compatible proxies) accept
        top-level ``reasoning_effort`` only. The nested ``reasoning`` object is
        OpenRouter / Responses-API style and is rejected as
        ``Unknown parameter: 'reasoning'`` on strict Chat Completions gateways.
        """
        out = dict(kwargs)
        out["reasoning_effort"] = reasoning
        if not include_openrouter_body:
            return out
        extra_body = dict(out.get("extra_body") or {})
        body = dict(extra_body.get("reasoning") or {})
        body["effort"] = reasoning
        body["exclude"] = True
        extra_body["reasoning"] = body
        out["extra_body"] = extra_body
        return out

    @staticmethod
    def _apply_anthropic_reasoning(kwargs, reasoning):
        """Anthropic adaptive thinking + ``output_config.effort``.

        ``none`` leaves thinking off. Any other label enables
        ``thinking: {type: adaptive}`` and merges ``effort`` into ``output_config``
        (alongside an existing JSON-schema ``format`` when present).
        """
        out = dict(kwargs)
        if reasoning == "none":
            return out
        out["thinking"] = {"type": "adaptive"}
        output_config = dict(out.get("output_config") or {})
        output_config["effort"] = reasoning
        out["output_config"] = output_config
        return out

    # -- output writing --------------------------------------------------------
    @staticmethod
    def write_outputs(
        output: dict[str, Any],
        output_dir: Path,
        *,
        config_name: str = "xodr_config.json",
        meta_name: str = "agent_run.json",
    ) -> None:
        config_path = output_dir / config_name
        meta_path = output_dir / meta_name
        dump_json(config_path, output["config"])
        dump_json(meta_path, {k: v for k, v in output.items() if k != "config"})
        print(f"\nWrote {config_path} and {meta_path}", flush=True)


# --------------------------------------------------------------------------- #
# Cost tracking
# --------------------------------------------------------------------------- #

def _price_for(model, source=None):
    """Look up USD/1M rates for ``source`` + exact model key."""
    if not source:
        return None
    table = PRICING.get(source) or {}
    return table.get(model)


def _usage_obj(resp):
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    return usage


def _extract_usage(resp: Any) -> dict[str, int]:
    """Best-effort token counts from a provider response (Anthropic/OpenAI SDK
    object or dict). Missing fields default to 0."""
    usage = _usage_obj(resp)
    if usage is None:
        return {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}

    def field(*names: str) -> int:
        for n in names:
            v = getattr(usage, n, None)
            if v is None and isinstance(usage, dict):
                v = usage.get(n)
            if v is not None:
                return int(v)
        return 0

    return {
        "input": field("input_tokens", "prompt_tokens"),
        "output": field("output_tokens", "completion_tokens"),
        "cache_read": field("cache_read_input_tokens"),
        "cache_write": field("cache_creation_input_tokens"),
    }


def _cost_usd(model, usage, source=None):
    """Estimate USD from ``llm_pricing.PRICING`` (token counts × USD/1M rates)."""
    price = _price_for(model, source=source)
    if price is None:
        return None
    return sum(usage[k] * price.get(k, 0.0) for k in usage) / 1_000_000


class CostTracker:
    """Accumulates per-step token usage and estimated cost across both stages.

    Set ``silent=True`` to skip per-step cost logs (data is still recorded).
    ``print_summary`` always prints when called.
    """

    _FIELDS = ("input", "output", "cache_read", "cache_write")

    def __init__(self, source=None, silent=False):
        self.source = source
        self.silent = bool(silent)
        self.steps = []

    def record(self, stage, model, resp, source=None):
        src = self.source if source is None else source
        usage = _extract_usage(resp)
        cost = _cost_usd(model, usage, source=src)
        cost_from = "table" if cost is not None else None
        self.steps.append({"stage": stage, "model": model, "source": src, **usage, "cost_usd": cost, "cost_from": cost_from})
        if self.silent:
            return
        if cost is not None:
            cost_note = f"  ${cost:.4f} ({cost_from})"
        else:
            cost_note = f"  $?(no price for source={src!r} model={model!r})"
        BaseLLM.log(
            f"[cost] {stage}: in={usage['input']} out={usage['output']} "
            f"cache_r={usage['cache_read']} cache_w={usage['cache_write']}{cost_note}",
            indent=1,
        )

    def totals(self) -> dict:
        tot = {f: 0 for f in self._FIELDS}
        cost, priced = 0.0, False
        for s in self.steps:
            for f in self._FIELDS:
                tot[f] += s[f]
            if s["cost_usd"] is not None:
                cost += s["cost_usd"]
                priced = True
        tot["cost_usd"] = cost if priced else None
        return tot

    def summary(self) -> dict:
        return {"steps": self.steps, "totals": self.totals()}

    def print_summary(self) -> None:
        BaseLLM.rule("COST")
        for s in self.steps:
            cost = f"${s['cost_usd']:.4f}" if s["cost_usd"] is not None else "$?"
            BaseLLM.log(
                f"{s['stage']:<16} {s['model']:<22} "
                f"in={s['input']:>7} out={s['output']:>6} "
                f"cache_r={s['cache_read']:>7} cache_w={s['cache_write']:>7} {cost}"
            )
        tot = self.totals()
        total_cost = f"${tot['cost_usd']:.4f}" if tot["cost_usd"] is not None else "$? (some models unpriced)"
        BaseLLM.log(
            f"{'TOTAL':<16} {'':<22} "
            f"in={tot['input']:>7} out={tot['output']:>6} "
            f"cache_r={tot['cache_read']:>7} cache_w={tot['cache_write']:>7} {total_cost}"
        )


def _strip_trailing_v1_base_url(base_url):
    """Drop a trailing ``/v1`` so the Anthropic SDK does not produce ``/v1/v1/...``.

    OpenAI-compatible gateways store ``https://host/v1``. The Anthropic client
    always appends ``/v1/messages`` and ``/v1/files`` itself.
    """
    if not base_url:
        return base_url
    url = str(base_url).rstrip("/")
    if url.lower().endswith("/v1"):
        url = url[:-3].rstrip("/")
    return url or None


def _is_official_anthropic_base_url(base_url):
    """True when the SDK should talk to api.anthropic.com (default host)."""
    if not base_url:
        return True
    host = str(base_url).lower()
    return "api.anthropic.com" in host


class AnthropicLLM(BaseLLM):
    """Claude via Anthropic SDK (Files API upload, cache_control prompt caching,
    structured-output call_json, and the normalized tool-use agent interface).""" 

    provider = "anthropic"
    FILES_API_BETA = "files-api-2025-04-14"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        proxy=_PROXY_UNSET,
        use_files_api: Optional[bool] = None,
    ) -> None:
        import anthropic

        provider = resolve_provider("anthropic")
        if proxy is _PROXY_UNSET:
            proxy = provider["proxy"]
        kwargs: dict[str, Any] = {
            "api_key": api_key or provider["api_key"],
            "max_retries": LLM_MAX_RETRIES,
            "timeout": LLM_HTTP_TIMEOUT,
            "http_client": _http_client(proxy),
        }
        resolved_base_url = base_url if base_url is not None else provider["base_url"]
        resolved_base_url = _strip_trailing_v1_base_url(resolved_base_url)
        if resolved_base_url:
            kwargs["base_url"] = resolved_base_url
        self._client = anthropic.Anthropic(**kwargs)
        # Prefer the Files API for image upload on official Anthropic. Compatible
        # gateways often only implement ``messages.create`` and reject
        # ``beta.files.upload``, so skip the upload and send inline base64.
        if use_files_api is None:
            use_files_api = _is_official_anthropic_base_url(resolved_base_url)
        self._use_files_api = bool(use_files_api)
        self._using_file_refs = False

    @property
    def client(self) -> Any:
        return self._client

    @property
    def using_file_refs(self) -> bool:
        """Whether the current image is delivered as a Files API reference."""
        return self._using_file_refs

    @staticmethod
    def image_to_base64(image_path: PathLike) -> str:
        return base64.standard_b64encode(Path(image_path).read_bytes()).decode("ascii")

    def upload_image(self, image_path: PathLike) -> str:
        path = Path(image_path)
        with path.open("rb") as f:
            uploaded = self._client.beta.files.upload(
                file=(path.name, f, self.guess_mime_type(path)),
            )
        return uploaded.id

    def upload_pil_image(self, image: Image.Image, *, name: str = "image.png") -> str:
        """Upload an in-memory PIL image via the Files API; returns its file id.

        Avoids touching disk: the image is encoded under the size limit in a
        BytesIO buffer and streamed straight to the Files API.
        """
        raw, media_type = encode_image_under_limit(image)
        ext = "jpg" if media_type == "image/jpeg" else "png"
        upload_name = Path(name).stem + "." + ext
        buf = io.BytesIO(raw)
        uploaded = self._client.beta.files.upload(file=(upload_name, buf, media_type))
        return uploaded.id

    @staticmethod
    def make_image_message(file_id: str, prompt: str) -> dict:
        """Image-by-reference message (requires the Files API beta)."""
        return {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "file", "file_id": file_id}},
                {"type": "text", "text": prompt},
            ],
        }

    @classmethod
    def make_image_message_base64(cls, image_path: PathLike, prompt: str) -> dict:
        """Inline base64 image message (works on the non-beta endpoint)."""
        path = Path(image_path)
        return {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": cls.guess_mime_type(path),
                        "data": cls.image_to_base64(path),
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }

    @staticmethod
    def make_user_message(prompt: str) -> dict:
        return {"role": "user", "content": [{"type": "text", "text": prompt}]}

    @staticmethod
    def make_assistant_message(text: str) -> dict:
        return {"role": "assistant", "content": [{"type": "text", "text": text or ""}]}

    @staticmethod
    def set_image_file_id(messages: list[dict], file_id: str) -> None:
        for message in messages:
            for block in message.get("content", []):
                if isinstance(block, dict) and block.get("type") == "image":
                    source = block.get("source")
                    if isinstance(source, dict) and source.get("type") == "file":
                        source["file_id"] = file_id

    @staticmethod
    def message_text(response: Any) -> str:
        if not getattr(response, "content", None):
            return ""
        parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "\n".join(parts).strip()

    @staticmethod
    def _raise_if_truncated(response: Any) -> None:
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise TruncatedResponseError(
                "Anthropic response truncated (stop_reason='max_tokens'); "
                "the returned JSON is incomplete. Increase max_tokens and retry."
            )

    @staticmethod
    def _build_system(
        system_prompt: str,
        *,
        use_prompt_caching: bool,
        cache_ttl: Optional[str],
    ) -> Any:
        """Place an explicit cache breakpoint on the (static) system prompt.

        Automatic top-level caching would put the breakpoint on the last
        cacheable block, which for a per-image request is the user content
        that changes every call -- producing cache writes but no reads.
        Marking the system block caches the stable prefix instead.
        """
        if not use_prompt_caching:
            return system_prompt
        cache_control: dict[str, str] = {"type": "ephemeral"}
        if cache_ttl:
            cache_control["ttl"] = cache_ttl
        return [{"type": "text", "text": system_prompt, "cache_control": cache_control}]

    # -- normalized agent interface -------------------------------------------
    def image_user_message(self, image: Image.Image, text: str) -> dict:
        """Image+text user turn.

        Uploads via the Files API and references the file id when possible; on
        any upload failure (or when Files API is disabled) falls back to an
        inline base64 image block. Sets ``self._using_file_refs`` so later
        requests know whether to route through the beta endpoint. A cache_control
        breakpoint is placed on the turn so the image is cached and reused across
        the Stage 2 agent loop."""
        if self._use_files_api:
            try:
                file_id = self.upload_pil_image(image, name="full_diagram.png")
                self._using_file_refs = True
                BaseLLM.log(f"[files-api] uploaded image -> {file_id}")
                return self._add_cache_breakpoint(self.make_image_message(file_id, text))
            except Exception as exc:  # noqa: BLE001 - fall back to inline base64
                BaseLLM.log(f"[files-api upload failed -> base64 fallback: {exc}]")
        self._using_file_refs = False
        return self._add_cache_breakpoint(
            {
                "role": "user",
                "content": [image_content_block(image), {"type": "text", "text": text}],
            }
        )

    def create_message(
        self,
        *,
        model: str,
        messages: list[dict],
        max_tokens: int,
        system: Optional[Any] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[dict] = None,
        temperature: Optional[float] = None,
        reasoning_effort=None,
    ) -> Any:
        """Raw Messages create for the tool-use agent loop. Tool defs and
        tool_choice are already in Anthropic-native shape, so pass through.

        When the conversation references an uploaded file the request must go
        through the beta endpoint with the Files API beta header; the inline
        base64 fallback uses the standard endpoint.
        """
        if system is not None:
            system, messages = self._cap_cache_breakpoints(system, messages)
        else:
            _, messages = self._cap_cache_breakpoints([], messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system is not None:
            kwargs["system"] = system
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if temperature is not None:
            kwargs["temperature"] = temperature
        kwargs = self.apply_reasoning(kwargs, reasoning_effort)

        if self._using_file_refs:
            return self._client.beta.messages.create(
                betas=[self.FILES_API_BETA], **kwargs
            )
        return self._client.messages.create(**kwargs)

    @staticmethod
    def assistant_message(response: Any) -> dict:
        return {"role": "assistant", "content": response.content}

    @staticmethod
    def tool_calls(response: Any) -> list[dict]:
        calls = []
        for blk in getattr(response, "content", None) or []:
            if getattr(blk, "type", None) == "tool_use":
                calls.append({"id": blk.id, "name": blk.name, "input": blk.input})
        return calls

    def append_tool_results(self, messages: list[dict], results: list[dict]) -> None:
        blocks: list[dict] = []
        for r in results:
            content: list[dict] = [{"type": "text", "text": r["text"]}]
            for img in r.get("images") or ([] if r.get("image") is None else [r["image"]]):
                content.append(image_content_block(img))
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": r["tool_use_id"],
                "content": content,
            }
            if r.get("is_error"):
                block["is_error"] = True
            blocks.append(block)
        messages.append({"role": "user", "content": blocks})

    @staticmethod
    def _add_cache_breakpoint(message: dict, *, ttl: Optional[str] = None) -> dict:
        """Stamp a cache_control breakpoint on the turn's last content block, so
        the prefix up to and including this user turn (system + image) is cached.
        In the Stage 2 tool-use loop every later turn then reads the image from
        cache instead of re-paying for it as fresh input tokens."""
        cache_control: dict[str, str] = {"type": "ephemeral"}
        if ttl:
            cache_control["ttl"] = ttl
        content = message.get("content")
        if isinstance(content, list) and content:
            content[-1] = {**content[-1], "cache_control": cache_control}
        return message

    @classmethod
    def _cap_cache_breakpoints(cls, system, messages, *, max_blocks=4):
        """Anthropic allows at most 4 ``cache_control`` blocks per request.

        Keep the system breakpoint(s) when possible, then the latest message
        breakpoints. Returns copies when stripping is needed so caller-owned
        message history is left intact.
        """
        system_out = system
        if isinstance(system, list):
            system_out = [dict(block) if isinstance(block, dict) else block for block in system]
        messages_out = []
        for message in messages or []:
            if not isinstance(message, dict):
                messages_out.append(message)
                continue
            content = message.get("content")
            if isinstance(content, list):
                messages_out.append({**message, "content": [dict(block) if isinstance(block, dict) else block for block in content]})
            else:
                messages_out.append(message)

        marked = []
        if isinstance(system_out, list):
            for idx, block in enumerate(system_out):
                if isinstance(block, dict) and "cache_control" in block:
                    marked.append(("system", idx))
        for msg_idx, message in enumerate(messages_out):
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            for block_idx, block in enumerate(content):
                if isinstance(block, dict) and "cache_control" in block:
                    marked.append(("message", msg_idx, block_idx))

        if len(marked) <= max_blocks:
            return system_out, messages_out

        system_marks = [item for item in marked if item[0] == "system"]
        other_marks = [item for item in marked if item[0] != "system"]
        if len(system_marks) >= max_blocks:
            keep = set(system_marks[-max_blocks:])
        else:
            keep = set(system_marks)
            budget = max_blocks - len(keep)
            if budget > 0:
                keep.update(other_marks[-budget:])

        for item in marked:
            if item in keep:
                continue
            if item[0] == "system":
                block = system_out[item[1]]
                system_out[item[1]] = {key: value for key, value in block.items() if key != "cache_control"}
            else:
                _, msg_idx, block_idx = item
                block = messages_out[msg_idx]["content"][block_idx]
                messages_out[msg_idx]["content"][block_idx] = {key: value for key, value in block.items() if key != "cache_control"}
        BaseLLM.log(f"[cache] capped cache_control blocks {len(marked)} -> {len(keep)} (max={max_blocks})")
        return system_out, messages_out

    # -- structured output -----------------------------------------------------
    def call_json(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[dict],
        schema: dict,
        max_tokens: int,
        use_files_api: Optional[bool] = None,
        use_prompt_caching: bool = True,
        cache_ttl: Optional[str] = None,
        temperature: Optional[float] = None,
        reasoning_effort=None,
    ):
        # Auto: follow whatever image_user_message produced. When the image is
        # a Files API reference we must hit the beta endpoint; when it fell
        # back to inline base64 we stay on the standard endpoint (and avoid
        # sending a beta header that the account may not have enabled).
        if use_files_api is None:
            use_files_api = self._using_file_refs

        create = (
            self._client.beta.messages.create
            if use_files_api
            else self._client.messages.create
        )

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        if system_prompt:
            system = self._build_system(
                system_prompt,
                use_prompt_caching=use_prompt_caching,
                cache_ttl=cache_ttl,
            )
            system, messages = self._cap_cache_breakpoints(system, messages)
            kwargs["system"] = system
            kwargs["messages"] = messages
        if temperature is not None:
            kwargs["temperature"] = temperature
        kwargs = self.apply_reasoning(kwargs, reasoning_effort)

        if use_files_api:
            kwargs["betas"] = [self.FILES_API_BETA]

        response = create(**kwargs)
        self._raise_if_truncated(response)
        text = self.message_text(response)
        return self.parse_response_json(text), text, response

    def call_with_image_json(
        self,
        *,
        model: str,
        system_prompt: str,
        prompt: str,
        schema: dict,
        max_tokens: int,
        image_path: Optional[PathLike] = None,
        image_file_id: Optional[str] = None,
        use_files_api: bool = True,
        use_prompt_caching: bool = True,
        cache_ttl: Optional[str] = None,
        temperature: Optional[float] = None,
        reasoning_effort=None,
    ) -> tuple[dict, str, list[dict]]:
        if use_files_api:
            if image_file_id is None:
                if image_path is None:
                    raise ValueError("image_path or image_file_id is required")
                image_file_id = self.upload_image(image_path)
            messages = [self.make_image_message(image_file_id, prompt)]
            self._using_file_refs = True
        else:
            if image_file_id is not None:
                raise ValueError(
                    "image_file_id requires use_files_api=True; "
                    "pass image_path with use_files_api=False to embed inline"
                )
            if image_path is None:
                raise ValueError("image_path is required when use_files_api=False")
            messages = [self.make_image_message_base64(image_path, prompt)]
            self._using_file_refs = False

        result, text, _ = self.call_json(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            schema=schema,
            max_tokens=max_tokens,
            use_files_api=use_files_api,
            use_prompt_caching=use_prompt_caching,
            cache_ttl=cache_ttl,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        messages.append(self.make_assistant_message(text))
        return result, text, messages


class OpenAILLM(BaseLLM):
    """OpenAI Chat Completions or compatible base_url (data-URL vision,
    json_schema structured output, and the normalized tool-use agent interface).

    OpenAI has no Anthropic-style Files API, so images are always embedded as
    base64 data URLs -- i.e. this provider is permanently on the base64 path.
    """

    provider = "openai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        proxy=_PROXY_UNSET,
        token_param: str = "max_completion_tokens",
        upstream_retries=None,
        upstream_retry_backoff_sec=None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required; install with: pip install openai"
            ) from exc

        provider = resolve_provider("openrouter")
        if proxy is _PROXY_UNSET:
            proxy = provider["proxy"]
        self._base_url = base_url or provider["base_url"] or os.getenv("OPENAI_BASE_URL")
        self._client = OpenAI(
            api_key=api_key or provider["api_key"],
            base_url=self._base_url,
            max_retries=LLM_MAX_RETRIES,
            timeout=LLM_HTTP_TIMEOUT,
            http_client=_http_client(proxy),
        )
        # Official OpenAI deprecated `max_tokens` for Chat Completions and
        # reasoning models reject it; default to `max_completion_tokens`.
        # Compatible base_urls that only accept `max_tokens` are handled by
        # the automatic fallback in `_create`.
        self._token_param = token_param
        self._upstream_retries = LLM_UPSTREAM_RETRIES if upstream_retries is None else upstream_retries
        self._upstream_retry_backoff_sec = (
            LLM_UPSTREAM_RETRY_BACKOFF_SEC if upstream_retry_backoff_sec is None else upstream_retry_backoff_sec
        )

    def _is_zhipu_endpoint(self):
        base = (self._base_url or "").lower()
        return "bigmodel.cn" in base

    def _is_openrouter_endpoint(self):
        base = (self._base_url or "").lower()
        return "openrouter.ai" in base

    def apply_reasoning(self, kwargs, reasoning):
        """Zhipu uses ``thinking``; OpenRouter may also take ``reasoning``;
        other Chat Completions endpoints use ``reasoning_effort`` only."""
        if reasoning is None:
            return kwargs
        if self._is_zhipu_endpoint():
            out = dict(kwargs)
            if reasoning == "none":
                out["extra_body"] = {**(out.get("extra_body") or {}), "thinking": {"type": "disabled"}}
            else:
                out["extra_body"] = {**(out.get("extra_body") or {}), "thinking": {"type": "enabled"}}
            return out
        return BaseLLM._apply_openai_reasoning(
            kwargs,
            reasoning,
            include_openrouter_body=self._is_openrouter_endpoint(),
        )

    @property
    def client(self) -> Any:
        return self._client

    @staticmethod
    def image_to_data_url(image_path: PathLike) -> str:
        path = Path(image_path)
        mime_type = OpenAILLM.guess_mime_type(path)
        payload = base64.standard_b64encode(path.read_bytes()).decode("ascii")
        return "data:%s;base64,%s" % (mime_type, payload)

    @staticmethod
    def _pil_to_data_url(image: Image.Image) -> str:
        return image_data_url_under_limit(image)

    @classmethod
    def make_image_message(cls, image_path: PathLike, prompt: str) -> dict:
        return cls.make_image_message_from_url(cls.image_to_data_url(image_path), prompt)

    @staticmethod
    def make_image_message_from_url(image_data_url: str, prompt: str) -> dict:
        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": prompt},
            ],
        }

    @staticmethod
    def make_user_message(prompt: str) -> dict:
        return {"role": "user", "content": prompt}

    @staticmethod
    def make_assistant_message(text: str) -> dict:
        return {"role": "assistant", "content": text or ""}

    @staticmethod
    def set_image_url(messages: list[dict], image_data_url: str) -> None:
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    block["image_url"] = {"url": image_data_url}

    @classmethod
    def refresh_image_in_messages(cls, messages: list[dict], image_path: PathLike) -> None:
        cls.set_image_url(messages, cls.image_to_data_url(image_path))

    @staticmethod
    def message_text(response: Any) -> str:
        if not getattr(response, "choices", None):
            return ""

        message = response.choices[0].message
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                else:
                    text = getattr(block, "text", None)
                    if text:
                        parts.append(text)
            return "\n".join(parts).strip()

        return ""

    @staticmethod
    def _raise_if_truncated(response: Any) -> None:
        choices = getattr(response, "choices", None)
        if choices and getattr(choices[0], "finish_reason", None) == "length":
            raise TruncatedResponseError(
                "OpenAI response truncated (finish_reason='length'); "
                "the returned JSON is incomplete. Increase max_tokens and retry."
            )

    @classmethod
    def _enforce_strict_schema(cls, schema: Any) -> Any:
        """Best-effort normalization so a schema passes OpenAI strict mode.

        OpenAI strict json_schema requires every object to set
        ``additionalProperties: false`` and list all properties in
        ``required``. Anthropic does not, so a schema authored for Anthropic
        may otherwise be rejected with a 400.

        Note: this marks ALL properties as required. Genuinely optional
        fields should be modeled as a nullable union (e.g. type
        ``["string", "null"]``) rather than omitted.
        """
        if not isinstance(schema, dict):
            return schema

        out = dict(schema)
        props = out.get("properties")
        if isinstance(props, dict):
            out["properties"] = {k: cls._enforce_strict_schema(v) for k, v in props.items()}
            out["additionalProperties"] = False
            out["required"] = list(props.keys())

        if "items" in out:
            out["items"] = cls._enforce_strict_schema(out["items"])

        for key in ("anyOf", "oneOf", "allOf"):
            if isinstance(out.get(key), list):
                out[key] = [cls._enforce_strict_schema(s) for s in out[key]]

        for key in ("$defs", "definitions"):
            if isinstance(out.get(key), dict):
                out[key] = {k: cls._enforce_strict_schema(v) for k, v in out[key].items()}

        return out

    def _request_chat_completion(self, attempt):
        try:
            return _call_with_upstream_retry(
                lambda: self._client.chat.completions.create(**attempt),
                max_retries=self._upstream_retries,
                backoff_sec=self._upstream_retry_backoff_sec,
            )
        except Exception as exc:
            message = str(exc)
            if "content.type" in message and "text" in message and "image" in str(attempt.get("messages") or "").lower():
                model = attempt.get("model")
                raise RuntimeError(
                    f"model {model!r} rejected image_url content (text-only endpoint). "
                    f"For vision tasks use a VLM such as glm-5v-turbo / glm-4.6v. Original error: {exc}"
                ) from exc
            raise

    def _create(self, kwargs: dict, max_tokens: int, reasoning_effort="none") -> Any:
        """Call the API, retrying with the other token field if rejected."""
        primary = self._token_param
        fallback = "max_tokens" if primary != "max_tokens" else "max_completion_tokens"

        kwargs = self.apply_reasoning(kwargs, reasoning_effort)
        attempt = dict(kwargs)
        attempt[primary] = max_tokens
        try:
            return self._request_chat_completion(attempt)
        except Exception as exc:  # noqa: BLE001 - inspect message then re-raise
            message = str(exc)
            param_rejected = (primary in message) and (
                "unsupported" in message.lower()
                or "not supported" in message.lower()
                or "unexpected" in message.lower()
                or "use 'max_completion_tokens'" in message.lower()
            )
            if not param_rejected:
                raise
            retry = dict(kwargs)
            retry[fallback] = max_tokens
            return self._request_chat_completion(retry)

    # -- tool-def / tool_choice translation (Anthropic-native -> OpenAI) ------
    @classmethod
    def _to_openai_tools(cls, tools: Optional[list]) -> Optional[list]:
        if tools is None:
            return None
        out = []
        for t in tools:
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": cls._enforce_strict_schema(t["input_schema"]),
                        "strict": True,
                    },
                }
            )
        return out

    @staticmethod
    def _to_openai_tool_choice(tool_choice: Optional[dict]) -> Optional[Any]:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, dict):
            kind = tool_choice.get("type")
            if kind == "tool" and tool_choice.get("name"):
                return {"type": "function", "function": {"name": tool_choice["name"]}}
            if kind == "auto":
                return "auto"
            if kind == "any":
                return "required"
        return tool_choice

    # -- normalized agent interface -------------------------------------------
    def image_user_message(self, image: Image.Image, text: str) -> dict:
        """Image+text user turn. OpenAI has no Files API, so the image is always
        embedded as a base64 data URL."""
        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": self._pil_to_data_url(image)}},
                {"type": "text", "text": text},
            ],
        }

    def create_message(
        self,
        *,
        model: str,
        messages: list[dict],
        max_tokens: int,
        system: Optional[str] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[Any] = None,
        temperature: Optional[float] = None,
        reasoning_effort="none",
    ) -> Any:
        """Raw Chat Completions create for the tool-use agent loop. Translates
        Anthropic-native tool defs / tool_choice into OpenAI shape."""
        api_messages: list[dict] = []
        if system is not None:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(messages)

        kwargs: dict[str, Any] = {"model": model, "messages": api_messages}
        oa_tools = self._to_openai_tools(tools)
        if oa_tools is not None:
            kwargs["tools"] = oa_tools
        oa_choice = self._to_openai_tool_choice(tool_choice)
        if oa_choice is not None:
            kwargs["tool_choice"] = oa_choice
        if temperature is not None:
            kwargs["temperature"] = temperature
        return self._create(kwargs, max_tokens, reasoning_effort=reasoning_effort)

    @staticmethod
    def assistant_message(response: Any) -> dict:
        msg = response.choices[0].message
        out: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
        return out

    @staticmethod
    def tool_calls(response: Any) -> list[dict]:
        msg = response.choices[0].message
        calls = []
        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except (ValueError, TypeError):
                args = {}
            calls.append({"id": tc.id, "name": tc.function.name, "input": args})
        return calls

    def append_tool_results(self, messages: list[dict], results: list[dict]) -> None:
        # All tool responses must immediately follow the assistant tool-call
        # turn. OpenAI tool messages are text-only, so any returned images are
        # appended afterwards as a separate user message.
        images: list[Image.Image] = []
        for r in results:
            body = ("ERROR: " + r["text"]) if r.get("is_error") else r["text"]
            result_images = r.get("images") or ([] if r.get("image") is None else [r["image"]])
            if result_images:
                body += " (zoomed image(s) attached in the following message)"
                images.extend(result_images)
            messages.append(
                {"role": "tool", "tool_call_id": r["tool_use_id"], "content": body}
            )
        if images:
            parts: list[dict] = [
                {"type": "image_url", "image_url": {"url": self._pil_to_data_url(img)}}
                for img in images
            ]
            parts.append(
                {"type": "text", "text": "Zoomed crop(s) for the preceding tool call(s)."}
            )
            messages.append({"role": "user", "content": parts})

    # -- structured output -----------------------------------------------------
    def call_json(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[dict],
        schema: dict,
        max_tokens: int,
        schema_name: str = "response",
        use_prompt_caching: bool = False,
        prompt_cache_key: Optional[str] = None,
        strict: bool = True,
        temperature: Optional[float] = None,
        reasoning_effort="none",
    ):
        api_messages: list[dict] = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(messages)

        json_schema = self._enforce_strict_schema(schema) if strict else schema

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": json_schema,
                    "strict": strict,
                },
            },
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        if use_prompt_caching and prompt_cache_key:
            kwargs["prompt_cache_key"] = prompt_cache_key

        response = self._create(kwargs, max_tokens, reasoning_effort=reasoning_effort)
        self._raise_if_truncated(response)
        text = self.message_text(response)
        return self.parse_response_json(text), text, response

    def call_with_image_json(
        self,
        *,
        model: str,
        system_prompt: str,
        prompt: str,
        schema: dict,
        image_path: PathLike,
        max_tokens: int,
        schema_name: str = "response",
        use_prompt_caching: bool = False,
        prompt_cache_key: Optional[str] = None,
        strict: bool = True,
        temperature: Optional[float] = None,
        reasoning_effort="none",
    ) -> tuple[dict, str, list[dict]]:
        messages = [self.make_image_message(image_path, prompt)]
        result, text, _ = self.call_json(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            schema=schema,
            max_tokens=max_tokens,
            schema_name=schema_name,
            use_prompt_caching=use_prompt_caching,
            prompt_cache_key=prompt_cache_key,
            strict=strict,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        messages.append(self.make_assistant_message(text))
        return result, text, messages


def build_llm(
    provider: str,
    api_key: Optional[str],
    base_url: Optional[str],
    proxy=_PROXY_UNSET,
    upstream_retries=None,
    upstream_retry_backoff_sec=None,
) -> BaseLLM:
    """Pick provider backend. ``api_key`` / ``proxy`` may be omitted to use api_providers defaults."""
    if provider == "openai":
        return OpenAILLM(
            api_key=api_key,
            base_url=base_url,
            proxy=proxy,
            upstream_retries=upstream_retries,
            upstream_retry_backoff_sec=upstream_retry_backoff_sec,
        )
    if provider == "anthropic":
        return AnthropicLLM(api_key=api_key, base_url=base_url, proxy=proxy)
    raise ValueError(f"unknown provider: {provider!r} (use 'anthropic' or 'openai')")