"""vertex-proxy FastAPI app.

Exposes:
  - POST /anthropic/v1/messages                    : Anthropic-compatible, forwards to Vertex.
  - POST /gemini/v1beta/models/{m}:generateContent : Gemini-compatible, forwards to Vertex.
  - GET  /health                                   : liveness + token status.
  - GET  /v1/models                                : list routable models.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .auth import TokenManager
from .config import Settings, load_settings

logger = logging.getLogger(__name__)

DEFAULT_HTTP_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
# Vertex streaming responses can legitimately go quiet for longer than the
# default read window while the model is thinking. Keep connect/write/pool
# bounded, but do not abort a live stream just because no chunk arrived.
STREAM_HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=120.0, pool=120.0)


def _vertex_base_url(region: str) -> str:
    """Return the Vertex AI base URL for a given region.

    Matches LiteLLM's ``get_vertex_base_url`` semantics:
      - ``"global"`` → ``https://aiplatform.googleapis.com``
      - Multi-region geographies (no hyphen, e.g. ``"us"``, ``"eu"``) →
        ``https://aiplatform.{geo}.rep.googleapis.com``
      - Regional locations (e.g. ``"us-east5"``) →
        ``https://{region}-aiplatform.googleapis.com``
    """
    if region == "global":
        return "https://aiplatform.googleapis.com"
    if "-" not in region:
        return f"https://aiplatform.{region}.rep.googleapis.com"
    return f"https://{region}-aiplatform.googleapis.com"


# --- Metrics (Prometheus-format, tiny in-memory counters) -------------------
# We deliberately don't pull in prometheus_client to keep the dep footprint
# minimal. This is good enough for a local proxy; use a real metrics library
# for production multi-instance deployments.


class _Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: Counter[tuple[str, str, str]] = Counter()
        self._tokens_in: Counter[str] = Counter()
        self._tokens_out: Counter[str] = Counter()
        self._started_at = time.time()

    def record_request(self, route: str, model: str, status: int) -> None:
        with self._lock:
            self._requests[(route, model, str(status))] += 1

    def record_tokens(self, model: str, prompt: int, completion: int) -> None:
        with self._lock:
            self._tokens_in[model] += prompt
            self._tokens_out[model] += completion

    def render(self) -> str:
        """Render Prometheus exposition format."""
        lines = [
            "# HELP vertex_proxy_uptime_seconds Seconds since proxy start",
            "# TYPE vertex_proxy_uptime_seconds gauge",
            f"vertex_proxy_uptime_seconds {time.time() - self._started_at:.0f}",
            "# HELP vertex_proxy_requests_total Total requests by route, model, and status",
            "# TYPE vertex_proxy_requests_total counter",
        ]
        with self._lock:
            for (route, model, status), count in self._requests.items():
                lines.append(
                    f'vertex_proxy_requests_total{{route="{route}",model="{model}",status="{status}"}} {count}'
                )
            lines.append("# HELP vertex_proxy_tokens_in_total Prompt tokens forwarded")
            lines.append("# TYPE vertex_proxy_tokens_in_total counter")
            for model, count in self._tokens_in.items():
                lines.append(f'vertex_proxy_tokens_in_total{{model="{model}"}} {count}')
            lines.append("# HELP vertex_proxy_tokens_out_total Completion tokens returned")
            lines.append("# TYPE vertex_proxy_tokens_out_total counter")
            for model, count in self._tokens_out.items():
                lines.append(f'vertex_proxy_tokens_out_total{{model="{model}"}} {count}')
        return "\n".join(lines) + "\n"


_METRICS = _Metrics()


# --- app factory ------------------------------------------------------------


def build_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or load_settings()
    token_mgr = TokenManager(
        credentials_path=cfg.credentials_path,
        refresh_seconds=cfg.token_refresh_seconds,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        await token_mgr.start()
        # Resolve project ID from credentials if not explicitly configured.
        if cfg.project_id is None:
            cfg.project_id = token_mgr.project_id
        if not cfg.project_id:
            raise RuntimeError(
                "no GCP project_id: set VERTEX_PROXY_PROJECT_ID "
                "or use a service-account key that includes project_id"
            )
        logger.info("vertex-proxy ready; project=%s", cfg.project_id)
        app.state.token_mgr = token_mgr
        app.state.cfg = cfg
        app.state.http = httpx.AsyncClient(timeout=DEFAULT_HTTP_TIMEOUT)
        try:
            yield
        finally:
            await app.state.http.aclose()
            await token_mgr.stop()

    app = FastAPI(
        title="vertex-proxy",
        description="Anthropic + Gemini API-compatible proxy for Google Cloud Vertex AI",
        version="0.1.0",
        lifespan=lifespan,
    )

    # --- optional bearer-token auth on the proxy itself ------------------------
    # When VERTEX_PROXY_API_KEY is set, every non-health route requires it.
    # Use when exposing the proxy on a LAN or reverse-proxying to the internet.
    bearer = HTTPBearer(auto_error=False)

    async def require_api_key(
        creds: HTTPAuthorizationCredentials | None = Depends(bearer),  # noqa: B008
    ) -> None:
        if not cfg.api_key:
            return  # auth not required
        if creds is None or creds.credentials != cfg.api_key:
            raise HTTPException(
                status_code=401,
                detail="missing or invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # --- health ----------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, Any]:
        try:
            # Try to get a token; proves auth is working.
            await token_mgr.get_token()
            return {"status": "ok", "project": cfg.project_id}
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": str(exc)},
            )

    # --- metrics (Prometheus, opt-in) ----------------------------------------

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        if not cfg.metrics_enabled:
            raise HTTPException(
                status_code=404,
                detail="metrics disabled; set VERTEX_PROXY_METRICS_ENABLED=true to enable",
            )
        return PlainTextResponse(_METRICS.render(), media_type="text/plain; version=0.0.4")

    @app.get("/v1/models", dependencies=[Depends(require_api_key)])
    @app.get("/anthropic/v1/models", dependencies=[Depends(require_api_key)])
    @app.get("/anthropic/models", dependencies=[Depends(require_api_key)])
    @app.get("/openai/v1/models", dependencies=[Depends(require_api_key)])
    @app.get("/openai/models", dependencies=[Depends(require_api_key)])
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": alias,
                    "object": "model",
                    "vertex_model_id": real,
                    "provider": "anthropic-vertex",
                    "region": cfg.anthropic_region,
                }
                for alias, real in cfg.anthropic_model_aliases.items()
            ]
            + [
                {
                    "id": alias,
                    "object": "model",
                    "vertex_model_id": real,
                    "provider": "gemini-vertex",
                    "region": cfg.gemini_region,
                }
                for alias, real in cfg.gemini_model_aliases.items()
            ]
            + [
                {
                    "id": alias,
                    "object": "model",
                    "vertex_model_id": path,
                    "provider": "maas-vertex",
                    "region": cfg.maas_region,
                }
                for alias, path in cfg.maas_model_aliases.items()
            ],
        }

    # --- Anthropic routes ------------------------------------------------------

    @app.post("/anthropic/v1/messages", dependencies=[Depends(require_api_key)])
    async def anthropic_messages(request: Request) -> Any:
        return await _handle_anthropic(request, cfg, token_mgr)

    # Also accept /v1/messages directly (some clients won't let you override path).
    @app.post("/v1/messages", dependencies=[Depends(require_api_key)])
    async def anthropic_messages_root(request: Request) -> Any:
        return await _handle_anthropic(request, cfg, token_mgr)

    # --- Gemini routes ---------------------------------------------------------
    # Gemini SDK hits /v1beta/models/{model}:generateContent and :streamGenerateContent.
    # We pass-through both.

    @app.post(
        "/gemini/v1beta/models/{model_and_action:path}", dependencies=[Depends(require_api_key)]
    )
    async def gemini_generate(model_and_action: str, request: Request) -> Any:
        return await _handle_gemini(model_and_action, request, cfg, token_mgr)

    @app.post("/v1beta/models/{model_and_action:path}", dependencies=[Depends(require_api_key)])
    async def gemini_generate_root(model_and_action: str, request: Request) -> Any:
        return await _handle_gemini(model_and_action, request, cfg, token_mgr)

    # --- OpenAI-compatible route for Vertex MaaS models ------------------------
    # Kimi K2.5, GLM 5, MiniMax-M2.5, Qwen 3.5, Grok 4.20, etc.
    # Vertex exposes these through an OpenAI Chat Completions-compatible
    # endpoint at /v1beta1/.../endpoints/openapi/chat/completions.

    @app.post("/openai/v1/chat/completions", dependencies=[Depends(require_api_key)])
    async def openai_chat_completions(request: Request) -> Any:
        return await _handle_openai(request, cfg, token_mgr)

    @app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
    async def openai_chat_completions_root(request: Request) -> Any:
        return await _handle_openai(request, cfg, token_mgr)

    # Some OpenAI clients (notably Hermes's internal one) drop the /v1 prefix
    # when you set base_url to the server root. Accept that shape too.
    @app.post("/chat/completions", dependencies=[Depends(require_api_key)])
    async def openai_chat_completions_bare(request: Request) -> Any:
        return await _handle_openai(request, cfg, token_mgr)

    # When base_url is set to /openai (without /v1), the OpenAI client appends
    # /chat/completions directly, producing /openai/chat/completions.
    @app.post("/openai/chat/completions", dependencies=[Depends(require_api_key)])
    async def openai_chat_completions_no_v1(request: Request) -> Any:
        return await _handle_openai(request, cfg, token_mgr)

    # /v1/models/{model} — some clients probe for a specific model's existence
    # before dispatching. Return minimal metadata so they don't bail.
    @app.get("/v1/models/{model_id:path}")
    async def get_model(model_id: str) -> dict[str, Any]:
        if (
            model_id in cfg.anthropic_model_aliases
            or model_id in cfg.gemini_model_aliases
            or model_id in cfg.maas_model_aliases
            or model_id.startswith("google/")
        ):
            return {"id": model_id, "object": "model", "owned_by": "vertex-proxy"}
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")

    return app


# --- Anthropic handler ------------------------------------------------------


async def _handle_anthropic(request: Request, cfg: Settings, tm: TokenManager) -> Any:
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="request body must be JSON") from exc

    requested_model = (body.get("model") or "").strip()
    if not requested_model:
        raise HTTPException(status_code=400, detail="missing 'model' in request body")

    # Alias resolution.
    vertex_model = cfg.anthropic_model_aliases.get(requested_model, requested_model)
    if vertex_model == requested_model and requested_model not in cfg.anthropic_model_aliases:
        # Unknown model -- not in aliases and not an exact Vertex model ID.
        raise HTTPException(
            status_code=400,
            detail=f"unknown anthropic model '{requested_model}'. "
            f"known aliases: {sorted(cfg.anthropic_model_aliases.keys())}",
        )

    # Anthropic-on-Vertex wants `anthropic_version` and removes `model`.
    upstream_body = {k: v for k, v in body.items() if k != "model"}
    upstream_body.setdefault("anthropic_version", "vertex-2023-10-16")

    streaming = bool(body.get("stream"))
    # Vertex endpoint: :streamRawPredict for streaming, :rawPredict for one-shot.
    action = "streamRawPredict" if streaming else "rawPredict"
    base = _vertex_base_url(cfg.anthropic_region)
    url = (
        f"{base}/v1/projects/"
        f"{cfg.project_id}/locations/{cfg.anthropic_region}/publishers/anthropic/"
        f"models/{vertex_model}:{action}"
    )

    token = await tm.get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    logger.info(
        "anthropic: model=%s → vertex_model=%s streaming=%s",
        requested_model,
        vertex_model,
        streaming,
    )

    http: httpx.AsyncClient = request.app.state.http
    if streaming:
        _METRICS.record_request("anthropic", requested_model, 200)
        return StreamingResponse(
            _stream_bytes(http, url, headers, upstream_body),
            media_type="text/event-stream",
        )

    try:
        resp = await http.post(url, headers=headers, json=upstream_body)
    except httpx.HTTPError as exc:
        logger.error("anthropic upstream error: %s", exc)
        raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc

    return _passthrough_response(resp, route="anthropic", model=requested_model)


# --- OpenAI → Anthropic translation -----------------------------------------
# Translates OpenAI Chat Completions format to Anthropic Messages format
# so that Claude models can be called via the /openai endpoint. This lets
# clients that only speak OpenAI wire (e.g. Hermes auxiliary tasks) use
# Claude without needing the /anthropic endpoint directly.


async def _handle_openai_to_anthropic(
    body: dict[str, Any],
    requested_model: str,
    cfg: Settings,
    tm: TokenManager,
    request: Request,
) -> Any:
    """Translate an OpenAI Chat Completions request to Anthropic Messages format."""

    # Resolve model alias.
    vertex_model = cfg.anthropic_model_aliases.get(requested_model, requested_model)

    # Extract system messages and convert to Anthropic system parameter.
    messages = body.get("messages", [])
    system_parts: list[str] = []
    anthropic_messages: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_parts.append(part["text"])
                    elif isinstance(part, str):
                        system_parts.append(part)
        elif role in ("user", "assistant"):
            anthropic_messages.append({"role": role, "content": content})

    # Build Anthropic request body.
    anthropic_body: dict[str, Any] = {
        "model": requested_model,
        "messages": anthropic_messages,
        "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens") or 4096,
    }
    if system_parts:
        anthropic_body["system"] = "\n\n".join(system_parts)
    if body.get("temperature") is not None:
        anthropic_body["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        anthropic_body["top_p"] = body["top_p"]
    if body.get("stream"):
        anthropic_body["stream"] = True

    # Forward to the Anthropic handler path.
    upstream_body = {k: v for k, v in anthropic_body.items() if k != "model"}
    upstream_body.setdefault("anthropic_version", "vertex-2023-10-16")

    streaming = bool(body.get("stream"))
    action = "streamRawPredict" if streaming else "rawPredict"
    base = _vertex_base_url(cfg.anthropic_region)
    url = (
        f"{base}/v1/projects/"
        f"{cfg.project_id}/locations/{cfg.anthropic_region}/publishers/anthropic/"
        f"models/{vertex_model}:{action}"
    )

    token = await tm.get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    logger.info(
        "openai→anthropic: model=%s → vertex_model=%s streaming=%s",
        requested_model,
        vertex_model,
        streaming,
    )

    http: httpx.AsyncClient = request.app.state.http
    if streaming:
        _METRICS.record_request("openai→anthropic", requested_model, 200)
        # Anthropic SSE format differs from OpenAI -- for now, non-streaming
        # is simpler and auxiliary tasks don't need streaming.
        # TODO: add SSE format translation for full streaming support.

    try:
        resp = await http.post(url, headers=headers, json=upstream_body)
    except httpx.HTTPError as exc:
        logger.error("openai→anthropic upstream error: %s", exc)
        raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc

    if resp.status_code >= 400:
        return JSONResponse(status_code=resp.status_code, content=resp.json())

    # Translate Anthropic response back to OpenAI format.
    anthropic_resp = resp.json()
    content_blocks = anthropic_resp.get("content", [])
    text_parts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
    assistant_text = "".join(text_parts)

    usage = anthropic_resp.get("usage", {})
    openai_response = {
        "id": anthropic_resp.get("id", ""),
        "object": "chat.completion",
        "created": int(__import__("time").time()),
        "model": requested_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": assistant_text},
                "finish_reason": "stop" if anthropic_resp.get("stop_reason") == "end_turn" else "length",
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }

    _METRICS.record_request("openai→anthropic", requested_model, 200)
    _METRICS.record_tokens(
        requested_model,
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )

    return JSONResponse(content=openai_response)


# --- Gemini handler ---------------------------------------------------------


async def _handle_gemini(
    model_and_action: str, request: Request, cfg: Settings, tm: TokenManager
) -> Any:
    # model_and_action is like "gemini-2.5-pro:generateContent" or
    # "gemini-2.5-flash:streamGenerateContent".
    if ":" not in model_and_action:
        raise HTTPException(
            status_code=400,
            detail="gemini path must include action (e.g., ':generateContent')",
        )
    requested_model, action = model_and_action.rsplit(":", 1)
    vertex_model = cfg.gemini_model_aliases.get(requested_model, requested_model)
    streaming = "stream" in action.lower()

    try:
        body = await request.json()
    except Exception:
        body = {}

    base = _vertex_base_url(cfg.gemini_region)
    url = (
        f"{base}/v1/projects/"
        f"{cfg.project_id}/locations/{cfg.gemini_region}/publishers/google/"
        f"models/{vertex_model}:{action}"
    )
    # Pass through query params (e.g., alt=sse).
    if request.url.query:
        url = f"{url}?{request.url.query}"

    token = await tm.get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    logger.info(
        "gemini: model=%s action=%s streaming=%s",
        requested_model,
        action,
        streaming,
    )

    http: httpx.AsyncClient = request.app.state.http
    if streaming:
        _METRICS.record_request("gemini", requested_model, 200)
        return StreamingResponse(
            _stream_bytes(http, url, headers, body),
            media_type="text/event-stream",
        )

    try:
        resp = await http.post(url, headers=headers, json=body)
    except httpx.HTTPError as exc:
        logger.error("gemini upstream error: %s", exc)
        raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc

    return _passthrough_response(resp, route="gemini", model=requested_model)


# --- OpenAI-compatible (Vertex MaaS) handler -------------------------------


async def _handle_openai(request: Request, cfg: Settings, tm: TokenManager) -> Any:
    """Forward OpenAI Chat Completions requests to Vertex AI.

    Supports Claude (auto-translated to Anthropic wire), Gemini (OpenAI-compat),
    and MaaS partner models (Kimi, GLM, MiniMax, Qwen, Grok).
    """
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="request body must be JSON") from exc

    requested_model = (body.get("model") or "").strip()
    if not requested_model:
        raise HTTPException(status_code=400, detail="missing 'model' in request body")

    # --- Claude models: translate OpenAI → Anthropic wire and forward. ---
    # This lets Hermes auxiliary tasks (title_generation, compression, etc.)
    # that always speak OpenAI wire use Claude without explicit config.
    if requested_model in cfg.anthropic_model_aliases or "claude" in requested_model.lower():
        return await _handle_openai_to_anthropic(body, requested_model, cfg, tm, request)

    streaming = bool(body.get("stream"))
    token = await tm.get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # --- routing: Gemini via Vertex OpenAI-compat, or MaaS partner model. ---
    if requested_model in cfg.gemini_model_aliases or requested_model.startswith("google/"):
        # Gemini models through Vertex's OpenAI-compat endpoint.
        # See: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-gemini-using-openai-library
        bare_model = requested_model.removeprefix("google/")
        vertex_model = cfg.gemini_model_aliases.get(bare_model, bare_model)
        gemini_base = _vertex_base_url(cfg.gemini_region)
        url = (
            f"{gemini_base}/v1beta1/projects/"
            f"{cfg.project_id}/locations/{cfg.gemini_region}/endpoints/openapi/chat/completions"
        )
        upstream_body = dict(body)
        upstream_body["model"] = f"google/{vertex_model}"
        logger.info(
            "openai→gemini: model=%s → %s streaming=%s",
            requested_model,
            upstream_body["model"],
            streaming,
        )
    else:
        # MaaS partner models (Kimi, GLM, MiniMax, Qwen, Grok).
        path_fragment = cfg.maas_model_aliases.get(requested_model)
        if path_fragment is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"unknown MaaS model '{requested_model}'. "
                    f"known aliases: {sorted(cfg.maas_model_aliases.keys())} "
                    f"or gemini: {sorted(cfg.gemini_model_aliases.keys())}"
                ),
            )
        maas_base = _vertex_base_url(cfg.maas_region)
        url = (
            f"{maas_base}/v1beta1/projects/"
            f"{cfg.project_id}/locations/{cfg.maas_region}/{path_fragment}/chat/completions"
        )
        upstream_body = dict(body)
        upstream_body["model"] = path_fragment.rsplit("/", 1)[-1]
        logger.info(
            "openai→maas: model=%s → path=%s streaming=%s",
            requested_model,
            path_fragment,
            streaming,
        )

    http: httpx.AsyncClient = request.app.state.http
    if streaming:
        _METRICS.record_request("openai", requested_model, 200)
        return StreamingResponse(
            _stream_bytes(http, url, headers, upstream_body),
            media_type="text/event-stream",
        )

    try:
        resp = await http.post(url, headers=headers, json=upstream_body)
    except httpx.HTTPError as exc:
        logger.error("maas upstream error: %s", exc)
        raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc

    return _passthrough_response(resp, route="openai", model=requested_model)


# --- helpers ----------------------------------------------------------------


async def _stream_bytes(
    http: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
) -> AsyncGenerator[bytes, None]:
    try:
        async with http.stream(
            "POST",
            url,
            headers=headers,
            json=body,
            timeout=STREAM_HTTP_TIMEOUT,
        ) as r:
            if r.status_code >= 400:
                # StreamingResponse has already committed to a 200 status by
                # the time this generator runs, so emit a structured SSE error
                # instead of raising and leaving the client with a broken chunk.
                err_body = b""
                async for chunk in r.aiter_bytes():
                    err_body += chunk
                detail = err_body.decode("utf-8", errors="replace")[:2000]
                logger.warning("upstream stream returned %s: %s", r.status_code, detail)
                yield _stream_error("upstream_http_error", detail, status_code=r.status_code)
                return
            async for chunk in r.aiter_bytes():
                yield chunk
    except httpx.ReadTimeout as exc:
        logger.warning("upstream stream read timeout: %s", exc)
        yield _stream_error(
            "upstream_read_timeout",
            "upstream stream stalled before completion",
        )
    except httpx.HTTPError as exc:
        logger.error("upstream stream error: %s", exc)
        yield _stream_error("upstream_stream_error", str(exc))


def _stream_error(error_type: str, message: str, status_code: int | None = None) -> bytes:
    payload: dict[str, Any] = {
        "error": {
            "type": error_type,
            "message": message[:2000],
        }
    }
    if status_code is not None:
        payload["error"]["status_code"] = status_code
    return f"event: error\ndata: {json.dumps(payload)}\n\n".encode()


def _passthrough_response(resp: httpx.Response, route: str = "", model: str = "") -> JSONResponse:
    """Forward upstream status + JSON body to the client.

    If ``route`` + ``model`` are provided and metrics are enabled, record
    request count + token usage from the OpenAI/Anthropic-style ``usage`` field.
    """
    try:
        payload = resp.json()
    except json.JSONDecodeError:
        # Not JSON; forward as text wrapped.
        payload = {"raw": resp.text[:4000]}

    if route and model:
        _METRICS.record_request(route, model, resp.status_code)
        usage = payload.get("usage") if isinstance(payload, dict) else None
        if isinstance(usage, dict):
            prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            if prompt or completion:
                _METRICS.record_tokens(model, prompt, completion)

    return JSONResponse(status_code=resp.status_code, content=payload)
