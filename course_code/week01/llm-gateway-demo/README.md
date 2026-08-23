# LLM Gateway Demo

Local-first LLM Gateway using a canonical request/event model and provider adapters.

## Security boundary

This repository must never contain provider API keys or other credentials. Do not create
`.env` files, secret files, credential fixtures, request dumps, or logs containing
authorization headers in this workspace.

Local provider credentials are resolved at runtime from Windows Credential Manager. The
gateway only keeps a credential in memory for the duration of an outbound request.

Templates and usage records are stored in external application data by default:
`%LOCALAPPDATA%\llm-gateway\gateway.sqlite3`. The gateway rejects a database path inside
this repository.

## Providers

- `local/qwen3.8-27b-q5ks`: local llama.cpp server; no API key is used.
- `openai/<provider-model>`: OpenAI adapter; credential is external.
- `anthropic/<provider-model>`: Claude Messages adapter; credential is external.
- `deepseek/<provider-model>`: DeepSeek Chat Completions adapter; credential is external.

## Run the local llama.cpp server

Start `llama-server.exe` separately with the Qwen GGUF model. Keep it bound to localhost.
The exact binary flags depend on the installed llama.cpp build; verify them with
`llama-server.exe --help`.

For the current local files, a conservative starting command is:

```powershell
& "D:\titan-llama\llama-server.exe" `
  -m "D:\titan-models\qwen\Qwen3.8-27B-Q5_K_S.gguf" `
  --host 127.0.0.1 `
  --port 8080 `
  --ctx-size 4096 `
  --parallel 1
```

The gateway expects the local server at `http://127.0.0.1:8080` by default.

The gateway does not start llama.cpp automatically by default. To opt into process
management, set the non-sensitive process settings in the external execution environment
before starting the gateway:

```powershell
$env:LLM_GATEWAY_MANAGE_LLAMA = "true"
```

The default local profile is conservative: context size 4096, one parallel request, and
batch size 256. GPU layers and CPU threads should be selected only after checking local
hardware and can be supplied through `LLAMA_CPP_GPU_LAYERS` and `LLAMA_CPP_THREADS`.

## Install and run

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,clients]"
.\.venv\Scripts\python.exe -m uvicorn gateway.main:app --host 127.0.0.1 --port 9000
```

No secret is needed for local llama.cpp testing.

## Feature overview

### Streaming SSE

Both completion endpoints accept `stream: true`:

```json
{
  "model": "local/qwen3.8-27b-q5ks",
  "messages": [{"role": "user", "content": "Explain SSE briefly."}],
  "stream": true
}
```

`/v1/chat/completions` emits OpenAI-compatible `data:` chunks and terminates with
`data: [DONE]`. `/v1/responses` emits named response events. Provider errors after
headers have been sent are emitted as an SSE `error` event when possible.

### Structured JSON output

Use `json_object` or `json_schema`:

```json
{
  "model": "local/qwen3.8-27b-q5ks",
  "messages": [{"role": "user", "content": "Return a person object."}],
  "response_format": {"type": "json_object"}
}
```

For a schema, use either the compact form accepted by the local adapter or the nested
OpenAI-style form:

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "person",
    "schema": {
      "type": "object",
      "properties": {"name": {"type": "string"}},
      "required": ["name"],
      "additionalProperties": false
    }
  }
}
```

The gateway validates the final output. Invalid JSON returns the unified
`invalid_json_output` error code instead of silently returning malformed content.

### Prompt templates

Template endpoints:

```text
GET    /v1/templates
POST   /v1/templates
GET    /v1/templates/{name}
PUT    /v1/templates/{name}
DELETE /v1/templates/{name}
POST   /v1/templates/{name}/render
```

Template syntax:

```text
{{ persona }}
{{ user.name }}
{{>base-system}}
```

When a completion request contains `template` and `variables`, the rendered template is
inserted as the first system message. Missing variables and circular references return
stable template error codes.

Example:

```json
{
  "model": "local/qwen3.8-27b-q5ks",
  "template": "support-agent",
  "variables": {"customer": {"name": "Ada"}},
  "messages": [{"role": "user", "content": "Help me."}]
}
```

### Usage, token categories, latency and TTFT

Every completion attempt is recorded in the external SQLite store. Records include:

- prompt/input tokens
- completion/output tokens
- reasoning tokens when the provider reports them
- cached prompt tokens and cache-creation tokens when reported
- total tokens
- total latency in milliseconds
- TTFT (time to first token) for streaming calls
- retry count, status and unified error code

Query recent records and grouped summaries with:

```text
GET /v1/usage
GET /v1/usage?model=local/qwen3.8-27b-q5ks&limit=50
```

### Retry and per-model rate limiting

Transient provider timeouts, 429s and 5xx responses use exponential backoff. The default
is up to 3 retries after the first attempt. Rate limiting is independent for each model.
When the configured model window is exhausted, the gateway returns HTTP `492` with the
unified error code `rate_limit_exceeded` and a `Retry-After` header.

Non-sensitive runtime settings can be injected by the process environment:

```text
LLM_GATEWAY_RATE_LIMIT_REQUESTS
LLM_GATEWAY_RATE_LIMIT_WINDOW_SECONDS
LLM_GATEWAY_MAX_RETRIES
LLM_GATEWAY_RETRY_BASE_DELAY_SECONDS
LLM_GATEWAY_RETRY_MAX_DELAY_SECONDS
LLM_GATEWAY_DATA_PATH
```

No `.env` loader is used.

## Configure external credentials

After installing dependencies, use the interactive command below. It hides input and
writes to Windows Credential Manager, not to this repository:

```powershell
llm-gateway-secrets set openai
llm-gateway-secrets set anthropic
llm-gateway-secrets set deepseek
```

Never pass a credential as a command-line argument.

## API examples

```text
GET  http://127.0.0.1:9000/health
GET  http://127.0.0.1:9000/ready
GET  http://127.0.0.1:9000/v1/models
POST http://127.0.0.1:9000/v1/chat/completions
POST http://127.0.0.1:9000/v1/responses
GET  http://127.0.0.1:9000/v1/templates
GET  http://127.0.0.1:9000/v1/usage
```

## Verification

The verification script uses a fake in-memory provider and a temporary database. It
covers SSE chunks, JSON validation, template variables and references, usage/TTFT,
retry/backoff, model-scoped limiting and HTTP 492. It does not require any provider key:

```powershell
python scripts/verify_features.py
```

The unit tests cover the same core paths plus the llama.cpp adapter:

```powershell
python -m unittest discover -s tests -v
```
