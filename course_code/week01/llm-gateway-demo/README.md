# LLM Gateway Demo

Local-first LLM Gateway using a canonical request/event model and provider adapters.

## Security boundary

This repository must never contain provider API keys or other credentials. Do not create
`.env` files, secret files, credential fixtures, request dumps, or logs containing
authorization headers in this workspace.

Local provider credentials are resolved at runtime from Windows Credential Manager. The
gateway only keeps a credential in memory for the duration of an outbound request.

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
```
