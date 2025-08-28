# Local MCP Example: LangGraph Client + FastMCP Server + Ollama

This repository demonstrates a minimal local setup where a Python client (LangChain/LangGraph) connects to a local Model Context Protocol (MCP) server (implemented using `fastmcp`) and uses an Ollama LLM (`qwen3:30b`).

- MCP server exposes a single example tool: `add_numbers(num1, num2)`.
- Client connects to the MCP server over streamable HTTP, discovers tools dynamically, and streams updates while responding in the user's language (defaults to English).
- All components run via Docker Compose. Ollama data (models, keys, history) are persisted under `./ollama_data`.

## Prerequisites
- Docker (latest) and Docker Compose (v2)
- ~20–30 GB free disk space to download and run `qwen3:30b` locally (you can change the model if needed)

## Quick Start

1) Clone the repository

```
git clone https://github.com/claudobahn/langgraph-mcp-ollama-example
cd langgraph-mcp-ollama-example
```

2) Start services (Ollama + MCP server)

```
docker compose up -d --build ollama mcp-server
```

3) Pull the model inside the Ollama container (first-time only)

```
docker exec -it ollama ollama pull qwen3:30b
```

- You can check that Ollama is healthy from the host:

```
curl http://localhost:11434/api/tags
```

## Run the client interactively

Option A: interactive prompt
```
docker compose run --rm client
# Then type your prompt at the terminal
```

Option B: non-interactive prompt (pipe input)
```
echo "Montre un exemple d'utilisation d'outil." | docker compose run -T --rm client
```

The client connects to:
- MCP server at `http://mcp-server:13744/mcp`
- Ollama at `http://ollama:11434`

By default, the client uses `qwen3:30b` with reasoning mode enabled and responds in the user's language (defaults to English).

## What’s Running

- Ollama service
  - Image: `ollama/ollama`
  - Port: `127.0.0.1:11434` exposed on host
  - Persistent data: `./ollama_data` mounted to `/root/.ollama`

- MCP Server (FastMCP)
  - Defined in `mcp-server/server.py`
  - Runs `FastMCP("math")` and exposes one tool:
    - `add_numbers(num1: int, num2: int) -> int`
  - Serves via `transport="streamable-http"` on `0.0.0.0:13744`

- Client (LangGraph + MCP tools)
  - Defined in `client/client.py`
  - Discovers MCP tools with `load_mcp_tools(session)`
  - LLM: `ChatOllama` (model `qwen3:30b`, temperature 0.8, reasoning mode)
  - Streams updates using `langgraph.prebuilt.create_react_agent`

## Repository Structure

```
langgraph-mcp-ollama-example/
├─ client/
│  ├─ Dockerfile
│  ├─ client.py
│  └─ requirements.txt
├─ mcp-server/
│  ├─ Dockerfile
│  ├─ requirements.txt
│  └─ server.py
├─ docker-compose.yml
├─ README.md
└─ ollama_data/
   ├─ models/ ... (persisted by Ollama)
   └─ ...
```

## Common Tasks

- Rebuild and restart everything:
```
docker compose down
docker compose up -d --build
```

- Update/pull a different model (example: `llama3.1`):
```
docker exec -it ollama ollama pull llama3.1
```
Then update `model=` in `client/client.py` if you want to make it default.

## Troubleshooting

- Client fails to connect to Ollama
  - Ensure the `ollama` container is healthy: `docker ps` and `curl http://localhost:11434/api/tags`
  - Pull the required model: `docker exec -it ollama ollama pull qwen3:30b`

- Client fails to connect to MCP server
  - Ensure `mcp-server` is running: `docker compose ps`
  - Logs: `docker compose logs -f mcp-server`

- Startup ordering
  - If you get connection errors running the client too soon, start `ollama` and `mcp-server` first, wait until healthy/ready, then run the client.

## License
This project is provided as-is, without any warranty, under the "MIT No Attribution" license.

## Notes
- You can customize prompts, the system message language, and model parameters in `client/client.py`.
