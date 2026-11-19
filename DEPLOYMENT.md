# RepoMind deployment

The container setup serves the Vite production build from Nginx and proxies `/api/` to FastAPI. Repository data, temporary uploads, embedding collections, repository summaries, and downloaded Hugging Face models are stored in the persistent `repomind-data` volume. The API container runs one worker so its repository locks and in-memory caches remain process-wide; the runtime lazily prepares analysis and embeddings on first use.

The LLM provider defaults to `mock`, so the application can start without an API key. Investigation will return a clear warning until a provider is configured. Production LLM calls require `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, and `OPENAI_MODEL`; the configured endpoint must satisfy the provider's HTTPS policy. Never commit `.env` or pass secrets as image build arguments.

## Local development

In PowerShell, start the backend from the repository root:

```powershell
$env:LLM_PROVIDER = "mock"
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
Set-Location frontend
npm ci
npm run dev
```

Vite proxies `/api/` requests to `http://127.0.0.1:8000` by default. Set `VITE_API_PROXY_TARGET` to change that development target. Visit `http://127.0.0.1:5173`; the API health check is `http://127.0.0.1:8000/api/v1/health`.

## Docker Compose

From the repository root, build and start the stack:

```powershell
docker compose up --build -d
docker compose ps
```

Open `http://localhost:8080` (or configure `REPOMIND_PORT`). The API health check is available at `http://localhost:8080/api/v1/health` through Nginx. View logs with `docker compose logs -f api frontend` and stop with `docker compose down`. Repository and intelligence data remain in the named volume; `docker compose down -v` deletes it.

To configure the production provider, set these variables in the shell or an untracked root `.env` before `docker compose up`:

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=<secret>
OPENAI_MODEL=<provider-model>
OPENAI_BASE_URL=https://api.openai.com/v1
```

Other supported Compose settings include `LLM_REQUEST_TIMEOUT`, `LLM_MAX_RETRIES`, `MAX_UPLOAD_SIZE_MB`, `MAX_FILE_SIZE_MB`, `MAX_REPOSITORY_FILES`, `REPOMIND_PORT`, and `VITE_API_BASE_URL` (normally empty when using the bundled same-origin Nginx proxy).

The first semantic retrieval request may download the configured Sentence Transformers model into the persistent runtime cache. Ensure the API container has outbound access and sufficient disk/RAM. The default upload allowance is 50 MiB; the frontend proxy permits 55 MiB to allow multipart overhead.

For public deployment, terminate TLS at a trusted reverse proxy or load balancer and restrict its ingress. This Compose file exposes only the frontend service; FastAPI remains on the internal Compose network.
