# Cloud deployment

This project runs as two services:

```text
Browser -> Streamlit Community Cloud -> FastAPI on Render
                                      -> NVIDIA / Speechmatics
```

Streamlit makes server-to-server requests to FastAPI. Provider credentials
belong only on the FastAPI service. The shared `BACKEND_API_TOKEN` belongs on
both services so the public API cannot be used directly by unauthorised callers.

## 0. Rotate exposed credentials first

Do not deploy with any credential that has been pasted into chat, source code,
an issue, or a public log.

1. In Supabase, revoke the previously exposed key. No replacement is needed
   because this deployment does not use Supabase.
2. In NVIDIA NGC, rotate or delete the exposed personal key and create a
   replacement with only the services this demo requires.
3. In Speechmatics, revoke the exposed key and create a replacement.
4. Review each provider's usage logs for unexpected activity.

Never commit the replacements or paste them into chat.

## 1. Deploy FastAPI on Render

The repository includes `render.yaml`, so Render can create the backend from a
Blueprint.

1. Push this repository to GitHub.
2. Sign in to Render and choose **New -> Blueprint**.
3. Connect the GitHub repository and select the `main` branch.
4. Render will detect `render.yaml`. Enter fresh values for every variable it
   marks as secret:

   - `BACKEND_API_TOKEN`
   - `LLM_API_KEY`
   - `SPEECHMATICS_API_KEY`

Generate `BACKEND_API_TOKEN` locally and store it in a password manager:

```bash
openssl rand -hex 32
```

Use the same generated token in Streamlit later. Do not use a provider key as
this token.

Render installs `requirements.txt`, starts one FastAPI process through
`scripts/run_api.py`, binds to the host-provided `PORT`, and checks `/health`.

When deployment completes, open:

```text
https://YOUR-SERVICE.onrender.com/health
```

The response should include:

```json
{
  "status": "ok",
  "repository": "memory",
  "llm_enabled": true,
  "stt_provider": "speechmatics",
  "max_probes_per_anchor": 2
}
```

The deployed backend intentionally uses its in-memory repository and requires
no database configuration. Sessions and interview data disappear whenever the
backend restarts or redeploys.

## 2. Connect Streamlit Community Cloud

Deploy `streamlit_app.py` from the same GitHub repository and choose Python
3.13. In **Advanced settings -> Secrets**, paste:

```toml
STREAMLIT_API_URL = "https://YOUR-SERVICE.onrender.com"
BACKEND_API_TOKEN = "THE-SAME-RANDOM-TOKEN-USED-ON-RENDER"
LLM_TIMEOUT_SECONDS = "30"
STT_TIMEOUT_SECONDS = "60"
STREAMLIT_API_TIMEOUT_SECONDS = "120"
STREAMLIT_VOICE_TIMEOUT_SECONDS = "210"
```

Use the FastAPI base URL without `/health`. Save the secrets and reboot the
Streamlit app.

Do not put NVIDIA or Speechmatics credentials in Streamlit Secrets. Only
Streamlit's server needs the shared backend token.

## 3. Verify the deployed path

1. Confirm the Streamlit sidebar says **FastAPI connected**.
2. Start a new interview and accept consent.
3. Complete one text answer and confirm the analysis source is `LLM`.
4. Submit a non-sensitive test voice note and confirm Speechmatics is shown as
   the transcription provider.
5. Complete the interview, download its JSON export, and confirm probe usage
   never exceeds `2/2`.

If `/health` works but Streamlit receives `401`, the two
`BACKEND_API_TOKEN` values do not match. If an interview disappears after a
Render restart or redeploy, that is expected for the in-memory repository.
