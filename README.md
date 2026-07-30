# Otermans Kenya AI Interviewer Demo

A working FastAPI + Streamlit demonstration of an adaptive qualitative
interviewer for Kenyan AI-literacy training alumni.

With the supplied server-side environment configuration, the application uses:

- Supabase/Postgres for persistent interview state, turns, and tags.
- Speechmatics Batch transcription for recorded or uploaded voice answers.
- NVIDIA NIM's OpenAI-compatible API with
  `meta/llama-3.1-70b-instruct` as the primary moderator for substantive
  interview answers.

Tests use mock HTTP transports and consume no external quota. If credentials are
absent, the application can still run with the memory, mock-STT, and disabled-LLM
configuration documented in `.env.example`.

See `IMPLEMENTATION_NOTES.md` for the exact brief/guidance decisions applied to
this version and the proposed protocol changes deliberately left disabled.

## What the demo proves

- Explicit consent gate. A declined response ends the session and stores no
  respondent turn or follow-up.
- Seven demographic questions, each skippable.
- Four research anchors with deterministic positive, no-change, and mixed
  branching.
- At most two bounded, sequential probes per anchor. The second is asked only
  when a distinct unresolved gap remains.
- Catch-all answer retained verbatim without forced coding.
- Deterministic `stop` handling at any point after consent.
- Text, real voice, and mixed input-mode tracking.
- Sparse Swahili localization using `Karibu`, `Safi sana!`, `Sawa`, `Naam`,
  `Pole`, `Usijali!`, and `Asante sana` under explicit trigger rules.
- Live research tags, transcript export, and a visible decision trace.
- LLM-first interpretation, neutral response-specific reflections, and adaptive
  clarity, drill-down, or tension probes inside deterministic protocol bounds.
- A visible analysis source on every substantive answer, with local-rule
  fallback if the model is unavailable or the configured call cap is reached.
- Explicit job/income-loss handling and a one-time elaboration prompt when the
  respondent answers the final open invitation with only “Yes”.
- One simulated transport nudge at 10 hours and abandonment at 24 hours.
- Replaceable real and mock LLM, STT, transport, and repository adapters.

## Architecture

| Layer | Responsibility |
| --- | --- |
| `streamlit_app.py` | Researcher-facing demo and chat interface |
| `app/api` | FastAPI HTTP contract and dependency wiring |
| `app/services` | Atomic transcript/tag persistence per answer |
| `app/interview` | Protocol state, LLM moderation handoff, fallback classifier, Swahili cues |
| `app/repositories` | Supabase/PostgREST and in-memory repositories |
| `app/providers` | NVIDIA-compatible LLM, Speechmatics STT, and mock adapters |
| `app/workers` | 10-hour nudge and 24-hour abandonment pass |

The interview engine is channel-independent. A future WhatsApp webhook can call
the same service methods used by Streamlit. FastAPI runs the inactivity pass
hourly by default; the interval is configurable for deployment.

Consent, `stop`, demographics, question order, and probe limits remain
deterministic. Llama receives only substantive interview responses, not the
demographic collection turns. It returns structured coding plus a required,
answer-grounded neutral reflection for each on-topic turn and an explicit
`needs_probe`, probe type, and reason decision. When a probe is useful, it
proposes exactly one contextual question at a time. The engine rejects sparse,
generic, repetitive, unsafe, or ungrounded model language before replying and
always advances once the answer is sufficient or the two-probe cap is reached.

## Quick start

Requirements: Python 3.11 or newer.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
cp .env.example .env
# Populate only the server-side provider variables in .env.
python scripts/run_demo.py
```

Open [http://127.0.0.1:8502](http://127.0.0.1:8502). API documentation is
available at [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs).

The launcher starts both processes and stops them together when you press
`Ctrl+C`. It loads `.env` before selecting ports, rejects occupied ports with a
clear error, and waits for `/health` rather than assuming that an open socket
means FastAPI is ready.

### Run in two terminals

Terminal 1:

```bash
cd backend
source .venv/bin/activate
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

Terminal 2:

```bash
cd backend
source .venv/bin/activate
STREAMLIT_API_URL=http://127.0.0.1:8001 \
  python -m streamlit run streamlit_app.py
```

## Suggested demo walkthrough

1. Start an interview and accept consent.
2. Enter the respondent's demographic answers.
3. Answer Anchor 1 with the respondent's experience; the interview will adapt
   its follow-up path to the evidence in the response.
4. Switch to **Voice note**, record or upload an answer, and send it through
   Speechmatics. Raw audio is not stored by this application.
5. Complete the remaining questions and download the JSON research record.

The sidebar exposes current state, anchors covered, probes used, mixed-evidence
status, external AI call count, and the latest answer's actual analysis source,
confidence, polarity, follow-up decision, probe type, and decision reason.

## API contract

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service health |
| `POST` | `/api/v1/interviews/start` | Create a session |
| `POST` | `/api/v1/interviews/{id}/consent` | Accept or decline consent |
| `POST` | `/api/v1/interviews/{id}/text` | Submit a text answer |
| `POST` | `/api/v1/interviews/{id}/voice` | Upload multipart audio for transcription |
| `GET` | `/api/v1/interviews/{id}/state` | Read demo state and transcript |
| `POST` | `/api/v1/interviews/{id}/stop` | Stop deterministically |
| `GET` | `/api/v1/interviews/{id}/export` | Export the complete record |
| `POST` | `/api/v1/internal/interviews/{id}/batch-audit` | Report audit status |

The internal batch-audit endpoint reports `not_run`. With `LLM_MODE=always`,
NVIDIA Llama moderates every substantive turn until the configured per-session
cap. `LLM_MODE=fallback` retains the earlier local-first behaviour.

## Test

```bash
cd backend
source .venv/bin/activate
python -m pytest
python -m compileall -q app streamlit_app.py scripts
```

The suite covers the API contract, every research branch, bounded probes,
consent privacy, localized vocabulary, input modes, transcript/tag integrity,
inactivity behavior, Supabase serialization, NVIDIA-compatible structured
moderation, dynamic probes, FastAPI handoff, client timeout budgeting, and the
Speechmatics job lifecycle.

## Configuration

All runtime configuration comes from environment variables. `.env.example`
documents the supported names. Live providers fail closed if their required
server-side settings are missing. The legacy `NVIDIA_API_KEY`,
`NVIDIA_MODEL`, `SPEECHMATICS_API_KEY`, and
`MIN_CONFIDENCE_FOR_GEMINI` names are accepted as aliases.

For the live-Llama demo, use:

```dotenv
LLM_ENABLED=true
LLM_MODE=always
LLM_MAX_CALLS_PER_SESSION=16
LLM_MAX_OUTPUT_TOKENS=512
MAX_PROBES_PER_ANCHOR=2
STREAMLIT_API_TIMEOUT_SECONDS=120
STREAMLIT_VOICE_TIMEOUT_SECONDS=210
```

The probe limit is captured when an interview starts, so start a new interview
after changing it. Existing stored interviews created before this setting was
introduced retain the previous one-probe limit.

The Streamlit timeout must exceed the possible provider duration. The previous
10-second client timeout could report FastAPI as unavailable even while NVIDIA
later returned `200 OK`; the configured defaults now leave enough time for the
backend's retry budget.

Secrets are represented with Pydantic `SecretStr`. Structured logging redacts
common sensitive keys. `.env` is gitignored, excluded from the handoff archive,
and should have filesystem mode `600`.

### Provider checks

The default smoke command performs read-only connectivity checks: it reads the
active Supabase protocol, lists NVIDIA models, and lists at most one
Speechmatics job. It prints no credentials or database rows.

```bash
python scripts/provider_smoke.py
```

These optional flags create potentially billable provider work:

```bash
python scripts/provider_smoke.py --llm-completion
python scripts/provider_smoke.py --llm-completion \
  --llm-answer "The synthetic Cedar-947 workflow cut weekly reporting from six hours to two."
python scripts/provider_smoke.py --audio path/to/answer.ogg
```

`--llm-answer` requires `--llm-completion`. The completion output includes the
validated analysis source, confidence, routing signals, `needs_probe`, probe
type, decision reason, reflection, and suggested probe so a custom synthetic
answer can demonstrate model-dependent behavior. The script does not print the
submitted answer directly or expose provider credentials. Use only
non-sensitive test text: command arguments can remain in shell history, and
generated reflections or probes may restate parts of the answer.

Speechmatics keys may be region-specific. If the supplied endpoint returns
`401`, set `STT_BASE_URL` to the account's regional `/v2` endpoint.

## Deliberate demo limits

- Heuristic tags are directional demo output, not validated research coding.
- There is no participant authentication or researcher admin role.
- Supabase writes use the existing schema. No migration, retention policy, or
  cleanup operation was applied.
- The existing schema does not expose an atomic multi-table RPC, so a network
  interruption between session, turn, and tag writes may require operational
  reconciliation before production scale.
- The 10-hour nudge uses a logging transport until WhatsApp is configured.
- The optional completed-transcript batch audit is not implemented.
- The companion guide's proposed `anchor_3b` worries question is not enabled;
  that document explicitly requires team sign-off before changing the finalized
  protocol.
- No existing TypeScript source was changed.

Production hardening should add authenticated admin export, a transactional
database RPC, provider idempotency, encrypted PII handling, audit review, key
rotation, explicit retention rules, and a WhatsApp transport only after the
relevant schema and provider choices are approved.
