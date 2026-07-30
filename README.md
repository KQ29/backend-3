# Otermans Kenya AI Interviewer Demo

A working standalone Streamlit demonstration of an adaptive qualitative
interviewer for Kenyan AI-literacy training alumni. An optional FastAPI adapter
remains available for integrations, but the demo UI needs no separate backend.

For each browser session, the demo uses:

- An in-memory repository for interview state, turns, and tags. Cloud sessions
  are non-persistent and must be exported before the Streamlit session ends.
- Optional Speechmatics Batch transcription for recorded or uploaded voice
  answers.
- NVIDIA NIM's OpenAI-compatible API with
  `meta/llama-3.1-70b-instruct` as the primary moderator for substantive
  interview answers.

Provider credentials are entered through masked Streamlit fields, verified, and
held only in that session's server memory. They are not stored in the repository,
Streamlit Secrets, interview records, or exports. Tests use mock HTTP transports
and consume no external quota.
An optional Supabase repository adapter remains in the codebase but is disabled
in the supplied local and cloud configurations.

See `IMPLEMENTATION_NOTES.md` for the exact brief/guidance decisions applied to
this version and the proposed protocol changes deliberately left disabled.
See `DEPLOYMENT.md` for the Streamlit-only cloud workflow and privacy boundaries.

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
| `streamlit_app.py` | Standalone researcher-facing demo and chat interface |
| `app/api` | FastAPI HTTP contract and dependency wiring |
| `app/services` | Session runtime and atomic transcript/tag handling |
| `app/interview` | Protocol state, LLM moderation handoff, fallback classifier, Swahili cues |
| `app/repositories` | Supabase/PostgREST and in-memory repositories |
| `app/providers` | NVIDIA-compatible LLM, Speechmatics STT, and mock adapters |
| `app/workers` | 10-hour nudge and 24-hour abandonment pass |

The interview engine is channel-independent. Streamlit calls the interview
service directly with one isolated in-memory runtime per browser session. A
future WhatsApp webhook can use the optional FastAPI adapter and the same
service methods.

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
python scripts/run_demo.py
```

Open [http://127.0.0.1:8502](http://127.0.0.1:8502), enter fresh NVIDIA and
optional Speechmatics credentials in the masked form, and select **Verify and
continue**. The launcher starts only Streamlit and removes inherited provider
credentials from its child environment.

If port 8502 is occupied, choose another one for that command:

```bash
STREAMLIT_PORT=8503 python scripts/run_demo.py
```

## Cloud deployment

Deploy `streamlit_app.py` directly on Streamlit Community Cloud with Python
3.13. Leave the Secrets field empty: each operator supplies provider
credentials at runtime. No Render, FastAPI service, backend URL, shared token,
or database is required. Follow `DEPLOYMENT.md` for the exact workflow and
privacy boundaries.

### Optional FastAPI adapter

Integrations that need an HTTP contract can still run:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

API documentation is then available at
[http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs).

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

## Optional API contract

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

The suite covers the standalone runtime, API contract, every research branch,
session isolation, credential redaction, bounded probes,
consent privacy, localized vocabulary, input modes, transcript/tag integrity,
inactivity behavior, Supabase serialization, NVIDIA-compatible structured
moderation, dynamic probes, FastAPI handoff, and the
Speechmatics job lifecycle.

## Configuration

The standalone Streamlit demo deliberately does not read provider credentials
from `.env` or Streamlit Secrets. It builds an explicit, isolated runtime from
the masked credential form. Fixed provider endpoints prevent a user from
redirecting credentials to an arbitrary host.

`.env.example` documents configuration for the optional FastAPI adapter and
provider smoke script. The legacy `NVIDIA_API_KEY`, `NVIDIA_MODEL`,
`SPEECHMATICS_API_KEY`, and `MIN_CONFIDENCE_FOR_GEMINI` names remain accepted
there as aliases.

For a live-Llama session through the optional FastAPI adapter, use:

```dotenv
LLM_ENABLED=true
LLM_MODE=always
LLM_MAX_CALLS_PER_SESSION=16
LLM_MAX_OUTPUT_TOKENS=512
MAX_PROBES_PER_ANCHOR=2
```

The standalone Streamlit runtime fixes the probe limit at two and builds its
provider settings from the credential form instead of these variables.

In the standalone runtime, secrets use Pydantic `SecretStr` where applicable,
structured logging redacts provider-key fields, and JSON export is limited to
the interview record. `.env` remains gitignored for optional API development.

### Provider checks

The default smoke command performs read-only connectivity checks for enabled
providers. It lists NVIDIA models and at most one Speechmatics job; Supabase is
reported as disabled by the supplied configuration. It prints no credentials
or database rows.

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
- The supplied cloud deployment stores sessions only in process memory. A
  refresh, disconnect, restart, or redeploy can remove them, so completed JSON
  must be downloaded first.
- The optional Supabase adapter uses an existing schema; no migration,
  retention policy, cleanup operation, or atomic multi-table RPC is included.
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
