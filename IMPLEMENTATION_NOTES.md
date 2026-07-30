# Implementation Notes

This build reconciles the following private project documents:

1. `Project_Brief_and_Interview_Script.pdf`
2. `Build Guide_ AI-Moderated WhatsApp Interview System - Private and confidential - For guidance only.pdf`
3. `Adaptive_Interviewer_Guide_Plain_English.pdf`

## Implemented

- Final consent gate and fixed Anchor 1-4, catch-all, wrap-up, and close flow.
- Llama-first moderation for every substantive response when
  `LLM_MODE=always`.
- Structured, enum-validated research coding.
- Required answer-grounded neutral reflections and bounded adaptive
  clarity, drill-down, and tension probes. Each anchor decision records whether
  a probe is needed, its type, its reason, and whether it was actually asked.
  Sparse, generic, unsafe, or ungrounded model output fails closed to the local
  classifier.
- Deterministic question order, a configurable two-probe-per-anchor cap,
  `stop`, consent, demographics, and provider failure fallback. A second probe
  is permitted only for a distinct unresolved gap; the engine then advances
  regardless.
- Prior substantive context supplied to the shared LLM orchestrator.
- Explicit `analysis_source` trace so researchers can distinguish LLM output
  from local fallback.
- Streamlit/FastAPI timeout budgets aligned with NVIDIA and Speechmatics.
- Production FastAPI requests protected by a shared server-to-server token,
  while the hosting health check remains public.
- Cloud deployment files for a pinned, single-worker FastAPI service and
  Streamlit-only secret configuration.
- Supabase REST authentication supports both legacy service-role JWTs and
  newer rotatable `sb_secret_` keys.
- Final Swahili voice-cue rules supplied separately by the project team.
- Deterministic job/income-loss recognition, with `Pole` reserved for explicit
  material hardship and `Sawa` suppressed when a grounded reflection is shown.
- A one-time wrap-up elaboration prompt so a bare affirmative is not mistaken
  for the participant's final substantive comment.

## Deliberately not activated

- The proposed `anchor_3b` question, “What, if anything, worries you about
  relying on AI for this?”, because the adaptive companion explicitly marks it
  as requiring team sign-off.
- WhatsApp transport, because its live identifiers and tokens were not supplied.
- Full-context post-interview batch audit, which remains a reported but unrun
  operational step.
- Any Supabase schema migration, cleanup, or deletion.

## Decisions requiring owner confirmation

- The final project brief names Claude Haiku 4.5. This demo uses NVIDIA-hosted
  Llama 3.1 70B because the project owner explicitly selected and configured
  that provider for the demo.
- The Swahili guide requires `Safi sana!` for positive milestones and `Naam` for
  complex points. The later adaptive guide says never validate or agree. This
  build prevents the LLM from generating validating language but retains the
  explicitly required hard-coded Swahili cues pending owner confirmation.
