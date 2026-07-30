from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


API_BASE_URL = os.getenv(
    "STREAMLIT_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")
BACKEND_API_TOKEN = os.getenv("BACKEND_API_TOKEN", "").strip()
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
STT_TIMEOUT_SECONDS = float(os.getenv("STT_TIMEOUT_SECONDS", "60"))
MINIMUM_API_TIMEOUT_SECONDS = max(
    120.0,
    LLM_TIMEOUT_SECONDS * 3 + 15,
)
API_TIMEOUT_SECONDS = max(
    MINIMUM_API_TIMEOUT_SECONDS,
    float(
        os.getenv(
            "STREAMLIT_API_TIMEOUT_SECONDS",
            str(MINIMUM_API_TIMEOUT_SECONDS),
        )
    ),
)
MINIMUM_VOICE_TIMEOUT_SECONDS = max(
    180.0,
    API_TIMEOUT_SECONDS + STT_TIMEOUT_SECONDS + 30,
)
VOICE_TIMEOUT_SECONDS = max(
    MINIMUM_VOICE_TIMEOUT_SECONDS,
    float(
        os.getenv(
            "STREAMLIT_VOICE_TIMEOUT_SECONDS",
            str(MINIMUM_VOICE_TIMEOUT_SECONDS),
        )
    ),
)
ACTIVE_STATUSES = {"collecting_demographics", "in_progress"}
TERMINAL_STATUSES = {"completed", "declined", "stopped", "abandoned"}


class DemoApiError(RuntimeError):
    pass


def backend_request_headers() -> dict[str, str]:
    if not BACKEND_API_TOKEN:
        return {}
    return {"X-Backend-Token": BACKEND_API_TOKEN}


def api_call(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        with httpx.Client(
            base_url=API_BASE_URL,
            timeout=API_TIMEOUT_SECONDS,
            trust_env=False,
        ) as client:
            response = client.request(
                method,
                path,
                json=payload,
                headers=backend_request_headers(),
            )
    except httpx.TimeoutException as exc:
        raise DemoApiError(
            "The AI provider took longer than the configured wait time. The "
            "backend may still finish this turn; wait briefly and refresh before "
            "submitting the answer again."
        ) from exc
    except httpx.HTTPError as exc:
        raise DemoApiError(
            "The configured FastAPI backend is unavailable. Verify its deployment "
            "and STREAMLIT_API_URL, then try again."
        ) from exc

    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise DemoApiError(f"API request failed ({response.status_code}): {detail}")
    return response.json()


def api_upload(
    path: str,
    *,
    filename: str,
    content: bytes,
    mime_type: str,
) -> dict[str, Any]:
    try:
        with httpx.Client(
            base_url=API_BASE_URL,
            timeout=VOICE_TIMEOUT_SECONDS,
            trust_env=False,
        ) as client:
            response = client.post(
                path,
                files={"audio": (filename, content, mime_type)},
                data={"language_hint": "auto"},
                headers=backend_request_headers(),
            )
    except httpx.TimeoutException as exc:
        raise DemoApiError(
            "Voice transcription or AI moderation took longer than the configured "
            "wait time. The backend may still finish; wait briefly and refresh "
            "before submitting the recording again."
        ) from exc
    except httpx.HTTPError as exc:
        raise DemoApiError(
            "The configured FastAPI backend is unavailable or voice processing "
            "timed out."
        ) from exc

    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise DemoApiError(f"Voice request failed ({response.status_code}): {detail}")
    return response.json()


def start_interview() -> None:
    response = api_call(
        "POST",
        "/api/v1/interviews/start",
        payload={"channel": "streamlit_demo"},
    )
    st.session_state.session_id = response["session_id"]
    st.session_state.last_error = None


def submit_consent(choice: str) -> None:
    session_id = st.session_state.session_id
    api_call(
        "POST",
        f"/api/v1/interviews/{session_id}/consent",
        payload={"choice": choice},
    )
    st.session_state.last_error = None


def submit_answer(text: str) -> None:
    session_id = st.session_state.session_id
    api_call(
        "POST",
        f"/api/v1/interviews/{session_id}/text",
        payload={"text": text},
    )
    st.session_state.last_error = None


def submit_audio(audio_file: Any) -> None:
    session_id = st.session_state.session_id
    api_upload(
        f"/api/v1/interviews/{session_id}/voice",
        filename=getattr(audio_file, "name", "voice_note.wav"),
        content=audio_file.getvalue(),
        mime_type=getattr(audio_file, "type", None) or "audio/wav",
    )
    st.session_state.last_error = None


def load_state() -> dict[str, Any] | None:
    session_id = st.session_state.get("session_id")
    if not session_id:
        return None
    return api_call("GET", f"/api/v1/interviews/{session_id}/state")


def humanize(value: str | None) -> str:
    if not value:
        return "—"
    return value.replace("_", " ").title()


def render_header() -> None:
    st.markdown(
        """
        <div class="brand-card">
          <div class="eyebrow">OTERMANS INSTITUTE · KENYA</div>
          <h1>AI Interviewer Demo</h1>
          <p>Adaptive, research-led conversations for training outcomes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_transcript(
    state: dict[str, Any],
    health: dict[str, Any] | None,
) -> None:
    for turn in state["transcript"]:
        role = "assistant" if turn["role"] == "assistant" else "user"
        with st.chat_message(role):
            st.markdown(turn["content"])
            if role == "user" and turn.get("input_mode") == "voice":
                provider = humanize(
                    str((health or {}).get("stt_provider", "configured STT"))
                )
                st.caption(f"Voice note · transcribed by {provider}")


def render_sidebar(state: dict[str, Any] | None, health: dict[str, Any] | None) -> None:
    with st.sidebar:
        st.markdown("### Demo controls")
        if health:
            st.success("FastAPI connected")
            st.caption(
                f"Storage: {humanize(str(health.get('repository')))} · "
                f"STT: {humanize(str(health.get('stt_provider')))}"
            )
            st.caption(f"LLM: {health.get('llm_model', 'disabled')}")
            st.caption(
                "Moderation: "
                f"{humanize(str(health.get('llm_mode', 'fallback')))} · "
                f"up to {health.get('llm_max_calls_per_session', '—')} calls · "
                f"{health.get('max_probes_per_anchor', '—')} probes/anchor"
            )
        else:
            st.error("FastAPI offline")

        if state:
            st.progress(state["progress"] / 100, text=f"{state['progress']}% complete")
            left, right = st.columns(2)
            left.metric("Status", humanize(state["status"]))
            right.metric("Input", humanize(state["mode_of_input"]))
            st.caption(f"Session · {state['session_id'][:8]}…")

            st.markdown("#### Interview trace")
            st.write("Current question", humanize(state["question_id"]))
            st.write(
                "Anchors covered",
                ", ".join(humanize(item) for item in state["anchors_covered"]) or "—",
            )
            probe_counts = state.get("probe_counts") or {}
            probe_usage = ", ".join(
                (
                    f"{humanize(anchor)}: {count}/"
                    f"{state.get('max_probes_per_anchor', 1)}"
                )
                for anchor, count in sorted(probe_counts.items())
            )
            st.write("Probe usage", probe_usage or "—")
            st.write("Mixed evidence", "Yes" if state["mixed_evidence"] else "No")
            st.write("External AI calls", state["llm_calls_used"])
            if state.get("tags"):
                latest_tag = state["tags"][-1]
                metadata = latest_tag.get("metadata") or {}
                st.markdown("#### Last answer analysis")
                st.write(
                    "Source",
                    humanize(metadata.get("analysis_source")),
                )
                st.write(
                    "Polarity",
                    humanize(latest_tag.get("polarity")),
                )
                st.write(
                    "Confidence",
                    f"{float(latest_tag.get('confidence_in_tagging', 0)):.0%}",
                )
                st.write(
                    "Follow-up needed",
                    "Yes" if metadata.get("needs_probe") else "No",
                )
                st.write(
                    "Probe type",
                    humanize(
                        metadata.get("probe_type")
                        or metadata.get("probe_strategy")
                    ),
                )
                if metadata.get("probe_reason"):
                    st.write(
                        "Decision reason",
                        humanize(metadata["probe_reason"]),
                    )
                if metadata.get("needs_probe"):
                    probe_number = metadata.get("probe_number")
                    asked_label = "Yes" if metadata.get("probe_asked") else "No"
                    if probe_number:
                        asked_label = (
                            f"{asked_label} ({probe_number}/"
                            f"{state.get('max_probes_per_anchor', 1)})"
                        )
                    st.write(
                        "Follow-up asked",
                        asked_label,
                    )
                if metadata.get("llm_reflection"):
                    st.write(
                        "Grounded reflection",
                        metadata["llm_reflection"],
                    )
                if metadata.get("llm_suggested_probe"):
                    st.write(
                        "Adaptive probe",
                        metadata["llm_suggested_probe"],
                    )

            if st.button("Start a new interview", use_container_width=True):
                start_interview()
                st.rerun()

            if state["status"] in ACTIVE_STATUSES and st.button(
                "Stop interview",
                type="secondary",
                use_container_width=True,
            ):
                api_call(
                    "POST",
                    f"/api/v1/interviews/{state['session_id']}/stop",
                )
                st.rerun()

            try:
                export_data = api_call(
                    "GET",
                    f"/api/v1/interviews/{state['session_id']}/export",
                )
            except DemoApiError as exc:
                st.warning(f"Research record export is temporarily unavailable: {exc}")
            else:
                st.download_button(
                    "Download research record",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"interview-{state['session_id'][:8]}.json",
                    mime="application/json",
                    use_container_width=True,
                )
        else:
            st.caption("Start an interview to see the live decision trace.")

        st.divider()
        st.caption(
            "Private research demo · Audio is transcribed by the configured STT "
            "provider and interview records use the configured backend."
        )


def main() -> None:
    st.set_page_config(
        page_title="Kenya AI Interviewer",
        page_icon="🇰🇪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
          .stApp { background: #f5f1e9; }
          .block-container { max-width: 920px; padding-top: 2rem; }
          .brand-card {
            padding: 1.8rem 2rem;
            border-radius: 24px;
            color: white;
            background:
              radial-gradient(circle at 92% 18%, #f0b323 0 4%, transparent 4.5%),
              linear-gradient(125deg, #102f25 0%, #176344 62%, #23885d 100%);
            box-shadow: 0 15px 35px rgba(16, 47, 37, .18);
            margin-bottom: 1.2rem;
          }
          .brand-card .eyebrow {
            color: #f1c75b; font-weight: 800; letter-spacing: .12em;
            font-size: .75rem;
          }
          .brand-card h1 { margin: .25rem 0; font-size: 2.35rem; }
          .brand-card p { margin: 0; color: #dcf2e6; }
          [data-testid="stChatMessage"] {
            border-radius: 18px;
            border: 1px solid rgba(23, 99, 68, .12);
            background: rgba(255, 255, 255, .82);
          }
          [data-testid="stSidebar"] { background: #eef3ed; }
          .privacy-note {
            border-left: 4px solid #f0b323;
            background: #fffaf0;
            padding: .7rem 1rem;
            border-radius: 8px;
            color: #5b4b1e;
            margin-bottom: 1rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("session_id", None)
    st.session_state.setdefault("last_error", None)

    health: dict[str, Any] | None = None
    state: dict[str, Any] | None = None
    try:
        health = api_call("GET", "/health")
        state = load_state()
        st.session_state.last_error = None
    except DemoApiError as exc:
        st.session_state.last_error = str(exc)

    render_sidebar(state, health)
    render_header()
    st.markdown(
        """
        <div class="privacy-note">
          This demonstration can use live Speechmatics, NVIDIA Llama and Supabase.
          Audio and answers may leave this browser and be stored for research.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    if not state:
        st.markdown(
            "Experience the full consent gate, demographics, adaptive anchors, "
            "bounded probes, Swahili localization, live AI fallback, real voice "
            "transcription, persistent tags, and JSON export."
        )
        if st.button(
            "Start demo interview",
            type="primary",
            use_container_width=True,
            disabled=health is None,
        ):
            try:
                start_interview()
                st.rerun()
            except DemoApiError as exc:
                st.session_state.last_error = str(exc)
                st.rerun()
        return

    render_transcript(state, health)

    if state["status"] == "awaiting_consent":
        st.caption("Choose one option to continue.")
        yes_col, no_col = st.columns(2)
        if yes_col.button(
            "Yes, continue",
            type="primary",
            use_container_width=True,
        ):
            try:
                submit_consent("consent_yes")
                st.rerun()
            except DemoApiError as exc:
                st.error(str(exc))
        if no_col.button("No, end here", use_container_width=True):
            try:
                submit_consent("consent_no")
                st.rerun()
            except DemoApiError as exc:
                st.error(str(exc))
        return

    if state["status"] in TERMINAL_STATUSES:
        if state["status"] == "declined":
            st.info(
                "Consent was declined. No respondent response or follow-up was stored."
            )
        else:
            st.success(f"Interview {humanize(state['status']).lower()}.")
        return

    mode = st.radio(
        "Response channel",
        ["Text", "Voice note"],
        horizontal=True,
        help="Voice audio is sent to the configured backend speech-to-text provider.",
    )

    if mode == "Text":
        answer = st.chat_input(
            "Type a response, or type “stop” to end the interview",
            max_chars=8000,
        )
        if answer:
            try:
                with st.spinner("Processing the response..."):
                    submit_answer(answer)
                st.rerun()
            except DemoApiError as exc:
                st.error(str(exc))
    else:
        recorded_audio = st.audio_input("Record your answer")
        uploaded_audio = st.file_uploader(
            "Or upload an audio file",
            type=["wav", "mp3", "m4a", "mp4", "ogg", "aac", "flac", "webm"],
        )
        selected_audio = recorded_audio or uploaded_audio
        if selected_audio and st.button(
            "Transcribe and send",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("Transcribing your voice note..."):
                    submit_audio(selected_audio)
                st.rerun()
            except DemoApiError as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
