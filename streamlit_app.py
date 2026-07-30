from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, TypeVar

import streamlit as st

from app.services.standalone import (
    StandaloneInterviewRuntime,
    StandaloneRuntimeError,
    create_standalone_runtime,
)

ACTIVE_STATUSES = {"collecting_demographics", "in_progress"}
TERMINAL_STATUSES = {"completed", "declined", "stopped", "abandoned"}
RUNTIME_SESSION_KEY = "_standalone_runtime"
CREDENTIAL_CLEANUP_KEY = "_credential_cleanup_pending"
AUDIO_WIDGET_VERSION_KEY = "_audio_widget_version"
NVIDIA_WIDGET_KEY = "_nvidia_api_key_input"
SPEECHMATICS_WIDGET_KEY = "_speechmatics_api_key_input"
CREDENTIAL_NOTICE_WIDGET_KEY = "_credential_notice_accepted"
CREDENTIAL_WIDGET_KEYS = (
    NVIDIA_WIDGET_KEY,
    SPEECHMATICS_WIDGET_KEY,
    CREDENTIAL_NOTICE_WIDGET_KEY,
)
AUDIO_WIDGET_PREFIXES = ("recorded_audio_", "uploaded_audio_")
T = TypeVar("T")


class DemoRuntimeError(RuntimeError):
    pass


def initialize_session_state() -> None:
    st.session_state.setdefault("session_id", None)
    st.session_state.setdefault("last_error", None)
    st.session_state.setdefault(AUDIO_WIDGET_VERSION_KEY, 0)
    if st.session_state.pop(CREDENTIAL_CLEANUP_KEY, False):
        for key in CREDENTIAL_WIDGET_KEYS:
            st.session_state.pop(key, None)
    current_audio_version = int(
        st.session_state.get(AUDIO_WIDGET_VERSION_KEY, 0)
    )
    active_audio_keys = {
        f"{prefix}{current_audio_version}"
        for prefix in AUDIO_WIDGET_PREFIXES
    }
    for key in list(st.session_state):
        if (
            isinstance(key, str)
            and key.startswith(AUDIO_WIDGET_PREFIXES)
            and key not in active_audio_keys
        ):
            st.session_state.pop(key, None)


def current_runtime() -> StandaloneInterviewRuntime | None:
    runtime = st.session_state.get(RUNTIME_SESSION_KEY)
    return runtime if isinstance(runtime, StandaloneInterviewRuntime) else None


def require_runtime() -> StandaloneInterviewRuntime:
    runtime = current_runtime()
    if runtime is None:
        raise DemoRuntimeError("Enter and verify provider credentials first.")
    return runtime


def runtime_action(action: Callable[[], T]) -> T:
    try:
        return action()
    except StandaloneRuntimeError as exc:
        raise DemoRuntimeError(str(exc)) from None
    except Exception:
        raise DemoRuntimeError(
            "The session backend could not complete this action. Try again or "
            "clear the session and reconnect the providers."
        ) from None


def start_interview() -> None:
    response = runtime_action(lambda: require_runtime().start())
    st.session_state.session_id = response["session_id"]
    st.session_state.last_error = None


def submit_consent(choice: str) -> None:
    session_id = st.session_state.session_id
    runtime_action(lambda: require_runtime().consent(session_id, choice))
    st.session_state.last_error = None


def submit_answer(text: str) -> None:
    session_id = st.session_state.session_id
    runtime_action(lambda: require_runtime().text(session_id, text))
    st.session_state.last_error = None


def submit_audio(audio_file: Any) -> None:
    session_id = st.session_state.session_id
    runtime_action(
        lambda: require_runtime().voice(
            session_id,
            audio=audio_file.getvalue(),
            filename=getattr(audio_file, "name", "voice_note.wav"),
            mime_type=getattr(audio_file, "type", None) or "audio/wav",
            language_hint="auto",
        )
    )
    st.session_state.last_error = None
    st.session_state[AUDIO_WIDGET_VERSION_KEY] += 1


def load_state() -> dict[str, Any] | None:
    session_id = st.session_state.get("session_id")
    if not session_id:
        return None
    return runtime_action(lambda: require_runtime().state(session_id))


def clear_credentials_and_interview_data() -> None:
    runtime = st.session_state.pop(RUNTIME_SESSION_KEY, None)
    if isinstance(runtime, StandaloneInterviewRuntime):
        runtime.clear_credentials()
    st.session_state.session_id = None
    st.session_state.last_error = None
    st.session_state[AUDIO_WIDGET_VERSION_KEY] = (
        int(st.session_state.get(AUDIO_WIDGET_VERSION_KEY, 0)) + 1
    )
    for key in list(st.session_state):
        if isinstance(key, str) and key.startswith(AUDIO_WIDGET_PREFIXES):
            st.session_state.pop(key, None)
    for key in CREDENTIAL_WIDGET_KEYS:
        st.session_state.pop(key, None)


def humanize(value: str | None) -> str:
    if not value:
        return "—"
    return value.replace("_", " ").title()


def analysis_source_warning(analysis_source: str | None) -> str | None:
    if analysis_source == "local_provider_fallback":
        return (
            "The NVIDIA request failed for this answer. Local safety rules "
            "were used instead of the live model."
        )
    if analysis_source == "local_limit_fallback":
        return (
            "The live-model call limit was reached. Local rules were used for "
            "this answer."
        )
    return None


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


def render_provider_setup() -> None:
    st.subheader("Connect your provider credentials")
    st.write(
        "A demo operator must connect NVIDIA before starting. Speechmatics is "
        "optional and enables voice answers."
    )
    st.markdown(
        "[Generate an NVIDIA key]"
        "(https://build.nvidia.com/meta/llama-3_1-70b-instruct)"
        " · [Open the Speechmatics portal](https://portal.speechmatics.com/)"
    )
    with st.form("provider_credentials", clear_on_submit=True):
        nvidia_api_key = st.text_input(
            "NVIDIA API key",
            type="password",
            key=NVIDIA_WIDGET_KEY,
            autocomplete="new-password",
            placeholder="nvapi-…",
            help="Used for live Llama moderation during this browser session.",
        )
        speechmatics_api_key = st.text_input(
            "Speechmatics API key (optional)",
            type="password",
            key=SPEECHMATICS_WIDGET_KEY,
            autocomplete="new-password",
            help="Required only for recorded or uploaded voice answers.",
        )
        notice_accepted = st.checkbox(
            (
                "I understand that these credentials are sent to the Streamlit "
                "server, provider calls may use my quota, and interview content "
                "is sent to the connected providers."
            ),
            key=CREDENTIAL_NOTICE_WIDGET_KEY,
        )
        submitted = st.form_submit_button(
            "Verify and continue",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return
    if not notice_accepted:
        st.error("Accept the credential-processing notice to continue.")
        return
    if not (nvidia_api_key or "").strip():
        st.error("Enter an NVIDIA API key.")
        return

    try:
        with st.spinner(
            "Verifying NVIDIA with one small classification and checking "
            "Speechmatics if supplied..."
        ):
            runtime = create_standalone_runtime(
                nvidia_api_key=nvidia_api_key or "",
                speechmatics_api_key=speechmatics_api_key or None,
                verify=True,
            )
    except StandaloneRuntimeError as exc:
        st.error(str(exc))
        return
    except Exception:
        st.error(
            "Provider verification could not complete. Check the credentials "
            "and try again."
        )
        return

    st.session_state[RUNTIME_SESSION_KEY] = runtime
    st.session_state.session_id = None
    st.session_state.last_error = None
    st.session_state[CREDENTIAL_CLEANUP_KEY] = True
    st.rerun()


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
                    str((health or {}).get("stt_provider", "speechmatics"))
                )
                st.caption(f"Voice note · transcribed by {provider}")


def render_sidebar(state: dict[str, Any] | None, health: dict[str, Any] | None) -> None:
    with st.sidebar:
        st.markdown("### Demo controls")
        if health:
            st.success("Session backend ready")
            st.caption(
                f"Storage: {humanize(str(health.get('repository')))} · "
                f"STT: {humanize(str(health.get('stt_provider')))}"
            )
            st.caption(f"LLM: {health.get('llm_model', 'disabled')}")
            st.caption(
                "Moderation: "
                f"{humanize(str(health.get('llm_mode', 'always')))} · "
                f"up to {health.get('llm_max_calls_per_session', '—')} calls · "
                f"{health.get('max_probes_per_anchor', '—')} probes/anchor"
            )
        else:
            st.info("Connect provider credentials to initialize the session backend.")

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
                analysis_source = metadata.get("analysis_source")
                st.markdown("#### Last answer analysis")
                st.write("Source", humanize(analysis_source))
                fallback_warning = analysis_source_warning(analysis_source)
                if fallback_warning:
                    st.warning(fallback_warning)
                st.write("Polarity", humanize(latest_tag.get("polarity")))
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
                    st.write("Follow-up asked", asked_label)
                if metadata.get("llm_reflection"):
                    st.write("Grounded reflection", metadata["llm_reflection"])
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
                runtime_action(
                    lambda: require_runtime().stop(state["session_id"])
                )
                st.rerun()

            try:
                export_data = runtime_action(
                    lambda: require_runtime().export(state["session_id"])
                )
            except DemoRuntimeError as exc:
                st.warning(f"Research record export is unavailable: {exc}")
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

        if health and st.button(
            "Clear credentials and interview data",
            type="secondary",
            use_container_width=True,
        ):
            clear_credentials_and_interview_data()
            st.rerun()

        st.divider()
        st.caption(
            "Session-only research demo · Provider credentials and interview "
            "state are discarded when this Streamlit session ends."
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

    initialize_session_state()
    runtime = current_runtime()
    health: dict[str, Any] | None = None
    state: dict[str, Any] | None = None
    if runtime is not None:
        try:
            health = runtime_action(runtime.health)
            state = load_state()
            st.session_state.last_error = None
        except DemoRuntimeError as exc:
            st.session_state.last_error = str(exc)

    render_sidebar(state, health)
    render_header()
    st.markdown(
        """
        <div class="privacy-note">
          This is a bring-your-own-key demo. Credentials and interview content
          pass through the Streamlit server to NVIDIA and, for voice,
          Speechmatics. The session record is kept only in memory. Its JSON
          export contains identifying demographics and verbatim answers.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    if runtime is None:
        render_provider_setup()
        return

    if not state:
        st.markdown(
            "Experience the full consent gate, demographics, adaptive anchors, "
            "bounded probes, Swahili localization, live Llama moderation, "
            "session-scoped tags, and JSON export."
        )
        if health and health.get("stt_provider") != "speechmatics":
            st.caption(
                "Text mode is ready. Voice mode is unavailable because no "
                "Speechmatics key was connected."
            )
        if st.button(
            "Start demo interview",
            type="primary",
            use_container_width=True,
        ):
            try:
                start_interview()
                st.rerun()
            except DemoRuntimeError as exc:
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
            except DemoRuntimeError as exc:
                st.error(str(exc))
        if no_col.button("No, end here", use_container_width=True):
            try:
                submit_consent("consent_no")
                st.rerun()
            except DemoRuntimeError as exc:
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

    voice_enabled = bool(
        health and health.get("stt_provider") == "speechmatics"
    )
    response_channels = ["Text", "Voice note"] if voice_enabled else ["Text"]
    mode = st.radio(
        "Response channel",
        response_channels,
        horizontal=True,
        help=(
            "Voice audio is sent to Speechmatics under the connected "
            "credential owner's account."
            if voice_enabled
            else "Reconnect with a Speechmatics key to enable voice."
        ),
    )

    if mode == "Text":
        answer = st.chat_input(
            "Type a response, or type “stop” to end the interview",
            max_chars=8000,
        )
        if answer:
            try:
                with st.spinner("Processing the response with live moderation..."):
                    submit_answer(answer)
                st.rerun()
            except DemoRuntimeError as exc:
                st.error(str(exc))
    else:
        audio_version = st.session_state[AUDIO_WIDGET_VERSION_KEY]
        recorded_audio = st.audio_input(
            "Record your answer",
            key=f"recorded_audio_{audio_version}",
        )
        uploaded_audio = st.file_uploader(
            "Or upload an audio file",
            type=["wav", "mp3", "m4a", "mp4", "ogg", "aac", "flac", "webm"],
            key=f"uploaded_audio_{audio_version}",
        )
        selected_audio = recorded_audio or uploaded_audio
        if selected_audio and st.button(
            "Transcribe and send",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("Transcribing and processing your voice note..."):
                    submit_audio(selected_audio)
                st.rerun()
            except DemoRuntimeError as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
