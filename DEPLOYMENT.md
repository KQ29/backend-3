# Streamlit-only deployment

The hosted demo is a single Streamlit service:

```text
Browser -> Streamlit Community Cloud
               -> NVIDIA Llama
               -> Speechmatics (optional voice)
```

FastAPI, Render, Supabase, environment variables, and Streamlit Secrets are not
required. The Streamlit Python process is the backend: each operator supplies
their own provider credentials through masked fields before starting an
interview.

## 0. Rotate previously exposed credentials

Do not reuse a credential that has appeared in chat, source code, an issue, or
a public log.

1. Revoke the previously exposed Supabase key. No replacement is required.
2. Rotate or delete the exposed NVIDIA personal key.
3. Revoke the exposed Speechmatics key.
4. Create fresh NVIDIA and, if voice is needed, Speechmatics credentials.

Never commit or paste the replacements into Streamlit Secrets.

## 1. Deploy one Streamlit app

1. Open Streamlit Community Cloud and choose **Create app**.
2. Select repository `KQ29/backend-3`.
3. Select branch `main`.
4. Set the entrypoint to `streamlit_app.py`.
5. In **Advanced settings**, select Python `3.13`.
6. Leave **Secrets** empty.
7. Click **Deploy**.

The root `requirements.txt` supplies all Python dependencies. No Render URL,
backend token, database, or provider credential is needed during deployment.

## 2. Connect providers in the app

On each new Streamlit browser session:

1. Enter a fresh NVIDIA API key in the masked credential form.
2. Optionally enter a Speechmatics key to enable voice answers.
3. Accept the credential-processing notice.
4. Select **Verify and continue**.

NVIDIA verification performs one small synthetic classification and may consume
provider quota. Speechmatics verification is read-only and creates no
transcription job.

The credentials travel over HTTPS to the Streamlit server and from there to
their respective providers. They are kept only in that Streamlit session's
server memory; they are not written to files, logs, interview records, or JSON
exports. Selecting **Clear credentials and interview data**, disconnecting the
session, or restarting the app discards the active runtime.

Do not ask research respondents to supply organization credentials. The person
operating the demo should enter a revocable key that has appropriate spending
limits.

## 3. Data and privacy boundaries

- Substantive answers and a bounded rolling topic summary are sent to NVIDIA.
  Demographic collection turns are not sent to the LLM.
- Voice audio is sent to Speechmatics under the credential owner's account and
  provider terms.
- Name, email, demographics, verbatim answers, and tags remain in the
  session-scoped interview record and its JSON export. The raw export is
  identifiable even if later reporting is anonymized.
- No Supabase or other persistent database is used.
- A refresh, disconnect, Streamlit restart, or explicit reset can remove the
  interview. Download the JSON record before leaving the session.
- Provider calls may incur charges to the credential owner's account.

For a controlled research demo, use Streamlit's sharing settings to restrict
the app to intended viewers.

## 4. Verify the live path

1. Confirm the sidebar says **Session backend ready** and **Storage: Memory**.
2. Start an interview and accept consent.
3. Complete one substantive text answer and confirm its analysis source is
   `LLM`, not `Local Provider Fallback`.
4. If Speechmatics was connected, submit a non-sensitive voice note.
5. Confirm probe usage never exceeds `2/2`.
6. Complete or stop the interview and download its JSON export.
7. Select **Clear credentials and interview data** when finished.

If an answer shows `Local Provider Fallback`, the live NVIDIA request failed and
the deterministic safety rules handled that answer instead. Reconnect with a
valid key before treating the interview as a live-model demonstration.
