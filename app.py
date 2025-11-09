import os
import json
import threading
import asyncio
import requests
import streamlit as st
from openai import AzureOpenAI
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import websockets

# ============================================================
#  ENV + GLOBAL CONFIG
# ============================================================

import os

def get_secret(name: str, default=None):
    """
    Retrieve a secret value, preferring environment variables (Azure, Docker)
    but falling back to Streamlit secrets for local dev and Streamlit Cloud.
    """
    if name in os.environ:
        return os.environ[name]
    if name in st.secrets:
        return st.secrets[name]
    if default is not None:
        return default
    raise KeyError(f"Missing required secret: {name}")


AZURE_API_KEY = get_secret("AZURE_API_KEY")
AZURE_ENDPOINT = get_secret("AZURE_ENDPOINT")
AZURE_DEPLOYMENT = get_secret("AZURE_DEPLOYMENT")
AZURE_API_VERSION = get_secret("AZURE_API_VERSION")

TRANSCRIBER_WS = "ws://localhost:9000/transcribe"
HOVER_TTS_PORT = 8765

st.set_page_config(page_title="Aphasia Conversation Trainer", layout="wide")

import os
import streamlit as st

import os, requests, streamlit as st

def azure_diagnostics():
    st.header("🔍 Azure Connectivity Diagnostics")

    # Collect key variables
    endpoint_raw = os.environ.get("AZURE_ENDPOINT") or "(not set)"
    endpoint = endpoint_raw.strip().rstrip("/") if endpoint_raw else None
    deployment = os.environ.get("AZURE_DEPLOYMENT", "(not set)")
    api_version = os.environ.get("AZURE_API_VERSION", "(not set)")
    api_key = os.environ.get("AZURE_API_KEY", "(not set)")
    region_from_url = endpoint.split(".")[1] if endpoint and "." in endpoint else "(unknown)"

    st.subheader("📦 Environment Variables")
    st.write(f"**AZURE_ENDPOINT:** `{endpoint_raw}`")
    st.write(f"**Trimmed Endpoint:** `{endpoint}`")
    st.write(f"**AZURE_DEPLOYMENT:** `{deployment}`")
    st.write(f"**AZURE_API_VERSION:** `{api_version}`")
    st.write(f"**Region from Endpoint:** `{region_from_url}`")
    st.write(f"**Key length:** {len(api_key) if api_key!='(not set)' else 0}")

    # Proxy check
    for var in ["HTTP_PROXY", "HTTPS_PROXY"]:
        if os.environ.get(var):
            st.warning(f"⚠️ Proxy variable detected: {var}={os.environ[var]}")
            os.environ.pop(var, None)
            st.info(f"Removed {var} to prevent Azure SSL issues.")
        else:
            st.success(f"No proxy variable set: {var}")

    if not endpoint or not api_key or not deployment:
        st.error("❌ Missing one or more required environment variables.")
        return

    # Form URL
    test_url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    st.subheader("🌐 Direct Endpoint Test")
    st.code(test_url, language="bash")

    try:
        r = requests.post(
            test_url,
            headers={"api-key": api_key, "Content-Type": "application/json"},
            json={"messages": [{"role": "user", "content": "Hello"}]},
            timeout=10,
        )
        st.write("**HTTP Status:**", r.status_code)
        st.text(r.text[:400])

        if r.status_code == 200:
            st.success("✅ Connection OK — Azure OpenAI reachable and deployment valid.")
        elif r.status_code == 401:
            st.error("❌ Unauthorized (invalid API key). Double-check your AZURE_API_KEY.")
        elif r.status_code == 404:
            st.error("❌ Deployment or endpoint not found — check AZURE_DEPLOYMENT name and region.")
        else:
            st.warning(f"⚠️ Unexpected response code: {r.status_code}")
    except Exception as e:
        st.exception(e)

# --- Run diagnostics on demand ---
if st.sidebar.button("Run Azure Diagnostics"):
    azure_diagnostics()
    

# ============================================================
#  UNIVERSAL CHOICES
# ============================================================

UNIVERSAL_CHOICES = [
    {"display": "🆘", "meaning": "Hjælp", "meta": "HELP"},
    {"display": "😕", "meaning": "Forstår ikke", "meta": "CONFUSED"},
    {"display": "👍", "meaning": "Ja", "meta": "YES"},
    {"display": "👎", "meaning": "Nej", "meta": "NO"}
]

# ============================================================
#  SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
Du er en venlig dansk sprogtræner, der hjælper personer med afasi med at øve hverdagssamtaler. Specifikt øver vi os i dag 
i at købe ind hos slagteren. Du skal spille rollen som en venlig slagter der hjælper kunden.

Hvis samtalen ikke fungerer for brugeren kan du bryde ud af rollen og i stedet være en sprogterapeut der prøvet at hjælpe brugeren.

Du skal starte så simplet som muligt, men må gerne udforde mere både med spørgsmål og svarmuligheder hvis du vurderer at brugeren
klarer sig godt nok til at blive udfordret mere.

Du må gerne kommunikere med emoji og andre billeder, hvis det virker som om det er nødvendigt. Samtalen slutter når kunden har opnået 
deres mål som er at købe ind til et måltid ELLER har opgivet opgaven

Tal i korte, tydelige sætninger. Gentag nøgleord

Svar altid på dansk.

Når du modtager strengen \"<session_start>\", skal du begynde samtalen med en venlig dansk hilsen og foreslå 3–5 helt enkle svarmuligheder.

Hvis brugeren siger eller klikker på noget af det følgende, skal du reagere som en sprogtræner
i stedet for at fortsætte scenariet:

- "Hjælp" eller meta:HELP → Forklar kort, hvad brugeren kan sige, eller giv et forslag.
- "Forstår ikke" eller meta:CONFUSED → Forklar langsomt, gentag sidste sætning enklere. Hvis du ser den flere gange eller vurderer at brugeren er i affekt så bryd ud af rollespillet og vurder om der skal fortsættes.
- "Ja" eller meta:YES → Bekræft venligt, evt. med et simpelt opfølgende spørgsmål.
- "Nej" eller meta:NO → Anerkend svaret og tilbyd et alternativ.

Returnér ALTID gyldig JSON med denne struktur:
{
"assistant_reply": "<din korte sætning>",
"text_suggestions": ["mulighed 1", "mulighed 2", "..."],
"emoji_suggestions": ["emoji1", "emoji2", "..."]
}

Når samtalen er slut, uanset hvordan, så giv en vurdering af, hvordan det gik, og hviklet niveau af udfordringer brugeren er klar til som næste øvelse.

Hvis du har historik på brugeren så tag den i betragtning, og kom med et nyt bud på aktuel status.

Krav:
- Kun gyldig JSON som svar. Ingen forklaringer eller tekst uden for JSON.
- `assistant_reply` er din tale til brugeren, max 1–2 korte sætninger.
- `text_suggestions` 3–8 korte danske muligheder.
- `emoji_suggestions` samme længde og rækkefølge som text_suggestions (1:1 match).
- Hvis en tekstmulighed ikke har en naturlig emoji, brug "🗨️".
- Hold en støttende, tydelig, rolig tone.
    """

import glob

SCENARIO_DIR = "scenarios"

def load_scenarios():
    scenarios = []
    for f in glob.glob(f"{SCENARIO_DIR}/*.json"):
        with open(f, "r", encoding="utf-8") as infile:
            try:
                data = json.load(infile)
                scenarios.append(data)
            except Exception as e:
                st.warning(f"Kunne ikke indlæse {f}: {e}")
    return sorted(scenarios, key=lambda s: s["title"])

SCENARIOS = load_scenarios()

# ============================================================
#  TTS MICROSERVER
# ============================================================

import os
import streamlit as st

import re

def strip_emojis(text: str) -> str:
    # Match and remove all emoji and pictographs
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub("", text)

# Try Azure TTS first
try:
    import azure.cognitiveservices.speech as speechsdk

    def speak(text: str):
        text = strip_emojis(text)
        speech_config = speechsdk.SpeechConfig(
            subscription=get_secret("AZURE_SPEECH_KEY"),
            region=get_secret("AZURE_SPEECH_REGION"),
        )
        # speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"  # choose your voice
        speech_config.speech_synthesis_voice_name = "da-DK-JeppeNeural"  # choose your voice
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=audio_config
        )
        result = synthesizer.speak_text_async(text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            st.audio(result.audio_data, format="audio/wav")
        else:
            st.error(f"TTS failed: {result.reason}")

except Exception as e:
    # Fallback to pyttsx3 locally
    try:
        import pyttsx3
        engine = pyttsx3.init()

        def speak(text: str):
            engine.say(text)
            engine.runAndWait()

    except ImportError:
        def speak(text: str):
            st.warning("TTS not available on this environment.")


class _TTSHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/_tts":
                qs = parse_qs(parsed.query)
                text = qs.get("text", [""])[0]
                if text:
                    threading.Thread(target=speak, args=(text,), daemon=True).start()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"ok")
                return
        except Exception:
            pass
        self.send_response(404)
        self.end_headers()


def _ensure_tts_server():
    def run():
        try:
            httpd = HTTPServer(("127.0.0.1", HOVER_TTS_PORT), _TTSHandler)
            httpd.serve_forever()
        except OSError:
            pass  # already running

    threading.Thread(target=run, daemon=True).start()


_ensure_tts_server()

# ============================================================
#  AZURE GPT-5 CLIENT
# ============================================================

client = AzureOpenAI(
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
)

# ============================================================
#  MODEL QUERY FUNCTION
# ============================================================

def query_model(user_input: str):
    """Send full chat history to Azure GPT-5 and return structured JSON."""
    try:
        scenario_extra = ""
        if "scenario_index" in st.session_state and SCENARIOS:
            scenario_extra = SCENARIOS[st.session_state.scenario_index].get("system_prompt_addition", "")
        system_prompt = SYSTEM_PROMPT + "\n\n" + scenario_extra

        # Build full history
        history = [{"role": "system", "content": system_prompt}]
        for m in st.session_state.messages:
            history.append({"role": m["role"], "content": m["content"]})
        history.append({"role": "user", "content": user_input})

        completion = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=history,
            temperature=0.2,
            max_tokens=400,
            response_format={"type": "json_object"},
        )

        raw = completion.choices[0].message.content
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            st.warning("⚠️ Model output was not valid JSON – showing raw text.")
            data = {
                "assistant_reply": raw,
                "text_suggestions": [],
                "emoji_suggestions": [],
            }

        data.setdefault("assistant_reply", "")
        data.setdefault("text_suggestions", [])
        data.setdefault("emoji_suggestions", [])

        # Optional quick debug
        with st.sidebar:
            st.markdown("#### 🧩 Model debug")
            st.code(
                f"{data['assistant_reply'][:120]}…\n"
                f"text_opts: {len(data['text_suggestions'])}, emoji_opts: {len(data['emoji_suggestions'])}"
            )

        return data

    except Exception as e:
        st.error(f"Model error: {e}")
        return {
            "assistant_reply": "Der opstod en fejl.",
            "text_suggestions": [],
            "emoji_suggestions": [],
        }

# ============================================================
#  SESSION STATE
# ============================================================

for key, default in {
    "messages": [],
    "text_opts": [],
    "emoji_opts": [],
    "input_mode": "Text",
    "listening": False,
    "sent_this_turn": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
#  SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### 🏪 Scenario")
    if SCENARIOS:
        titles = [s["title"] for s in SCENARIOS]
        selected_title = st.selectbox(
            "Vælg en situation:",
            titles,
            index=0 if "scenario_index" not in st.session_state else st.session_state.scenario_index,
        )
        st.session_state.scenario_index = titles.index(selected_title)
        current_scenario = SCENARIOS[st.session_state.scenario_index]
        st.markdown(f"🗒️ {current_scenario['description']}")
    else:
        st.warning("Ingen scenarier fundet i ./scenarios/")
        current_scenario = None

    st.header("🎛️ Indstillinger")
    new_listen = st.toggle("🎙 Taleinput (WebSocket)", value=st.session_state.listening)
    if new_listen != st.session_state.listening:
        st.session_state.listening = new_listen
        st.rerun()

    new_mode = st.radio(
        "Input-tilstand", ["Text", "Pictures (emoji)"],
        index=0 if st.session_state.input_mode == "Text" else 1,
    )
    if new_mode != st.session_state.input_mode:
        st.session_state.input_mode = new_mode
        st.rerun()

    st.markdown("---")
    if st.button("🔄 Nulstil samtale"):
        for k in ["messages", "text_opts", "emoji_opts"]:
            st.session_state[k] = []
        st.rerun()

# ============================================================
#  HEADER + HISTORY
# ============================================================

st.title("🗣️ Aphasia Conversation Trainer")

for msg in st.session_state.messages:
    avatar = "🧩" if msg["role"] == "assistant" else "👤"
    st.markdown(f"{avatar} **{msg['role'].capitalize()}:** {msg['content']}")

# ============================================================
#  SPEECH WEBSOCKET LISTENER
# ============================================================

async def ws_task():
    try:
        async with websockets.connect(TRANSCRIBER_WS) as ws:
            while st.session_state.listening:
                data = await ws.recv()
                try:
                    payload = json.loads(data)
                except Exception:
                    payload = {}
                if "partial" in payload and payload["partial"]:
                    st.write(f"🟡 Partiel: {payload['partial']}")
                if "final" in payload and payload["final"]:
                    text = payload["final"]
                    st.session_state.messages.append({"role": "user", "content": text})
                    with st.spinner("Tænker..."):
                        reply = query_model(text)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply.get("assistant_reply", "")}
                    )
                    st.session_state.text_opts = reply.get("text_suggestions", [])
                    st.session_state.emoji_opts = reply.get("emoji_suggestions", [])
                    threading.Thread(
                        target=speak,
                        args=(reply.get("assistant_reply", ""),),
                        daemon=True,
                    ).start()
                    st.session_state.listening = False
                    st.rerun()
    except Exception:
        pass


def start_ws():
    asyncio.run(ws_task())


if st.session_state.listening:
    threading.Thread(target=start_ws, daemon=True).start()

# ============================================================
#  INITIAL GREETING
# ============================================================

if not st.session_state.messages:
    with st.spinner("Starter samtalen..."):
        if current_scenario and current_scenario.get("first_message"):
            first = current_scenario["first_message"]
            reply = query_model(first)
        else:
            reply = query_model("<session_start>")
    st.session_state.messages.append(
        {"role": "assistant", "content": reply.get("assistant_reply", "")}
    )
    st.session_state.text_opts = reply.get("text_suggestions", [])
    st.session_state.emoji_opts = reply.get("emoji_suggestions", [])
    threading.Thread(
        target=speak,
        args=(reply.get("assistant_reply", ""),),
        daemon=True,
    ).start()
    st.rerun()

# ============================================================
#  SUGGESTION TILES
# ============================================================

st.markdown("### Vælg et svar:")


def build_options():
    opts = []
    if st.session_state.input_mode == "Text":
        opts = [{"display": t, "meaning": t, "meta": None}
                for t in (st.session_state.text_opts or [])]
    else:
        emj = st.session_state.emoji_opts or []
        txt = st.session_state.text_opts or []
        for i, e in enumerate(emj):
            meaning = txt[i] if i < len(txt) else e
            opts.append({"display": e, "meaning": meaning, "meta": None})
        if not opts and txt:
            opts = [{"display": "🗨️", "meaning": t, "meta": None}
                    for t in txt[:5]]
        if not opts:
            opts = [{"display": "🤝", "meaning": "Hej", "meta": None}]
    return opts


# ==== META BUTTON ROW ====
st.markdown("### Hurtige svar")
with st.container():
    st.markdown('<div class="meta-scope">', unsafe_allow_html=True)
    meta_cols = st.columns(len(UNIVERSAL_CHOICES))
    for i, opt in enumerate(UNIVERSAL_CHOICES):
        with meta_cols[i]:
            if st.button(opt["display"], key=f"meta_{i}", use_container_width=True):
                text_to_send = f"<meta:{opt['meta']}> {opt['meaning']}"
                st.session_state.messages.append({"role": "user", "content": text_to_send})
                with st.spinner("Tænker..."):
                    reply = query_model(text_to_send)
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply.get("assistant_reply", "")})
                st.session_state.text_opts = reply.get("text_suggestions", [])
                st.session_state.emoji_opts = reply.get("emoji_suggestions", [])
                threading.Thread(target=speak,
                                 args=(reply.get("assistant_reply",""),),
                                 daemon=True).start()
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ==== NORMAL OPTIONS ROW ====
st.markdown("### Mulige svar")
with st.container():
    st.markdown('<div class="opts-scope">', unsafe_allow_html=True)
    opts = build_options()
    num_cols = min(5, len(opts)) or 1
    cols = st.columns(num_cols)
    for i, opt in enumerate(opts):
        with cols[i % num_cols]:
            if st.button(opt["display"], key=f"opt_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": opt["meaning"]})
                with st.spinner("Tænker..."):
                    reply = query_model(opt["meaning"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply.get("assistant_reply","")})
                st.session_state.text_opts = reply.get("text_suggestions", [])
                st.session_state.emoji_opts = reply.get("emoji_suggestions", [])
                threading.Thread(target=speak,
                                 args=(reply.get("assistant_reply",""),),
                                 daemon=True).start()
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    hover_id = f"hover_{i}"
    st.markdown(
        f'<div id="{hover_id}" onmouseover="startHoverTimer(\'{opt["meaning"]}\')" onmouseout="cancelHoverTimer()"></div>',
        unsafe_allow_html=True,
    )

# ============================================================
#  MANUAL TEXT INPUT (FINAL, CLEAN + NO WARNINGS)
# ============================================================

def handle_user_input():
    """Triggered automatically when user presses Enter or leaves the text field."""
    user_text = st.session_state.manual_input.strip()
    if not user_text:
        return
    st.session_state.sent_this_turn = True
    st.session_state.messages.append({"role": "user", "content": user_text})

    with st.spinner("Tænker..."):
        reply = query_model(user_text)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply.get("assistant_reply", "")}
    )
    st.session_state.text_opts = reply.get("text_suggestions", [])
    st.session_state.emoji_opts = reply.get("emoji_suggestions", [])
    threading.Thread(
        target=speak,
        args=(reply.get("assistant_reply", ""),),
        daemon=True,
    ).start()

    # clear field and reset guard
    st.session_state.manual_input = ""
    st.session_state.sent_this_turn = False
    # no need for st.rerun() — Streamlit will re-render automatically


st.text_input(
    "Skriv selv:",
    key="manual_input",
    on_change=handle_user_input,
)

# ============================================================
#  HOVER-TO-SPEAK JAVASCRIPT
# ============================================================

st.markdown(
    f"""
<script>
let hoverTimer = null;
function startHoverTimer(text){{
  cancelHoverTimer();
  hoverTimer = setTimeout(() => {{
     fetch('http://127.0.0.1:{HOVER_TTS_PORT}/_tts?text=' + encodeURIComponent(text));
  }}, 10000);
}}
function cancelHoverTimer(){{
  if(hoverTimer){{ clearTimeout(hoverTimer); hoverTimer = null; }}
}}
</script>
""",
    unsafe_allow_html=True,
)

# CSS for buttons

st.markdown("""
<style>
/* Center column contents */
div[data-testid="column"] {
    display: flex;
    justify-content: center;
}

/* ========== META BUTTONS ========== */
.meta-scope div.stButton > button,
.meta-scope div.stButton > button * {
    width: 200px !important;
    height: 220px !important;
    margin: 8px !important;
    border-radius: 20px !important;
    background-color: #e5f1ff !important;
    border: 3px solid #5b9bff !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.15) !important;

    /* Safari fix */
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    -webkit-text-size-adjust: none !important;

    font-size: 110px !important;
    line-height: 1 !important;
    text-align: center !important;
    padding: 0 !important;
    font-family: "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji",
                 system-ui, sans-serif !important;
}

/* ========== NORMAL BUTTONS ========== */
.opts-scope div.stButton > button,
.opts-scope div.stButton > button * {
    width: 200px !important;
    height: 140px !important;
    margin: 8px !important;
    border-radius: 16px !important;
    background-color: #f9f9f9 !important;
    border: 2px solid #ccc !important;
    box-shadow: 0 2px 3px rgba(0,0,0,0.1) !important;

    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    -webkit-text-size-adjust: none !important;

    font-size: 36px !important;
    line-height: 1.1 !important;
    padding: 4px 8px !important;
}

/* Hover feedback */
.meta-scope div.stButton > button:hover {
    background-color: #d6e8ff !important;
    transform: scale(1.04);
}
.opts-scope div.stButton > button:hover {
    background-color: #f0f0f0 !important;
}
</style>
""", unsafe_allow_html=True)