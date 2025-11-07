import os
import json
import threading
import asyncio
import requests
import streamlit as st
from openai import AzureOpenAI
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
import pyttsx3
import websockets

# ============================================================
#  ENV + GLOBAL CONFIG
# ============================================================

load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-5-chat")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
TRANSCRIBER_WS = "ws://localhost:9000/transcribe"
HOVER_TTS_PORT = 8765

st.set_page_config(page_title="Aphasia Conversation Trainer", layout="wide")

# ============================================================
#  SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
Du er en venlig dansk sprogtræner, der hjælper personer med afasi med at øve hverdagssamtaler. Specifikt øver vi os i dag 
i at købe ind hos slagteren. Du skal spille rollen som en venlig slagter der hjælper kunden.

Hvis samtalen ikke fungerer for brugeren kan du bryde ud af rollen og i stedet være en sprogterapeut der prøvet at hjælpe brugeren.

Du må gerne kommunikere med emoji og andre billeder, hvis det virker som om det er nødvendigt. Samtalen slutter når kunden har opnået 
deres mål som er at købe ind til et måltid ELLER har opgivet opgaven

Tal i korte, tydelige sætninger. Gentag nøgleord

Svar altid på dansk.

Når du modtager strengen \"<session_start>\", skal du begynde samtalen med en venlig dansk hilsen og foreslå 3–5 helt enkle svarmuligheder.


Returnér ALTID gyldig JSON med denne struktur:
{
"assistant_reply": "<din korte sætning>",
"text_suggestions": ["mulighed 1", "mulighed 2", "..."],
"emoji_suggestions": ["emoji1", "emoji2", "..."]
}

Krav:
- Kun gyldig JSON som svar. Ingen forklaringer eller tekst uden for JSON.
- `assistant_reply` er din tale til brugeren, max 1–2 korte sætninger.
- `text_suggestions` 3–8 korte danske muligheder.
- `emoji_suggestions` samme længde og rækkefølge som text_suggestions (1:1 match).
- Hvis en tekstmulighed ikke har en naturlig emoji, brug "🗨️".
- Hold en støttende, tydelig, rolig tone.
    """


# ============================================================
#  TTS MICROSERVER
# ============================================================

def speak(text: str):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("TTS error:", e)


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
        # Build full message context
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
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
        opts = [{"display": t, "meaning": t} for t in (st.session_state.text_opts or [])]
    else:
        emj = st.session_state.emoji_opts or []
        txt = st.session_state.text_opts or []
        for i, e in enumerate(emj):
            meaning = txt[i] if i < len(txt) else e
            opts.append({"display": e, "meaning": meaning})
        if not opts and txt:
            opts = [{"display": "🗨️", "meaning": t} for t in txt[:5]]
        if not opts:
            opts = [{"display": "🤝", "meaning": "Hej"}]
    return opts[:10]


opts = build_options()
cols = st.columns(min(5, len(opts)) or 1)

for i, opt in enumerate(opts):
    if cols[i % 5].button(opt["display"], key=f"tile_{i}"):
        st.session_state.messages.append({"role": "user", "content": opt["display"]})
        with st.spinner("Tænker..."):
            reply = query_model(opt["meaning"])
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