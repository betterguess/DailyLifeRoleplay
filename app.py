import streamlit as st
import requests
import json
import threading
import time
import asyncio
import websockets
import pyttsx3  # placeholder for kokoro-tts
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# CONFIG
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "aphasia-trainer"
TRANSCRIBER_WS = "ws://localhost:9000/transcribe"
HOVER_TTS_PORT = 8765

st.set_page_config(page_title="Aphasia Trainer", layout="wide")

from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()  # Automatically loads environment variables from .env

# --- Build the client once ---
_client = AzureOpenAI(
    api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("AZURE_API_KEY"),
)
_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-5-chat")


# TTS
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
        self.send_response(404); self.end_headers()

def _ensure_tts_server():
    def run():
        try:
            httpd = HTTPServer(("127.0.0.1", HOVER_TTS_PORT), _TTSHandler)
            httpd.serve_forever()
        except OSError:
            pass
    threading.Thread(target=run, daemon=True).start()

_ensure_tts_server()

# Model call
def query_model(user_input: str):
    system_prompt = """"
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


    try:
        completion = _client.chat.completions.create(
            model=_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=0.2,
            max_tokens=400,
            response_format={"type": "json_object"},  # ensures JSON
        )

        raw = completion.choices[0].message.content
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            st.warning("⚠️ Model output was not valid JSON – showing raw text.")
            data = {"assistant_reply": raw, "text_suggestions": [], "emoji_suggestions": []}

        data.setdefault("assistant_reply", "")
        data.setdefault("text_suggestions", [])
        data.setdefault("emoji_suggestions", [])

        # Optional: quick debug in sidebar
        with st.sidebar:
            st.markdown("#### 🧩 Model debug")
            st.code(f"{data['assistant_reply'][:120]}…")

        return data

    except Exception as e:
        st.error(f"Model error: {e}")
        return {"assistant_reply": "Der opstod en fejl.", "text_suggestions": [], "emoji_suggestions": []}

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "text_opts" not in st.session_state:
    st.session_state.text_opts = []
if "emoji_opts" not in st.session_state:
    st.session_state.emoji_opts = []
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "Text"
if "listening" not in st.session_state:
    st.session_state.listening = False

# Sidebar
with st.sidebar:
    st.header("🎛️ Indstillinger")
    new_listen = st.toggle("🎙 Taleinput (WebSocket)", value=st.session_state.listening)
    if new_listen != st.session_state.listening:
        st.session_state.listening = new_listen
        st.rerun()
    new_mode = st.radio("Input-tilstand", ["Text", "Pictures (emoji)"], index=0 if st.session_state.input_mode == "Text" else 1)
    if new_mode != st.session_state.input_mode:
        st.session_state.input_mode = new_mode
        st.rerun()
    st.markdown("---")
    if st.button("🔄 Nulstil samtale"):
        for k in ["messages", "text_opts", "emoji_opts"]:
            st.session_state[k] = []
        st.rerun()

# Header + History
st.title("🗣️ Aphasia Conversation Trainer")
for msg in st.session_state.messages:
    avatar = "🧩" if msg["role"] == "assistant" else "👤"
    st.markdown(f"{avatar} **{msg['role'].capitalize()}:** {msg['content']}")

# WebSocket listener
async def ws_task():
    try:
        async with websockets.connect(TRANSCRIBER_WS) as ws:
            while st.session_state.listening:
                data = await ws.recv()
                try:
                    payload = json.loads(data)
                except Exception:
                    payload = {}
                if payload.get("partial"):
                    st.write(f"🟡 Partiel: {payload['partial']}")
                if payload.get("final"):
                    text = payload["final"]
                    st.session_state.messages.append({"role": "user", "content": text})
                    with st.spinner("Tænker..."):
                        reply = query_model(text)
                    st.session_state.messages.append({"role": "assistant", "content": reply.get("assistant_reply", "")})
                    st.session_state.text_opts = reply.get("text_suggestions", [])
                    st.session_state.emoji_opts = reply.get("emoji_suggestions", [])
                    threading.Thread(target=speak, args=(reply.get("assistant_reply", ""),), daemon=True).start()
                    st.session_state.listening = False
                    st.rerun()
    except Exception:
        pass

def start_ws():
    asyncio.run(ws_task())

if st.session_state.listening:
    threading.Thread(target=start_ws, daemon=True).start()

# Initial greeting
if not st.session_state.messages:
    with st.spinner("Starter samtalen..."):
        reply = query_model("<session_start>")
    st.session_state.messages.append({"role": "assistant", "content": reply.get("assistant_reply", "")})
    st.session_state.text_opts = reply.get("text_suggestions", [])
    st.session_state.emoji_opts = reply.get("emoji_suggestions", [])
    threading.Thread(target=speak, args=(reply.get("assistant_reply", ""),), daemon=True).start()
    st.rerun()

# ---------------------------
# Render suggestion tiles (with debug)
# ---------------------------
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

# Debug area
if "last_debug" not in st.session_state:
    st.session_state.last_debug = ""
st.markdown("#### 🔍 Debug info (for development)")
st.markdown(st.session_state.last_debug or "_Ingen data endnu._")

for i, opt in enumerate(opts):
    if cols[i % 5].button(opt["display"], key=f"tile_{i}"):
        # prepare and log input
        user_display = opt["display"]
        user_meaning = opt["meaning"]
        st.session_state.last_debug = f"Klik registreret: display='{user_display}', meaning='{user_meaning}'"
        st.session_state.messages.append({"role": "user", "content": user_display})
        try:
            with st.spinner(f"Tænker over: {user_meaning}"):
                resp = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": MODEL_NAME,
                        "prompt": f"Bruger siger (semantisk): {user_meaning}",
                        "stream": False,
                    },
                    timeout=180,
                )
                raw = resp.text
                st.session_state.last_debug += f"<br><br><b>Raw model output:</b><br><pre>{raw[:400]}</pre>"
                try:
                    parsed = json.loads(raw.splitlines()[-1])
                    reply_json = json.loads(parsed.get("response", "{}"))
                except Exception as e:
                    reply_json = {"assistant_reply": f"Fejl i parsing: {e}"}
                    st.session_state.last_debug += f"<br><b>Parse error:</b> {e}"
        except Exception as e:
            reply_json = {"assistant_reply": f"Fejl i API-kald: {e}"}
            st.session_state.last_debug += f"<br><b>API error:</b> {e}"

        # show reply and update state
        st.session_state.messages.append({"role": "assistant", "content": reply_json.get("assistant_reply", "")})
        st.session_state.text_opts = reply_json.get("text_suggestions", [])
        st.session_state.emoji_opts = reply_json.get("emoji_suggestions", [])
        threading.Thread(target=speak, args=(reply_json.get("assistant_reply", ""),), daemon=True).start()
        st.rerun()

    # hover div (for TTS)
    hover_id = f"hover_{i}"
    st.markdown(
        f'<div id="{hover_id}" onmouseover="startHoverTimer(\'{opt["meaning"]}\')" onmouseout="cancelHoverTimer()"></div>',
        unsafe_allow_html=True,
    )

# Manual text input
user_text = st.text_input("Skriv selv:", key="manual_input")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.spinner("Tænker..."):
        reply = query_model(user_text)
    st.session_state.messages.append({"role": "assistant", "content": reply.get("assistant_reply", "")})
    st.session_state.text_opts = reply.get("text_suggestions", [])
    st.session_state.emoji_opts = reply.get("emoji_suggestions", [])
    threading.Thread(target=speak, args=(reply.get("assistant_reply", ""),), daemon=True).start()
    st.rerun()

# Hover-to-speak JS
st.markdown(
    """
<script>
let hoverTimer = null;
function startHoverTimer(text){
  cancelHoverTimer();
  hoverTimer = setTimeout(() => {
     fetch('http://127.0.0.1:8765/_tts?text=' + encodeURIComponent(text));
  }, 10000);
}
function cancelHoverTimer(){
  if(hoverTimer){ clearTimeout(hoverTimer); hoverTimer = null; }
}
</script>
    """,
    unsafe_allow_html=True
)
