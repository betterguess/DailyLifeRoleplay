import streamlit as st
import requests
import json
import threading
import time
import asyncio
import websockets
import pyttsx3  # placeholder for kokoro-tts
import os

# ---------------------------
# CONFIG
# ---------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "aphasia-trainer"
TRANSCRIBER_WS = "ws://localhost:9000/transcribe"   # websocket stub for your realtime_transcriber.py

st.set_page_config(page_title="Aphasia Trainer", layout="wide")

# ---------------------------
# SPEECH OUTPUT
# ---------------------------
def speak(text: str):
    """Simple TTS wrapper — swap to kokoro-tts later"""
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("TTS error:", e)

# ---------------------------
# STATE
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "listening" not in st.session_state:
    st.session_state.listening = False
if "text_opts" not in st.session_state:
    st.session_state.text_opts = []
if "emoji_opts" not in st.session_state:
    st.session_state.emoji_opts = []

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.header("🎛️ Settings")
    st.session_state.listening = st.toggle("🎙 Speech input", value=st.session_state.listening)
    input_mode = st.radio("Input mode", ["Text", "Pictures (emoji)"])
    st.markdown("---")
    st.markdown("Hover a tile for 10 s to hear it read aloud.")

# ---------------------------
# CHAT HISTORY
# ---------------------------
st.title("🗣️ Aphasia Conversation Trainer")

for msg in st.session_state.messages:
    avatar = "🧩" if msg["role"] == "assistant" else "👤"
    st.markdown(f"{avatar} **{msg['role'].capitalize()}:** {msg['content']}")

# ---------------------------
# BACKGROUND SPEECH LISTENER (WebSocket)
# ---------------------------
async def listen_ws():
    try:
        async with websockets.connect(TRANSCRIBER_WS) as ws:
            while st.session_state.listening:
                msg = await ws.recv()
                data = json.loads(msg)
                if "final" in data:
                    text = data["final"]
                    st.session_state.messages.append({"role": "user", "content": text})
                    st.session_state.listening = False
                    st.rerun()
    except Exception as e:
        print("WebSocket listener error:", e)

def start_ws_listener():
    asyncio.run(listen_ws())

if st.session_state.listening:
    threading.Thread(target=start_ws_listener, daemon=True).start()

# ---------------------------
# MODEL CALL
# ---------------------------
def query_model(user_input):
    system_prompt = (
        """You will be speaking Danish.
        You are helping a person with aphasia practice everyday scenarios.
        Specifically we are practicing a visit to the butcher.
        You will be playing the part of a helpful butcher who is helping the customer. If conversaiton breaks down completely, feel free to break character and help the user move the conversation forward with encouragment and help.
        The conversation ends when the customer has achieved their goal of shopping for dinner OR have given up on the task."""
    )

    payload = {"model": MODEL_NAME, "prompt": f"{system_prompt}\nBruger: {user_input}\nSvar:", "stream": False}
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    raw = resp.text.splitlines()[-1]
    try:
        parsed = json.loads(raw)
        j = json.loads(parsed.get("response", "{}"))
    except Exception:
        j = {"assistant_reply": "Beklager, der gik noget galt.", "text_suggestions": [], "emoji_suggestions": []}
    return j

# ---------------------------
# HANDLE USER INPUT
# ---------------------------
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
    user_msg = st.session_state.messages[-1]["content"]
    with st.spinner("Tænker..."):
        reply = query_model(user_msg)
    st.session_state.messages.append({"role": "assistant", "content": reply.get("assistant_reply", "")})
    st.session_state.text_opts = reply.get("text_suggestions", [])
    st.session_state.emoji_opts = reply.get("emoji_suggestions", [])
    threading.Thread(target=speak, args=(reply.get("assistant_reply", ""),), daemon=True).start()
    st.rerun()

# ---------------------------
# RENDER SUGGESTIONS + HOVER-TO-SPEAK
# ---------------------------
st.markdown("### Vælg et svar:")

opts = (
    st.session_state.text_opts
    if input_mode == "Text"
    else st.session_state.emoji_opts or ["🥩", "🐖", "🍗", "🐟"]
)

cols = st.columns(min(5, len(opts)) or 1)
for i, s in enumerate(opts):
    btn = cols[i % 5].button(s, key=f"sugg_{i}")
    # Inject per-tile hover listener
    hover_id = f"hover_{i}"
    st.markdown(
        f"""
        <div id="{hover_id}" onmouseover="startHoverTimer('{s}', '{hover_id}')"
             onmouseout="cancelHoverTimer('{hover_id}')"></div>
        """,
        unsafe_allow_html=True,
    )
    if btn:
        st.session_state.messages.append({"role": "user", "content": s})
        st.rerun()

# ---------------------------
# TEXT INPUT FALLBACK
# ---------------------------
user_text = st.text_input("Skriv selv:", key="manual_input")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.rerun()

# ---------------------------
# HOVER-TO-SPEAK JavaScript (10 s)
# ---------------------------
st.markdown(
    """
<script>
let hoverTimers = {};
function startHoverTimer(text, id){
  cancelHoverTimer(id);
  hoverTimers[id] = setTimeout(() => {
      fetch('/_tts?text=' + encodeURIComponent(text));
  }, 10000);
}
function cancelHoverTimer(id){
  if(hoverTimers[id]){
      clearTimeout(hoverTimers[id]);
      delete hoverTimers[id];
  }
}
</script>
""",
    unsafe_allow_html=True,
)