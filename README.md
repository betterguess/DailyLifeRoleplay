# 🗣️ Aphasia Conversation Trainer (Proof of Concept)

A local **multimodal conversational trainer** for people with aphasia — designed to help practice everyday Danish scenarios such as shopping, ordering food, or small talk.  
Built with **Streamlit**, **Ollama**, and **Whisper**, and intended to eventually include **kokoro-tts** for natural speech output.

---

## 🚀 Quick Start

### 1. Clone or open the project folder
```bash
cd DailyLifeRoleplay
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Start required backend services

#### 🧠 Ollama (local LLM)
Make sure Ollama is installed and running:

```bash
ollama serve
ollama pull llama3.1:8b-instruct
```

#### 🎤 Whisper realtime transcriber
Run the provided speech service (realtime partial + final transcription):

```bash
python realtime_transcriber.py
```

It should expose an endpoint such as:
```
http://localhost:9000/final
```
that returns JSON:
```json
{"text": "Jeg vil gerne købe noget kød."}
```

### 5. Launch the Streamlit interface
```bash
python -m streamlit run app.py
```

The app will open in your browser (default: [http://localhost:8501](http://localhost:8501)).

---

## 🧩 Features

- ✅ Always-listening **speech input** (can be toggled off)  
- 💬 **Dual input modes:** text or pictorial (emoji for now)  
- 🔁 **Up to 10 clickable response suggestions** per turn  
- 🔊 **Spoken replies** via kokoro-tts (placeholder: `pyttsx3`)  
- 🧱 Built for **local operation** and full privacy  

---

## 🧠 Design Overview

```
🎤 Microphone → realtime_transcriber.py → JSON (partials/final)
                       ↓
         Streamlit frontend (active listening)
                       ↓
          Ollama LLM (llama3.1:8b-instruct)
                       ↓
  ┌──────────────────────────────────────────────┐
  │ Assistant reply + 10 candidate responses     │
  │ (text + emoji, later pictograms or images)   │
  └──────────────────────────────────────────────┘
                       ↓
           Kokoro-TTS reads replies aloud
```

---

## 🧰 Project Structure

```
DailyLifeRoleplay/
├── app.py                  # Streamlit PoC
├── realtime_transcriber.py # Whisper input backend
├── requirements.txt
├── README.md
└── .venv/                  # Virtual environment (local)
```

---

## ⚙️ Requirements

| Component | Description | Notes |
|------------|--------------|-------|
| **Python ≥ 3.9** | Core runtime | Tested on macOS |
| **Ollama** | Local LLM serving (`llama3.1:8b-instruct`) | [ollama.ai/download](https://ollama.ai/download) |
| **Whisper** | Realtime STT (`realtime_transcriber.py`) | ggerganov/whisper.cpp or faster-whisper |
| **Kokoro-TTS** | Natural speech output | Optional — placeholder uses `pyttsx3` |
| **Streamlit** | Frontend UI | Installed via `requirements.txt` |

---

## 🧭 Next Steps

- 🔄 Replace emoji with **real pictograms** or **generated images**
- 🖱️ Add **hover-to-speak** feature for response tiles
- 🔈 Integrate **kokoro-tts** playback via API
- 🧩 Add **custom Modelfile** for aphasia-friendly prompting
- 🧠 Optional: persist user progress or scenario tracking

---

## 📄 License

MIT License — for research and educational use.
