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

### 3b. Initialize database schema with Alembic

For a fresh local database:
```bash
alembic upgrade head
```

If `data/app.db` already exists and already has the current tables, mark it as current first:
```bash
alembic stamp head
```

### 3c. Optional: use PostgreSQL instead of SQLite

The app automatically uses PostgreSQL when host/user/password env vars are set.  
If they are missing, it falls back to local SQLite (`data/app.db`).

```bash
export PGSQL_HOST=localhost
export PGSQL_USER=app_user
export PGSQL_PASS=app_password
export PGSQL_DB=dailyliferoleplay      # optional (default: dailyliferoleplay)
export PGSQL_PORT=5432                  # optional (default: 5432)
export PGSQL_SSLMODE=prefer             # optional
```

Supported aliases are also accepted:
- `PSQL_HOST`
- `PSQL_USER` / `PSQL_User`
- `PSQL_PASS` / `PSQL_Pass`

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
- 🔐 **Role-based access control** with four user types

---

## 👤 Users And Roles

The app now supports four roles:

- `patient`: local username/password login and access to conversation training
- `therapist`: SSO/AD-style login (provisioned in app), can monitor own patients and create ad-hoc roleplays
- `manager`: SSO/AD-style login, can view cross-team user/activity overview
- `developer`: full access across all modules and health checks

### Hybrid authentication model

- Patients are created as **local users** (stored in the configured auth database, with salted PBKDF2 hashes).
- Employees use **SSO/Active Directory style provisioning** (email + role, restricted by optional domain policy).

### Environment options for employee login

- `STAFF_EMAIL_DOMAIN`: if set, employee email must match this domain
- `STAFF_ROLE_OVERRIDES_JSON`: optional JSON map from email to fixed role  
  Example: `{"alice@hospital.dk":"manager","bob@hospital.dk":"therapist"}`

### First startup account

If there are no users yet, the app bootstraps:

- username: `devadmin`
- password: `changeme123`

Change or replace this account immediately in non-test environments.

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

## 🗃️ Database migrations (Alembic)

The project now includes Alembic config in `alembic.ini` and migration scripts under `alembic/versions/`.

Create a new migration after model changes:
```bash
alembic revision --autogenerate -m "describe change"
```

Apply migrations:
```bash
alembic upgrade head
```

Rollback one migration:
```bash
alembic downgrade -1
```

---

## 📄 License

MIT License — for research and educational use.
