import asyncio
import json
import socket
import threading
from urllib.parse import urlparse

import streamlit as st
import streamlit.components.v1 as components
import websockets

from src.auth import (
    ROLE_DEVELOPER,
    ROLE_MANAGER,
    ROLE_PATIENT,
    ROLE_THERAPIST,
    User,
    authenticate_local_user,
    create_local_user,
    get_activity_counts,
    get_patients_for_therapist,
    get_therapists,
    has_permission,
    init_auth_store,
    list_users,
    log_event,
    provision_sso_user,
    sso_domain_allowed,
)
from src.config import load_azure_settings
from src.model import create_client, query_model as query_model_core
from src.scenarios import load_scenarios
from src.tts import build_speak, ensure_hover_tts_server, start_tts_server


TRANSCRIBER_WS = "ws://localhost:9000/transcribe"
HOVER_TTS_PORT = 8765

st.set_page_config(page_title="Aphasia Conversation Trainer", layout="wide")
init_auth_store()

# Runtime services
speak = build_speak()
tts_port = start_tts_server(speak)
if tts_port is None:
    tts_port = HOVER_TTS_PORT
ensure_hover_tts_server(speak, port=HOVER_TTS_PORT)


UNIVERSAL_CHOICES = [
    {"display": "🆘", "meaning": "Hjælp", "meta": "HELP"},
    {"display": "😕", "meaning": "Forstår ikke", "meta": "CONFUSED"},
    {"display": "👍", "meaning": "Ja", "meta": "YES"},
    {"display": "👎", "meaning": "Nej", "meta": "NO"},
]


SYSTEM_PROMPT = """
Du er en venlig dansk sprogtræner, der hjælper personer med afasi med at øve hverdagssamtaler.

Hvis samtalen ikke fungerer for brugeren kan du bryde ud af rollen og i stedet være en sprogterapeut der prøver at hjælpe brugeren.

Du skal starte så simplet som muligt, men må gerne udforde mere både med spørgsmål og svarmuligheder hvis du vurderer at brugeren
klarer sig godt nok til at blive udfordret mere.

Du må gerne kommunikere med emoji og andre billeder, hvis det virker som om det er nødvendigt. Samtalen slutter når kunden har opnået
deres mål som er at købe ind til et måltid ELLER har opgivet opgaven

Tal i korte, tydelige sætninger. Gentag nøgleord

Svar altid på dansk.

Når du modtager strengen "<session_start>", skal du begynde samtalen med en venlig dansk hilsen og foreslå 3-5 helt enkle svarmuligheder.

Hvis brugeren siger eller klikker på noget af det følgende, skal du reagere som en sprogtræner
i stedet for at fortsætte scenariet:

- "Hjælp" eller meta:HELP -> Forklar kort, hvad brugeren kan sige, eller giv et forslag.
- "Forstår ikke" eller meta:CONFUSED -> Forklar langsomt, gentag sidste sætning enklere. Hvis du ser den flere gange eller vurderer at brugeren er i affekt så bryd ud af rollespillet og vurder om der skal fortsættes.
- "Ja" eller meta:YES -> Bekræft venligt, evt. med et simpelt opfølgende spørgsmål.
- "Nej" eller meta:NO -> Anerkend svaret og tilbyd et alternativ.

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
- `assistant_reply` er din tale til brugeren, max 1-2 korte sætninger.
- `text_suggestions` 3-8 korte danske muligheder.
- `emoji_suggestions` samme længde og rækkefølge som text_suggestions (1:1 match).
- Hvis en tekstmulighed ikke har en naturlig emoji, brug "🗨️".
- Hold en støttende, tydelig, rolig tone.
"""


SCENARIOS = load_scenarios()
settings = load_azure_settings()
client = create_client(settings)


def query_model(user_input: str):
    """Compose scenario-aware system prompt and call Azure model."""
    try:
        scenario_extra = ""
        if st.session_state.get("use_custom_scenario") and st.session_state.get("custom_scenario"):
            scenario_extra = st.session_state.custom_scenario.get("system_prompt_addition", "")
        elif "scenario_index" in st.session_state and SCENARIOS:
            scenario_extra = SCENARIOS[st.session_state.scenario_index].get("system_prompt_addition", "")

        system_prompt = SYSTEM_PROMPT + "\n\n" + scenario_extra

        data = query_model_core(
            client=client,
            deployment=settings.deployment,
            system_prompt=system_prompt,
            user_input=user_input,
            messages=st.session_state.messages,
        )

        user = _current_user()
        if user and user.role == ROLE_DEVELOPER:
            with st.sidebar:
                st.markdown("#### 🧩 Model debug")
                st.code(
                    f"{data['assistant_reply'][:120]}…\n"
                    f"text_opts: {len(data['text_suggestions'])}, emoji_opts: {len(data['emoji_suggestions'])}"
                )

        return data
    except Exception as exc:
        st.error(f"Model error: {exc}")
        return {
            "assistant_reply": "Der opstod en fejl.",
            "text_suggestions": [],
            "emoji_suggestions": [],
        }


def _check_openai_health():
    try:
        if not settings.api_key:
            return False, "Mangler AZURE_API_KEY"
        if not settings.endpoint:
            return False, "Mangler AZURE_ENDPOINT"
        if not settings.deployment:
            return False, "Mangler AZURE_DEPLOYMENT"
        if not settings.api_version:
            return False, "Mangler AZURE_API_VERSION"
        if not settings.endpoint.startswith("https://") or "openai.azure.com" not in settings.endpoint:
            return False, "AZURE_ENDPOINT ser ikke ud som en Azure OpenAI endpoint"
        return True, "Konfiguration ser gyldig ud"
    except Exception as exc:
        return False, f"Fejl: {exc}"


def _check_speech_health():
    try:
        from src.config import get_secret

        speech_key = get_secret("AZURE_SPEECH_KEY", "")
        speech_region = get_secret("AZURE_SPEECH_REGION", "")
        if not speech_key or not speech_region:
            return False, "Mangler AZURE_SPEECH_KEY eller AZURE_SPEECH_REGION"
        if speech_key.startswith("http"):
            return False, "AZURE_SPEECH_KEY ligner en URL og ikke en key"
        try:
            import azure.cognitiveservices.speech as speechsdk  # noqa: F401
        except Exception:
            return False, "Azure Speech SDK er ikke tilgængelig i miljøet"
        return True, "Konfiguration ser gyldig ud"
    except Exception as exc:
        return False, f"Fejl: {exc}"


def _check_transcriber_health():
    try:
        parsed = urlparse(TRANSCRIBER_WS)
        host = parsed.hostname or "localhost"
        port = parsed.port or 80
        with socket.create_connection((host, port), timeout=1.5):
            pass
        return True, f"Port {host}:{port} svarer"
    except Exception as exc:
        return False, f"Ingen forbindelse: {exc}"


def run_health_checks():
    return {
        "openai": _check_openai_health(),
        "speech": _check_speech_health(),
        "transcriber": _check_transcriber_health(),
    }


def _role_label(role: str) -> str:
    labels = {
        ROLE_PATIENT: "Patient",
        ROLE_THERAPIST: "Terapeut",
        ROLE_MANAGER: "Leder",
        ROLE_DEVELOPER: "Udvikler",
    }
    return labels.get(role, role)


def _set_logged_in_user(user: User) -> None:
    st.session_state.current_user = {
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
        "auth_source": user.auth_source,
        "therapist_username": user.therapist_username,
    }


def _current_user() -> User | None:
    raw = st.session_state.get("current_user")
    if not raw:
        return None
    return User(
        username=raw["username"],
        display_name=raw["display_name"],
        role=raw["role"],
        auth_source=raw["auth_source"],
        therapist_username=raw.get("therapist_username"),
    )


def _can(permission: str) -> bool:
    user = _current_user()
    if not user:
        return False
    return has_permission(user.role, permission)


def _log_activity(event_type: str, payload: dict | None = None) -> None:
    user = _current_user()
    if not user:
        return
    log_event(user.username, event_type, payload or {})


for key, default in {
    "messages": [],
    "text_opts": [],
    "emoji_opts": [],
    "input_mode": "Text",
    "listening": False,
    "sent_this_turn": False,
    "scenario_index": 0,
    "last_scenario_index": None,
    "use_custom_scenario": False,
    "custom_scenario": {},
    "health_results": None,
    "current_user": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


if not _current_user():
    st.title("🔐 Login")
    st.write("Patienter logger ind lokalt. Ansatte bruger SSO/Active Directory (simuleret provisionering).")
    st.info("Første opstart indeholder dev-konto: `devadmin / changeme123`.")

    patient_tab, employee_tab, create_tab = st.tabs(
        ["Patient login", "Ansat login (SSO/AD)", "Opret patient"]
    )
    with patient_tab:
        with st.form("patient_login_form", clear_on_submit=False):
            p_username = st.text_input("Brugernavn")
            p_password = st.text_input("Kodeord", type="password")
            p_login = st.form_submit_button("Log ind som patient")
        if p_login:
            user = authenticate_local_user(p_username, p_password)
            if not user or user.role != ROLE_PATIENT:
                st.error("Forkert login for patient.")
            else:
                _set_logged_in_user(user)
                st.rerun()

    with employee_tab:
        if not sso_domain_allowed("person@example.org"):
            st.warning("STAFF_EMAIL_DOMAIN er aktiv. Brug email fra korrekt domæne.")
        with st.form("employee_login_form", clear_on_submit=False):
            e_email = st.text_input("Arbejdsemail")
            e_role = st.selectbox(
                "Ansat-rolle",
                [ROLE_THERAPIST, ROLE_MANAGER, ROLE_DEVELOPER],
                format_func=_role_label,
            )
            e_login = st.form_submit_button("Log ind med SSO")
        if e_login:
            try:
                user = provision_sso_user(e_email, e_role)
                _set_logged_in_user(user)
                st.rerun()
            except Exception as exc:
                st.error(f"Kunne ikke logge ind: {exc}")

    with create_tab:
        st.caption("Patientoprettelse til test/drift. I produktion bør dette ligge bag terapeut/udvikler-login.")
        therapists = get_therapists()
        therapist_usernames = [u.username for u in therapists]
        if not therapist_usernames:
            st.warning("Ingen terapeuter fundet endnu. Opret/log ind en terapeut via SSO-fanen først.")
        else:
            with st.form("create_patient_form", clear_on_submit=True):
                new_username = st.text_input("Nyt patient-brugernavn")
                new_name = st.text_input("Visningsnavn")
                new_password = st.text_input("Midlertidigt kodeord", type="password")
                therapist_username = st.selectbox("Tilknyt terapeut", therapist_usernames)
                create_patient = st.form_submit_button("Opret patient")
            if create_patient:
                try:
                    create_local_user(
                        username=new_username,
                        password=new_password,
                        role=ROLE_PATIENT,
                        display_name=new_name,
                        therapist_username=therapist_username,
                    )
                    st.success("Patient oprettet.")
                except Exception as exc:
                    st.error(f"Kunne ikke oprette patient: {exc}")

    st.stop()


current_user = _current_user()

with st.sidebar:
    st.markdown(
        f"**{current_user.display_name}**  \n"
        f"Rolle: **{_role_label(current_user.role)}**  \n"
        f"Login: `{current_user.auth_source}`"
    )
    if st.button("Log ud", use_container_width=True):
        st.session_state.current_user = None
        for key in ["messages", "text_opts", "emoji_opts", "health_results"]:
            st.session_state[key] = [] if isinstance(st.session_state[key], list) else None
        st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <a href="mailto:anderssewerin@mac.com?subject=Feedback%20p%C3%A5%20samtaletr%C3%A6ner"
           style="display:inline-block;background:#2b7;color:#fff;
                  padding:8px 14px;border-radius:6px;text-decoration:none;
                  font-weight:600;margin-bottom:12px;">
           ✉️  Send feedback
        </a>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ Scenarietilstand")
    scenario_modes = ["Foruddefineret"]
    if _can("create_roleplay"):
        scenario_modes.append("Eget (ad-hoc)")
    if st.session_state.get("use_custom_scenario") and "Eget (ad-hoc)" not in scenario_modes:
        st.session_state.use_custom_scenario = False

    mode = st.radio(
        "Vælg type af scenarie",
        scenario_modes,
        index=0 if not st.session_state.get("use_custom_scenario") else 1,
    )

    st.session_state.use_custom_scenario = mode == "Eget (ad-hoc)"

    if not st.session_state.use_custom_scenario:
        st.markdown("### 🏪 Foruddefineret scenarie")

        if SCENARIOS:
            titles = [s["title"] for s in SCENARIOS]
            selected_title = st.selectbox(
                "Vælg en situation:",
                titles,
                index=0 if "scenario_index" not in st.session_state else st.session_state.scenario_index,
            )
            new_index = titles.index(selected_title)

            if (
                st.session_state.last_scenario_index is None
                or new_index != st.session_state.last_scenario_index
            ):
                st.session_state.scenario_index = new_index
                st.session_state.last_scenario_index = new_index
                for key in ["messages", "text_opts", "emoji_opts", "sent_this_turn"]:
                    st.session_state[key] = [] if isinstance(st.session_state[key], list) else False
                st.session_state.use_custom_scenario = False
                st.rerun()

            current_scenario = SCENARIOS[new_index]
            st.markdown(
                f"🗒️ **{current_scenario['title']}**  \n"
                f"{current_scenario.get('description', '(ingen beskrivelse)')}"
            )
        else:
            st.warning("Ingen scenarier fundet i ./scenarios/")
            current_scenario = None
    else:
        st.markdown("### 🧪 Eget scenarie")

        current_scenario = st.session_state.custom_scenario or {}
        if current_scenario.get("title") or current_scenario.get("description"):
            st.info(
                f"**Aktivt ad-hoc scenarie:**  \n"
                f"🧩 **{current_scenario.get('title', '(uden titel)')}**  \n"
                f"{current_scenario.get('description', '')}"
            )

        with st.form("custom_scenario_form", clear_on_submit=False):
            c_title = st.text_input("Titel", value=current_scenario.get("title", "Mit ad-hoc scenarie"))
            c_desc = st.text_area(
                "Beskrivelse",
                value=current_scenario.get("description", "Ad-hoc scenarie uden fil."),
            )
            c_spa = st.text_area(
                "System prompt-tilføjelse (system_prompt_addition)",
                value=current_scenario.get("system_prompt_addition", ""),
                help="Tilføjes nederst i den faste systemprompt.",
            )
            c_first = st.text_area(
                "Første besked (first_message)",
                value=current_scenario.get("first_message", ""),
                help="Valgfrit - bruges som åbningsreplik fra assistenten.",
            )
            try_it = st.form_submit_button("▶️ Prøv det")

        if try_it:
            st.session_state.custom_scenario = {
                "title": c_title.strip() or "Mit ad-hoc scenarie",
                "description": c_desc.strip(),
                "system_prompt_addition": c_spa.strip(),
                "first_message": c_first.strip(),
            }
            st.session_state.use_custom_scenario = True

            for key in ["messages", "text_opts", "emoji_opts", "sent_this_turn"]:
                st.session_state[key] = [] if isinstance(st.session_state[key], list) else False

            st.session_state.last_scenario_index = None
            current_scenario = st.session_state.custom_scenario
            st.rerun()

    st.header("🎛️ Indstillinger")
    new_listen = st.toggle("🎙 Taleinput (WebSocket)", value=st.session_state.listening)
    if new_listen != st.session_state.listening:
        st.session_state.listening = new_listen
        st.rerun()

    new_mode = st.radio(
        "Input-tilstand",
        ["Text", "Pictures (emoji)"],
        index=0 if st.session_state.input_mode == "Text" else 1,
    )
    if new_mode != st.session_state.input_mode:
        st.session_state.input_mode = new_mode
        st.rerun()

    st.markdown("---")
    if st.button("🔄 Nulstil samtale"):
        for key in ["messages", "text_opts", "emoji_opts"]:
            st.session_state[key] = []
        st.rerun()

    if current_user.role == ROLE_DEVELOPER:
        st.markdown("---")
        st.markdown("### 🩺 Health")
        if st.button("Kør health check", use_container_width=True):
            st.session_state.health_results = run_health_checks()

        health_results = st.session_state.get("health_results")
        if health_results:
            for label, (ok, details) in [
                ("OpenAI", health_results["openai"]),
                ("Speech", health_results["speech"]),
                ("Transcriber", health_results["transcriber"]),
            ]:
                icon = "✅" if ok else "❌"
                st.write(f"{icon} {label}: {details}")

    if current_user.role in {ROLE_THERAPIST, ROLE_MANAGER, ROLE_DEVELOPER}:
        st.markdown("---")
        st.markdown("### 📈 Data")
        if current_user.role == ROLE_THERAPIST:
            own_patients = get_patients_for_therapist(current_user.username)
            counts = get_activity_counts([p.username for p in own_patients])
            st.caption(f"Tilknyttede patienter: {len(own_patients)}")
            for patient in own_patients:
                st.write(f"- {patient.display_name} (`{patient.username}`): {counts.get(patient.username, 0)} events")
        elif current_user.role == ROLE_MANAGER:
            all_users = list_users()
            therapists = [u for u in all_users if u.role == ROLE_THERAPIST]
            patients = [u for u in all_users if u.role == ROLE_PATIENT]
            counts = get_activity_counts([p.username for p in patients])
            st.caption(f"Terapeuter: {len(therapists)}")
            st.caption(f"Patienter: {len(patients)}")
            st.caption(f"Total events: {sum(counts.values())}")
        else:
            all_users = list_users()
            counts = get_activity_counts()
            st.caption(f"Brugere i alt: {len(all_users)}")
            st.caption(f"Events i alt: {sum(counts.values())}")

    if current_user.role in {ROLE_THERAPIST, ROLE_DEVELOPER}:
        st.markdown("---")
        st.markdown("### 👥 Opret patient")
        with st.form("staff_create_patient_form", clear_on_submit=True):
            p_user = st.text_input("Patient-brugernavn")
            p_name = st.text_input("Patient-navn")
            p_pwd = st.text_input("Midlertidigt kodeord", type="password")
            if current_user.role == ROLE_THERAPIST:
                assigned_therapist = current_user.username
                st.caption(f"Tilknyttes automatisk: `{assigned_therapist}`")
            else:
                therapist_options = [u.username for u in get_therapists()]
                if therapist_options:
                    assigned_therapist = st.selectbox("Tilknyt terapeut", therapist_options)
                else:
                    assigned_therapist = ""
                    st.caption("Ingen terapeuter oprettet endnu.")
            create_from_staff = st.form_submit_button("Opret")
        if create_from_staff:
            try:
                create_local_user(
                    username=p_user,
                    password=p_pwd,
                    role=ROLE_PATIENT,
                    display_name=p_name,
                    therapist_username=assigned_therapist,
                )
                st.success("Patient oprettet.")
            except Exception as exc:
                st.error(f"Fejl ved oprettelse: {exc}")


st.title("🗣️ Aphasia Conversation Trainer")

if not _can("use_program"):
    st.info("Denne rolle har ikke adgang til selve træningsforløbet, men kan se data i sidepanelet.")
    st.stop()

if st.session_state.get("use_custom_scenario") and st.session_state.get("custom_scenario"):
    st.caption(f"Ad-hoc scenarie: **{st.session_state.custom_scenario.get('title', '(uden titel)')}**")
elif SCENARIOS:
    st.caption(f"Scenarie: **{SCENARIOS[st.session_state.scenario_index]['title']}**")

for message in st.session_state.messages:
    avatar = "🧩" if message["role"] == "assistant" else "👤"
    st.markdown(f"{avatar} **{message['role'].capitalize()}:** {message['content']}")


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
                    _log_activity("user_message", {"source": "speech", "text": text})
                    with st.spinner("Tænker..."):
                        reply = query_model(text)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply.get("assistant_reply", "")}
                    )
                    _log_activity("assistant_reply", {"source": "speech"})
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


if not st.session_state.messages:
    with st.spinner("Starter samtalen..."):
        if current_scenario and current_scenario.get("first_message"):
            first = current_scenario["first_message"]
            reply = query_model(first)
        else:
            reply = query_model("<session_start>")

    st.session_state.messages.append({"role": "assistant", "content": reply.get("assistant_reply", "")})
    _log_activity("session_start", {"scenario_mode": "custom" if st.session_state.get("use_custom_scenario") else "predefined"})
    _log_activity("assistant_reply", {"source": "session_start"})
    st.session_state.text_opts = reply.get("text_suggestions", [])
    st.session_state.emoji_opts = reply.get("emoji_suggestions", [])
    threading.Thread(target=speak, args=(reply.get("assistant_reply", ""),), daemon=True).start()
    st.rerun()


st.markdown("### Vælg et svar:")


def build_options():
    options = []
    if st.session_state.input_mode == "Text":
        options = [{"display": t, "meaning": t, "meta": None} for t in (st.session_state.text_opts or [])]
    else:
        emojis = st.session_state.emoji_opts or []
        texts = st.session_state.text_opts or []
        for i, emoji in enumerate(emojis):
            meaning = texts[i] if i < len(texts) else emoji
            options.append({"display": emoji, "meaning": meaning, "meta": None})
        if not options and texts:
            options = [{"display": "🗨️", "meaning": t, "meta": None} for t in texts[:5]]
        if not options:
            options = [{"display": "🤝", "meaning": "Hej", "meta": None}]
    return options


st.markdown("### Hurtige svar")
with st.container():
    st.markdown('<div class="meta-scope">', unsafe_allow_html=True)
    meta_cols = st.columns(len(UNIVERSAL_CHOICES))
    for i, opt in enumerate(UNIVERSAL_CHOICES):
        with meta_cols[i]:
            if st.button(opt["display"], key=f"meta_{i}", use_container_width=True):
                text_to_send = f"<meta:{opt['meta']}> {opt['meaning']}"
                st.markdown("</div>", unsafe_allow_html=True)
                st.session_state.messages.append({"role": "user", "content": text_to_send})
                _log_activity("user_message", {"source": "meta_button", "meta": opt["meta"], "text": opt["meaning"]})
                with st.spinner("Tænker..."):
                    reply = query_model(text_to_send)
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply.get("assistant_reply", "")}
                )
                _log_activity("assistant_reply", {"source": "meta_button"})
                st.session_state.text_opts = reply.get("text_suggestions", [])
                st.session_state.emoji_opts = reply.get("emoji_suggestions", [])
                threading.Thread(
                    target=speak,
                    args=(reply.get("assistant_reply", ""),),
                    daemon=True,
                ).start()
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("### Mulige svar")
with st.container():
    st.markdown('<div class="opts-scope">', unsafe_allow_html=True)
    options = build_options()
    num_cols = min(5, len(options)) or 1
    cols = st.columns(num_cols)
    for i, opt in enumerate(options):
        with cols[i % num_cols]:
            if st.button(opt["display"], key=f"opt_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": opt["meaning"]})
                _log_activity("user_message", {"source": "quick_option", "text": opt["meaning"]})
                with st.spinner("Tænker..."):
                    reply = query_model(opt["meaning"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply.get("assistant_reply", "")}
                )
                _log_activity("assistant_reply", {"source": "quick_option"})
                st.session_state.text_opts = reply.get("text_suggestions", [])
                st.session_state.emoji_opts = reply.get("emoji_suggestions", [])
                threading.Thread(
                    target=speak,
                    args=(reply.get("assistant_reply", ""),),
                    daemon=True,
                ).start()
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


meta_speak_texts = [opt.get("meaning") or opt.get("display", "") for opt in UNIVERSAL_CHOICES]
opts_speak_texts = [opt.get("meaning") or opt.get("display", "") for opt in options]

script = """
<script>
(function () {
  const SPEAK_DELAY = 2000;
  let timer = null, activeElem = null, lastSpoken = "";

  const d = (window.parent && window.parent.document) ? window.parent.document : document;

  function cancelTimer() {
    if (timer) clearTimeout(timer);
    timer = null;
    if (activeElem) { try { activeElem.style.outline = ""; } catch(e){} activeElem = null; }
  }

  function startTimer(elem) {
    cancelTimer();
    activeElem = elem;
    timer = setTimeout(() => {
      const text = elem.dataset.tts || elem.innerText || elem.textContent || "";
      if (text && text !== lastSpoken) {
        lastSpoken = text;
        try { elem.style.outline = "2px solid orange"; setTimeout(()=>{elem.style.outline="";}, 900); } catch(e){}
        fetch("http://localhost:__TTS_PORT__/_tts?text=" + encodeURIComponent(text)).catch(()=>{});
      }
    }, SPEAK_DELAY);
  }

  if (!window.parent.__ttsHoverInstalled) {
    window.parent.__ttsHoverInstalled = true;
    d.addEventListener("mouseover", (e) => {
      const btn = e.target.closest("button");
      if (btn && btn.textContent.trim() !== "") startTimer(btn);
    });
    d.addEventListener("mouseout", (e) => {
      if (e.target.closest("button")) cancelTimer();
    });
    d.addEventListener("touchstart", (e) => {
      const btn = e.target.closest("button");
      if (btn && btn.textContent.trim() !== "") startTimer(btn);
    }, {passive:true});
    d.addEventListener("touchend", () => cancelTimer(), {passive:true});
  }

  const metaTexts = __META__;
  const optTexts  = __OPTS__;

  function applyMappings() {
    try {
      const allButtons = d.querySelectorAll("button, [role=button]");
      let metaIndex = 0;
      let optIndex  = 0;

      allButtons.forEach((button) => {
        const label = (button.innerText || button.textContent || "").trim();
        if (["🆘","😕","👍","👎"].includes(label)) {
          if (metaTexts[metaIndex]) {
            button.dataset.tts = metaTexts[metaIndex];
          }
          metaIndex++;
        } else {
          if (optTexts[optIndex]) {
            button.dataset.tts = optTexts[optIndex];
          }
          optIndex++;
        }
      });
    } catch (e) {
      setTimeout(applyMappings, 400);
    }
  }

  applyMappings();
})();
</script>
"""

components.html(
    script.replace("__META__", json.dumps(meta_speak_texts))
    .replace("__OPTS__", json.dumps(opts_speak_texts))
    .replace("__TTS_PORT__", str(tts_port)),
    height=0,
)


def handle_user_input():
    user_text = st.session_state.manual_input.strip()
    if not user_text:
        return

    st.session_state.sent_this_turn = True
    st.session_state.messages.append({"role": "user", "content": user_text})
    _log_activity("user_message", {"source": "text_input", "text": user_text})

    with st.spinner("Tænker..."):
        reply = query_model(user_text)

    st.session_state.messages.append({"role": "assistant", "content": reply.get("assistant_reply", "")})
    _log_activity("assistant_reply", {"source": "text_input"})
    st.session_state.text_opts = reply.get("text_suggestions", [])
    st.session_state.emoji_opts = reply.get("emoji_suggestions", [])
    threading.Thread(target=speak, args=(reply.get("assistant_reply", ""),), daemon=True).start()

    st.session_state.manual_input = ""
    st.session_state.sent_this_turn = False


st.text_input(
    "Skriv selv:",
    key="manual_input",
    on_change=handle_user_input,
)

st.markdown(
    """
<style>
div[data-testid="column"] {
    display: flex;
    justify-content: center;
}

.meta-scope div.stButton > button,
.meta-scope div.stButton > button * {
    width: 200px !important;
    height: 220px !important;
    margin: 8px !important;
    border-radius: 20px !important;
    background-color: #e5f1ff !important;
    border: 3px solid #5b9bff !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.15) !important;

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

.meta-scope div.stButton > button:hover {
    background-color: #d6e8ff !important;
    transform: scale(1.04);
}

.opts-scope div.stButton > button:hover {
    background-color: #f0f0f0 !important;
}
</style>
""",
    unsafe_allow_html=True,
)
