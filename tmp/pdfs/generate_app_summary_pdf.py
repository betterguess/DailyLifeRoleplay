import os
import textwrap

OUT = "output/pdf/app-summary-one-page.pdf"
os.makedirs("output/pdf", exist_ok=True)

PAGE_W = 612
PAGE_H = 792
LEFT = 44
TOP = 750
LINE_H = 12

sections = [
    ("title", "DailyLifeRoleplay App Summary"),
    ("body", "Repository analyzed: /Users/magnus/Work/DailyLifeRoleplay"),
    ("h", "What It Is"),
    ("body", "A local Streamlit proof-of-concept conversation trainer for Danish aphasia practice, focused on everyday roleplay scenarios. It combines chat, selectable response options, optional speech input via WebSocket, and spoken output via local/Azure TTS paths."),
    ("h", "Who It's For"),
    ("body", "Primary persona: a Danish-speaking person with aphasia practicing short, structured daily-life dialogues, often with caregiver or therapist support."),
    ("h", "What It Does"),
]

features = [
    "Runs a Streamlit conversation UI with message history and assistant turns.",
    "Loads predefined scenario JSON files from scenarios/*.json.",
    "Supports ad-hoc custom scenario setup in the sidebar.",
    "Offers two input modes: text suggestions or emoji-style option buttons.",
    "Provides universal quick actions: Hjaelp, Forstaar ikke, Ja, Nej.",
    "Can listen to speech input from ws://localhost:9000/transcribe.",
    "Speaks replies via Azure Speech when configured, with pyttsx3 fallback.",
]

architecture = [
    "UI layer: app.py (Streamlit) manages state, options, and scenario mode.",
    "Scenario data: scenarios/*.json adds per-scenario prompt additions and first message.",
    "Model layer: AzureOpenAI chat.completions returns strict JSON with reply + suggestions.",
    "Speech-in path: WebSocket client consumes partial/final transcript events.",
    "Speech-out path: local HTTP /_tts handlers call Azure Speech SDK or pyttsx3.",
    "Not found in repo: realtime_transcriber.py implementation referenced in README.",
    "Evidence mismatch: README describes Ollama flow; app.py uses Azure OpenAI client.",
]

run_steps = [
    "1. cd /Users/magnus/Work/DailyLifeRoleplay",
    "2. python3 -m venv .venv && source .venv/bin/activate",
    "3. pip install -r requirements.txt",
    "4. Set AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT, AZURE_API_VERSION",
    "5. Optional speech service at ws://localhost:9000/transcribe (Not found in repo)",
    "6. python -m streamlit run app.py",
]

lines = []


def add_wrapped(text, width, prefix="", style="body"):
    wrapped = textwrap.wrap(text, width=width)
    if not wrapped:
        lines.append((style, ""))
        return
    first = True
    for w in wrapped:
        if first:
            lines.append((style, prefix + w))
            first = False
        else:
            lines.append((style, (" " * len(prefix)) + w))

for typ, content in sections:
    if typ == "title":
        lines.append(("title", content))
    elif typ == "h":
        lines.append(("space", ""))
        lines.append(("h", content))
    else:
        add_wrapped(content, width=96, style="body")

for item in features:
    add_wrapped(item, width=92, prefix="- ", style="body")

lines.append(("space", ""))
lines.append(("h", "How It Works (Repo-Evidenced Architecture)"))
for item in architecture:
    add_wrapped(item, width=92, prefix="- ", style="body")

lines.append(("space", ""))
lines.append(("h", "How To Run (Minimal Getting Started)"))
for step in run_steps:
    add_wrapped(step, width=95, style="mono")

lines.append(("space", ""))
add_wrapped("Source basis: README.md, app.py, Dockerfile, requirements.txt, scenarios/*.json", width=96, style="body")

# Ensure one page by tightening line-height/font if needed
max_lines = int((TOP - 44) / LINE_H)
if len(lines) > max_lines:
    LINE_H = 11
    max_lines = int((TOP - 40) / LINE_H)

if len(lines) > max_lines:
    # Trim lowest-priority tail safely while preserving required sections
    while len(lines) > max_lines and lines:
        lines.pop(-2)


def esc(s: str) -> str:
    return s.replace('\\', r'\\').replace('(', r'\(').replace(')', r'\)')

content = ["BT"]
y = TOP
for style, text in lines:
    if style == "space":
        y -= LINE_H // 2
        continue
    if y < 36:
        break
    if style == "title":
        content.append("/F2 17 Tf")
    elif style == "h":
        content.append("/F2 11 Tf")
    elif style == "mono":
        content.append("/F3 8 Tf")
    else:
        content.append("/F1 9 Tf")
    content.append(f"1 0 0 1 {LEFT} {y} Tm")
    content.append(f"({esc(text)}) Tj")
    y -= LINE_H
content.append("ET")
stream = "\n".join(content).encode("latin-1", errors="replace")

objs = []
objs.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
objs.append(b"2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n")
objs.append(f"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] /Resources << /Font << /F1 4 0 R /F2 5 0 R /F3 6 0 R >> >> /Contents 7 0 R >>\nendobj\n".encode())
objs.append(b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
objs.append(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n")
objs.append(b"6 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>\nendobj\n")
objs.append(f"7 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream\nendobj\n")

pdf = bytearray(b"%PDF-1.4\n")
offsets = [0]
for obj in objs:
    offsets.append(len(pdf))
    pdf.extend(obj)

xref_pos = len(pdf)
pdf.extend(f"xref\n0 {len(objs)+1}\n".encode())
pdf.extend(b"0000000000 65535 f \n")
for off in offsets[1:]:
    pdf.extend(f"{off:010d} 00000 n \n".encode())
pdf.extend(f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode())

with open(OUT, "wb") as f:
    f.write(pdf)

print(OUT)
