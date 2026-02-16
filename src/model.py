import json
from typing import Any, Dict, List

from openai import AzureOpenAI

from src.config import AzureSettings


def create_client(settings: AzureSettings) -> AzureOpenAI:
    return AzureOpenAI(
        api_version=settings.api_version,
        azure_endpoint=settings.endpoint,
        api_key=settings.api_key,
    )


def query_model(
    *,
    client: AzureOpenAI,
    deployment: str,
    system_prompt: str,
    user_input: str,
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Send full chat history to Azure and parse strict JSON response."""
    history = [{"role": "system", "content": system_prompt}]
    for message in messages:
        history.append({"role": message["role"], "content": message["content"]})
    history.append({"role": "user", "content": user_input})

    completion = client.chat.completions.create(
        model=deployment,
        messages=history,
        temperature=0.2,
        max_tokens=400,
        response_format={"type": "json_object"},
    )

    raw = completion.choices[0].message.content or ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "assistant_reply": raw,
            "text_suggestions": [],
            "emoji_suggestions": [],
        }

    data.setdefault("assistant_reply", "")
    data.setdefault("text_suggestions", [])
    data.setdefault("emoji_suggestions", [])
    return data
