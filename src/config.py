import os
from dataclasses import dataclass

import streamlit as st


def get_secret(name: str, default=None):
    """Read a secret from env vars first, then Streamlit secrets."""
    if name in os.environ:
        return os.environ[name]
    if name in st.secrets:
        return st.secrets[name]
    if default is not None:
        return default
    raise KeyError(f"Missing required secret: {name}")


@dataclass(frozen=True)
class AzureSettings:
    api_key: str
    endpoint: str
    deployment: str
    api_version: str


def load_azure_settings() -> AzureSettings:
    return AzureSettings(
        api_key=get_secret("AZURE_API_KEY"),
        endpoint=get_secret("AZURE_ENDPOINT"),
        deployment=get_secret("AZURE_DEPLOYMENT"),
        api_version=get_secret("AZURE_API_VERSION"),
    )
