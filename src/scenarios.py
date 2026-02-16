import glob
import json

import streamlit as st


SCENARIO_DIR = "scenarios"


def load_scenarios():
    scenarios = []
    for file_path in glob.glob(f"{SCENARIO_DIR}/*.json"):
        with open(file_path, "r", encoding="utf-8") as infile:
            try:
                data = json.load(infile)
                scenarios.append(data)
            except Exception as exc:
                st.warning(f"Kunne ikke indlaese {file_path}: {exc}")
    return sorted(scenarios, key=lambda scenario: scenario["title"])
