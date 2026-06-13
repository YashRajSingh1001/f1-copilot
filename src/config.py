"""
Central config — reads from st.secrets (Streamlit Cloud) with .env fallback for local dev.
Import get() everywhere instead of os.getenv() directly.
"""

import os


def get(key: str, default: str = "") -> str:
    """Read a secret: st.secrets first (Streamlit Cloud), then env var, then default."""
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)
