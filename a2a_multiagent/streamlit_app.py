"""
A2A Multi-Agent Chatbot — Streamlit UI

This UI connects to the A2A multi-agent backend with auto-routing.
Follows the A2A protocol task/message format and streams routed responses.

Run:  streamlit run streamlit_app.py
Needs server.py running at API_URL (default localhost:8000)
"""

import os
import json
import uuid
import html
import re
import hashlib
import time
import requests
import streamlit as st
from huggingface_hub import InferenceClient

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

API_URL = os.getenv("API_URL", "http://localhost:8000")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_ASR_MODEL = os.getenv("HF_ASR_MODEL", "openai/whisper-large-v3-turbo")

AGENT_CONFIG = {
    "orchestrator": {
        "icon": "🤖✨", "color": "#f9a8d4",
        "label": "Avery",
        "desc":  "Swiftly coordinates the full parallel response",
    },
    "research-agent": {
        "icon": "🦉📚", "color": "#93c5fd",
        "label": "Riley",
        "desc":  "Wisdom-driven research and fact gathering",
    },
    "summarizer-agent": {
        "icon": "🌸📝", "color": "#fbcfe8",
        "label": "Casey",
        "desc":  "Turns complex findings into calm, clear summaries",
    },
    "general-qa-agent": {
        "icon": "☀️💬", "color": "#fde68a",
        "label": "Jordan",
        "desc":  "Cheerful conversational help for quick questions",
    },
}

INTENT_META = {
    "research_and_summarize": ("🔬 Research + Summarize", "#93c5fd"),
    "research_only":          ("🦉 Research Only",       "#bfdbfe"),
    "general_qa":             ("💬 Direct Answer",       "#f9a8d4"),
    "summarize_only":         ("📝 Summarize Only",      "#86efac"),
}

STARTERS = [
    "What is the A2A protocol and how does it differ from MCP?",
    "Explain how LangGraph works under the hood",
    "What are the best open-source LLMs on HuggingFace right now?",
    "How does RAG (Retrieval-Augmented Generation) work?",
    "Compare Mistral 7B vs Llama 3 8B for production use",
]

THEMES = {
    "harrypotter": {
        "label": "NOX",
        "icon": "🪄",
        "main_bg": "radial-gradient(circle at 12% 10%, rgba(250, 204, 21, 0.14), transparent 30%), radial-gradient(circle at 88% 14%, rgba(147, 51, 234, 0.18), transparent 36%), linear-gradient(180deg, #0b1023 0%, #15112d 52%, #1b1234 100%)",
        "sidebar_bg": "linear-gradient(180deg, #111827 0%, #1f1b3a 100%)",
        "button_bg": "linear-gradient(135deg, #3f2a6b 0%, #6d28d9 100%)",
        "hero_glow": "rgba(147, 51, 234, 0.35)",
        "hero_icon": "🪄🌙",
        "user_bubble": "linear-gradient(135deg, #2f2a52 0%, #3c2b6e 100%)",
        "agent_bubble": "rgba(22, 24, 43, 0.92)",
        "text_primary": "#e5e7eb",
        "text_muted": "#cbd5e1",
        "sidebar_text": "#e2e8f0",
        "sidebar_border": "#3b2f66",
        "card_bg": "rgba(29, 26, 53, 0.92)",
        "card_border": "#4c3f7a",
        "agent_icons": {
            "orchestrator": "🧙",
            "research-agent": "🪄",
            "summarizer-agent": "🧪",
            "general-qa-agent": "✨",
        },
        "agent_names": {
            "orchestrator": "Rowan Spellguide",
            "research-agent": "Mia Lorefinder",
            "summarizer-agent": "Nina Brewnote",
            "general-qa-agent": "Leo Charmchat",
        },
    },
    "nfc_playback_video_game": {
        "label": "NFX",
        "icon": "🎮",
        "main_bg": "radial-gradient(circle at 10% 8%, rgba(0, 255, 240, 0.2), transparent 28%), radial-gradient(circle at 90% 12%, rgba(255, 0, 212, 0.22), transparent 34%), linear-gradient(180deg, #060b18 0%, #0a1228 52%, #0d0f1f 100%)",
        "sidebar_bg": "linear-gradient(180deg, #071220 0%, #101638 100%)",
        "button_bg": "linear-gradient(135deg, #00c2ff 0%, #c026d3 100%)",
        "hero_glow": "rgba(0, 255, 240, 0.3)",
        "hero_icon": "🎮⚡",
        "user_bubble": "linear-gradient(135deg, #0f2a44 0%, #2b1454 100%)",
        "agent_bubble": "rgba(12, 20, 40, 0.94)",
        "text_primary": "#e2f7ff",
        "text_muted": "#a5f3fc",
        "sidebar_text": "#c4f1ff",
        "sidebar_border": "#1e3a8a",
        "card_bg": "rgba(14, 25, 51, 0.94)",
        "card_border": "#2563eb",
        "agent_icons": {
            "orchestrator": "🕹️",
            "research-agent": "🧠",
            "summarizer-agent": "💾",
            "general-qa-agent": "🎯",
        },
        "agent_names": {
            "orchestrator": "Dex Control",
            "research-agent": "Kai Scout",
            "summarizer-agent": "Milo Snapshot",
            "general-qa-agent": "Zoe Quickchat",
        },
    },
    "corporate": {
        "label": "PRM",
        "icon": "🏢",
        "main_bg": "radial-gradient(circle at 10% 8%, rgba(14, 165, 233, 0.14), transparent 30%), radial-gradient(circle at 90% 16%, rgba(148, 163, 184, 0.16), transparent 32%), linear-gradient(180deg, #f8fafc 0%, #f1f5f9 50%, #f8fafc 100%)",
        "sidebar_bg": "linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%)",
        "button_bg": "linear-gradient(135deg, #e2e8f0 0%, #dbeafe 100%)",
        "hero_glow": "rgba(14, 165, 233, 0.12)",
        "hero_icon": "🏢📊",
        "user_bubble": "linear-gradient(135deg, #e2e8f0 0%, #dbeafe 100%)",
        "agent_bubble": "rgba(255, 255, 255, 0.94)",
        "text_primary": "#1f2937",
        "text_muted": "#475569",
        "sidebar_text": "#334155",
        "sidebar_border": "#cbd5e1",
        "card_bg": "rgba(255, 255, 255, 0.94)",
        "card_border": "#cbd5e1",
        "agent_icons": {
            "orchestrator": "🧭",
            "research-agent": "📈",
            "summarizer-agent": "🗂️",
            "general-qa-agent": "🤝",
        },
        "agent_names": {
            "orchestrator": "Avery Ops",
            "research-agent": "Riley Insights",
            "summarizer-agent": "Casey Brief",
            "general-qa-agent": "Jordan Desk",
        },
    },
}

# ─────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="A2A Multi-Agent Chatbot",
    page_icon="🤖✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme" not in st.session_state:
    st.session_state.theme = "harrypotter"

theme_key = st.session_state.theme if st.session_state.theme in THEMES else "harrypotter"
theme_cfg = THEMES[theme_key]

st.markdown("""
<style>
/* ── Global ──────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', system-ui, sans-serif !important;
    background:
        radial-gradient(circle at top left, rgba(251, 191, 36, 0.18), transparent 32%),
        radial-gradient(circle at top right, rgba(125, 211, 252, 0.18), transparent 28%),
        linear-gradient(180deg, #fffdf8 0%, #f9fbff 48%, #fffafc 100%);
    color: #334155;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(251, 191, 36, 0.18), transparent 32%),
        radial-gradient(circle at top right, rgba(125, 211, 252, 0.18), transparent 28%),
        linear-gradient(180deg, #fffdf8 0%, #f9fbff 48%, #fffafc 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fffefb 0%, #f8fbff 100%) !important;
    border-right: 1px solid #e7edf7 !important;
}

/* Hide non-essential Streamlit chrome, keep header for sidebar toggle */
#MainMenu, footer { visibility: hidden; }
header {
    visibility: visible;
    background: transparent !important;
}
[data-testid="stToolbar"] { display: none; }
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
}

/* Main area bg */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    max-width: 860px;
}

div.stButton > button {
    background: linear-gradient(135deg, #fff4cc 0%, #dbeafe 100%);
    color: #334155;
    border: 1px solid #d7e3f3;
    border-radius: 14px;
    box-shadow: 0 8px 24px rgba(148, 163, 184, 0.16);
    transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}

div.stButton > button:hover {
    transform: translateY(-1px);
    border-color: #cbd5e1;
    box-shadow: 0 12px 28px rgba(148, 163, 184, 0.2);
}

div[data-testid="stChatInput"] {
    background: transparent;
    border: 1px solid #334155;
    border-radius: 18px;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.18);
    padding-left: 54px;
    padding-right: 54px;
}

[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] div[data-baseweb="textarea"],
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input {
    background: transparent !important;
    color: #0f172a !important;
}

[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInput"] input::placeholder {
    color: #475569 !important;
}

[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
[data-testid="stChatInputContainer"] {
    background: transparent !important;
}

[data-testid="stBottomBlockContainer"] {
    position: relative;
    display: flex;
    align-items: flex-end;
    gap: 10px;
}

div[data-testid="stChatInput"] {
    flex: 1 1 auto;
}

/* Group voice control with the chat input row */
div[data-testid="stAudioInput"] {
    position: relative;
    left: auto;
    right: auto;
    bottom: auto;
    z-index: 2;
    background: transparent !important;
    margin-bottom: 2px;
    flex: 0 0 auto;
}

@media (max-width: 980px) {
    [data-testid="stBottomBlockContainer"] {
        gap: 8px;
    }

    div[data-testid="stAudioInput"] {
        margin-bottom: 1px;
    }
}

div[data-testid="stAudioInput"] label {
    display: none !important;
}

div[data-testid="stAudioInput"] > div,
div[data-testid="stAudioInput"] [data-baseweb="file-uploader"] {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}

div[data-testid="stAudioInput"] button {
    width: 34px !important;
    height: 34px !important;
    min-height: 34px !important;
    border-radius: 9px !important;
    border: 1px solid #94a3b8 !important;
    background: transparent !important;
    color: #0f172a !important;
    padding: 0 !important;
}

/* ── Agent pipeline card ─────────────────────────────── */
.pipeline-wrap {
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid #e6edf7;
    box-shadow: 0 12px 30px rgba(148, 163, 184, 0.12);
    border-radius: 18px;
    padding: 14px 18px;
    margin-bottom: 12px;
    font-family: 'Space Grotesk', sans-serif;
}

.pipeline-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 12px;
}

.pipeline-steps {
    display: flex;
    align-items: center;
    gap: 0;
    flex-wrap: wrap;
}

.pipeline-step {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 500;
    border: 1px solid;
    white-space: nowrap;
}

.pipeline-arrow {
    color: #94a3b8;
    font-size: 14px;
    padding: 0 4px;
}

.pipeline-reason {
    margin-top: 8px;
    font-size: 12px;
    color: #64748b;
    font-style: italic;
    font-family: 'IBM Plex Mono', monospace;
    padding-left: 2px;
}

/* ── Chat bubbles ────────────────────────────────────── */
.msg-user {
    background: linear-gradient(135deg, #fef3c7 0%, #e0f2fe 100%);
    border: 1px solid #dbe7f4;
    border-radius: 18px 18px 4px 18px;
    box-shadow: 0 8px 20px rgba(148, 163, 184, 0.12);
    padding: 13px 16px;
    margin: 6px 0 6px 60px;
    font-size: 14px;
    line-height: 1.65;
    color: #334155;
}

.msg-agent {
    background: linear-gradient(135deg, #2f2a52 0%, #3c2b6e 100%);
    border: 1px solid #1e293b;
    border-radius: 4px 18px 18px 18px;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.3);
    padding: 14px 18px;
    margin: 6px 60px 6px 0;
    font-size: 14px;
    line-height: 1.7;
    color: #f8fafc;
}

.msg-agent code {
    background: #f8fafc;
    border: 1px solid #dbe4ef;
    border-radius: 4px;
    padding: 1px 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    color: #2563eb;
}

.msg-agent pre {
    background: #f8fbff;
    border: 1px solid #dbe4ef;
    border-radius: 8px;
    padding: 14px;
    overflow-x: auto;
    margin: 10px 0;
}

.msg-agent pre code {
    background: transparent;
    border: none;
    padding: 0;
    font-size: 12.5px;
    color: #0f766e;
}

.msg-agent p,
.msg-agent li,
.msg-agent h1,
.msg-agent h2,
.msg-agent h3,
.msg-agent h4,
.msg-agent strong,
.msg-agent em {
    color: #f8fafc !important;
}

/* ── Intent badge ────────────────────────────────────── */
.intent-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    border: 1px solid;
}

/* ── Sidebar agent card ──────────────────────────────── */
.agent-sidebar-card {
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid #e4ebf5;
    border-radius: 16px;
    padding: 10px 12px;
    margin-bottom: 6px;
    transition: border-color .2s, transform .2s, box-shadow .2s;
    box-shadow: 0 10px 22px rgba(148, 163, 184, 0.08);
}

.agent-sidebar-card.active {
    border-color: #c7d2fe;
    background: linear-gradient(135deg, rgba(255, 247, 237, 0.95) 0%, rgba(239, 246, 255, 0.95) 100%);
    transform: translateY(-1px);
}

.agent-card-header {
    display: flex;
    align-items: center;
    gap: 7px;
    margin-bottom: 3px;
}

.agent-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}

.agent-card-name {
    font-size: 13px;
    font-weight: 500;
    color: #334155;
}

.agent-card-model {
    font-size: 10px;
    color: #64748b;
    font-family: 'IBM Plex Mono', monospace;
    margin-left: 14px;
}

.agent-card-desc {
    font-size: 11px;
    color: #64748b;
    margin-left: 14px;
    line-height: 1.4;
}

/* ── Stat chips ──────────────────────────────────────── */
.stat-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 4px;
}

.stat-chip {
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid #dbe4ef;
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 12px;
    color: #64748b;
    font-family: 'IBM Plex Mono', monospace;
}

.stat-chip span {
    color: #334155;
    font-weight: 500;
}

/* ── Starter chips ───────────────────────────────────── */
.starter-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 12px;
}

.starter-chip {
    background: rgba(255, 255, 255, 0.85);
    border: 1px solid #dbe4ef;
    border-radius: 14px;
    padding: 10px 13px;
    font-size: 12.5px;
    color: #475569;
    cursor: pointer;
    line-height: 1.5;
    box-shadow: 0 10px 22px rgba(148, 163, 184, 0.08);
}

/* ── Hero ────────────────────────────────────────────── */
.hero {
    text-align: center;
    padding: 48px 0 32px;
    color: #475569;
}

.hero-icon { font-size: 54px; opacity: .95; margin-bottom: 12px; }
.hero-title { font-size: 22px; font-weight: 600; color: #334155; margin-bottom: 8px; }
.hero-sub { font-size: 13.5px; color: #64748b; max-width: 480px; margin: 0 auto; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<style>
.stApp {{
    background: {theme_cfg["main_bg"]};
}}

html, body, [class*="css"] {{
    background: {theme_cfg["main_bg"]};
}}

[data-testid="stSidebar"] {{
    background: {theme_cfg["sidebar_bg"]} !important;
    min-width: 320px;
    max-width: 320px;
    color: {theme_cfg.get("sidebar_text", "#334155")};
    transform: translateX(0) !important;
    z-index: 999 !important;
    border-right: 1px solid {theme_cfg.get("sidebar_border", "#334155")};
    box-shadow: 8px 0 24px rgba(0, 0, 0, 0.25);
}}

[data-testid="stSidebar"][aria-expanded="false"] {{
    min-width: 320px;
    max-width: 320px;
    transform: translateX(0) !important;
}}

[data-testid="stSidebarContent"] {{
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    padding: 0.35rem 0.55rem 1rem 0.55rem;
}}

[data-testid="stSidebarContent"] * {{
    color: {theme_cfg.get("sidebar_text", "#334155")};
}}

div.stButton > button {{
    background: {theme_cfg["button_bg"]};
    color: {theme_cfg.get("text_primary", "#334155")};
}}

.msg-user {{
    background: {theme_cfg["user_bubble"]};
    color: {theme_cfg.get("text_primary", "#334155")};
}}

.msg-agent {{
    background: {theme_cfg["user_bubble"]};
    border-color: #1e293b;
    color: #f8fafc;
}}

.hero-title, .hero-sub, .agent-card-name, .agent-card-model, .agent-card-desc, .stat-chip, .pipeline-title {{
    color: {theme_cfg.get("text_muted", "#64748b")} !important;
}}

.hero-title {{
    color: {theme_cfg.get("text_primary", "#334155")} !important;
}}

.main .block-container {{
    max-width: 920px;
    padding-top: 1.25rem;
    padding-bottom: 1.25rem;
}}

.agent-sidebar-card {{
    background: {theme_cfg.get("card_bg", "rgba(255,255,255,0.88)")} !important;
    border-color: {theme_cfg.get("card_border", "#dbe4ef")} !important;
    padding: 12px 13px;
    margin-bottom: 8px;
}}

.agent-card-name {{
    color: {theme_cfg.get("text_primary", "#334155")} !important;
    font-weight: 600;
}}

.agent-card-model,
.agent-card-desc,
.stat-chip,
.stat-chip span {{
    color: {theme_cfg.get("text_muted", "#64748b")} !important;
}}

div[data-testid="stChatInput"] {{
    background: transparent !important;
    border-color: #1f2937 !important;
    box-shadow: none !important;
    padding-left: 54px !important;
    padding-right: 54px !important;
}}

[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] div[data-baseweb="textarea"],
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input,
[data-testid="stChatInputContainer"],
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"] {{
    background: transparent !important;
    color: #0f172a !important;
}}

[data-testid="stBottomBlockContainer"] {{
    position: relative;
    display: flex;
    align-items: flex-end;
    gap: 10px;
}}

div[data-testid="stChatInput"] {{
    flex: 1 1 auto;
}}

[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInput"] input::placeholder {{
    color: #475569 !important;
}}

div[data-testid="stAudioInput"] {{
    position: relative;
    left: auto;
    right: auto;
    bottom: auto;
    z-index: 2;
    background: transparent !important;
    margin-bottom: 2px;
    flex: 0 0 auto;
}}

@media (max-width: 980px) {{
    [data-testid="stBottomBlockContainer"] {{
        gap: 8px;
    }}

    div[data-testid="stAudioInput"] {{
        margin-bottom: 1px;
    }}
}}

div[data-testid="stAudioInput"] label {{
    display: none !important;
}}

div[data-testid="stAudioInput"] > div,
div[data-testid="stAudioInput"] [data-baseweb="file-uploader"] {{
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}}

div[data-testid="stAudioInput"] button {{
    width: 34px !important;
    height: 34px !important;
    min-height: 34px !important;
    border-radius: 9px !important;
    border: 1px solid #94a3b8 !important;
    background: transparent !important;
    color: #0f172a !important;
    padding: 0 !important;
}}

.hero {{
    position: relative;
}}

.hero::before {{
    content: "";
    position: absolute;
    inset: 12% 20% auto 20%;
    height: 90px;
    border-radius: 999px;
    background: {theme_cfg["hero_glow"]};
    filter: blur(20px);
    z-index: -1;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────

defaults = {
    "messages":      [],           # [{role, content, meta}]
    "history":       [],           # raw user prompts
    "session_id":    str(uuid.uuid4()),
    "agents":        [],
    "total_queries": 0,
    "active_agents": set(),
    "_last_voice_hash": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def fetch_agents():
    try:
        r = requests.get(f"{API_URL}/agents", timeout=3)
        st.session_state.agents = r.json().get("agents", [])
    except Exception:
        st.session_state.agents = []


def simple_voice_nlp(text: str) -> str:
    """
    Simple NLP fallback to clean and extract meaning from voice input.
    """
    if not text or not text.strip():
        return ""
    text = text.strip()
    text = ' '.join(text.split())  # normalize whitespace
    text = text[0].upper() + text[1:] if text else text  # capitalize
    return text


def transcribe_voice_with_hf(audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
    last_error = None
    candidate_models = []
    for model in [
        HF_ASR_MODEL,
        "openai/whisper-large-v3-turbo",
        "distil-whisper/distil-large-v3",
    ]:
        if model and model not in candidate_models:
            candidate_models.append(model)

    # Primary path: official HF client (tokened access only).
    if HF_TOKEN:
        try:
            client = InferenceClient(api_key=HF_TOKEN, timeout=180)
            for model in candidate_models:
                for attempt in range(3):
                    try:
                        result = client.automatic_speech_recognition(
                            audio=audio_bytes,
                            model=model,
                        )
                        if isinstance(result, dict):
                            text = str(result.get("text", "")).strip()
                        else:
                            text = str(getattr(result, "text", "")).strip()
                        if text:
                            return text
                    except Exception as e:
                        last_error = e
                        msg = str(e).lower()
                        if "loading" in msg or "503" in msg or "not ready" in msg:
                            time.sleep(2 * (attempt + 1))
                            continue
                        break
        except Exception as e:
            last_error = e

    # Fallback path: raw inference HTTP call.
    headers = {
        "Content-Type": mime_type or "application/octet-stream",
        "Accept": "application/json",
    }
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    endpoints = ["https://api-inference.huggingface.co/models/{model}"]
    endpoints.append("https://router.huggingface.co/hf-inference/models/{model}")
    for model in candidate_models:
        for endpoint in endpoints:
            try:
                resp = requests.post(
                    endpoint.format(model=model),
                    headers=headers,
                    data=audio_bytes,
                    timeout=180,
                )
                if resp.ok:
                    data = resp.json()
                    if isinstance(data, dict):
                        text = str(data.get("text", "")).strip()
                        if text:
                            return text
                    elif isinstance(data, list) and data and isinstance(data[0], dict):
                        text = str(data[0].get("text", "")).strip()
                        if text:
                            return text
                else:
                    last_error = f"{resp.status_code}: {resp.text[:160]}"
                    if resp.status_code in (401, 403):
                        continue
            except Exception as e:
                last_error = e

    raise RuntimeError(f"Voice transcription unavailable right now ({last_error}).")


def fallback_agent_cards() -> list:
    return [
        {
            "agent_id": agent_id,
            "name": cfg.get("label", agent_id),
            "model": "local",
        }
        for agent_id, cfg in AGENT_CONFIG.items()
    ]

def check_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.json()
    except Exception:
        return None

def render_pipeline_card(agents_used: list, intent: str, reasoning: str):
    steps_html = ""
    for i, agent_id in enumerate(agents_used):
        cfg = AGENT_CONFIG.get(agent_id, {"icon": "🤖✨", "color": "#94a3b8", "label": agent_id})
        color = cfg["color"]
        bg    = color + "18"
        steps_html += f'''
        <span class="pipeline-step" style="color:{color};border-color:{color}40;background:{bg}">
            {cfg["icon"]} {cfg["label"]}
        </span>'''
        if i < len(agents_used) - 1:
            steps_html += '<span class="pipeline-arrow">›</span>'

    intent_label, intent_color = INTENT_META.get(intent, ("", "#888"))
    reason_html = f'<div class="pipeline-reason">💡 {reasoning}</div>' if reasoning else ""

    return f"""
    <div class="pipeline-wrap">
        <div class="pipeline-title">Agent Pipeline
            &nbsp;&nbsp;
            <span class="intent-badge" style="color:{intent_color};border-color:{intent_color}40;background:{intent_color}18">
                {intent_label}
            </span>
        </div>
        <div class="pipeline-steps">{steps_html}</div>
        {reason_html}
    </div>"""


def render_assistant_message(text: str, streaming: bool = False):
    suffix = "▌" if streaming else ""
    st.markdown(f'<div class="msg-agent">{text}{suffix}</div>', unsafe_allow_html=True)


def themed_agent_icon(agent_id: str) -> str:
    return theme_cfg.get("agent_icons", {}).get(agent_id, AGENT_CONFIG.get(agent_id, {}).get("icon", "🤖"))


def themed_agent_name(agent_id: str) -> str:
    return theme_cfg.get("agent_names", {}).get(
        agent_id,
        AGENT_CONFIG.get(agent_id, {}).get("label", agent_id),
    )


def apply_theme_to_agent_headers(text: str) -> str:
    if not text:
        return text
    replacements = {
        "general-qa-agent": ["Euphrosyne", "Jordan", "Assistant"],
        "research-agent": ["Sophia", "Riley", "Researcher"],
        "summarizer-agent": ["Eunoia", "Casey", "Summarizer"],
        "orchestrator": ["Hermes", "Avery", "Orchestrator", "Router"],
    }
    out = text
    for agent_id, aliases in replacements.items():
        icon = themed_agent_icon(agent_id)
        name = themed_agent_name(agent_id)
        alias_group = "|".join(re.escape(a) for a in aliases)
        out = re.sub(rf"^##\s+.*\s+({alias_group})$", f"## {icon} {name}", out, flags=re.MULTILINE)
    return out

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
    <div style="padding:20px 4px 12px">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
            <div style="width:36px;height:36px;background:linear-gradient(135deg,#fde68a,#bfdbfe);
                        border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:18px;
                        box-shadow:0 10px 18px rgba(148,163,184,0.14)">
                {theme_cfg['icon']}
            </div>
            <div>
                <div style="font-size:15px;font-weight:600;color:{theme_cfg.get('text_primary', '#334155')}">A2A Chatbot</div>
                <div style="font-size:10px;font-family:'IBM Plex Mono',monospace;color:{theme_cfg.get('text_muted', '#64748b')};letter-spacing:.5px">
                    A2A · HuggingFace
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Health check
    health = check_health()
    if health:
        dot_color = "#86efac"
        status_txt = f"Online · {health.get('agents', 0)} agents"
    else:
        dot_color = "#fda4af"
        status_txt = "Server offline"

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:6px;font-size:12px;
                color:{theme_cfg.get('text_muted', '#64748b')};padding:0 4px 14px;border-bottom:1px solid {theme_cfg.get('card_border', '#e6edf7')};margin-bottom:14px">
        <span style="width:7px;height:7px;border-radius:50%;background:{dot_color};
                     display:inline-block;box-shadow:0 0 0 3px {dot_color}30"></span>
        {status_txt}
    </div>
    """, unsafe_allow_html=True)

    # Load agents and always provide a fallback list for UI continuity
    fetch_agents()
    sidebar_agents = st.session_state.agents or fallback_agent_cards()

    st.markdown(f"""
    <div style="font-size:10px;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;
                color:{theme_cfg.get('text_muted', '#94a3b8')};padding:0 4px;margin-bottom:8px">
        Registered Agents
    </div>
    """, unsafe_allow_html=True)

    for agent in sidebar_agents:
        cfg = AGENT_CONFIG.get(agent["agent_id"], {
            "icon": "🤖✨", "color": "#94a3b8", "label": agent["name"], "desc": ""
        })
        themed_icon = themed_agent_icon(agent["agent_id"])
        is_active = agent["agent_id"] in st.session_state.active_agents
        active_cls = "active" if is_active else ""
        st.markdown(f"""
        <div class="agent-sidebar-card {active_cls}">
            <div class="agent-card-header">
                <span class="agent-dot" style="background:{cfg['color']}"></span>
                <span class="agent-card-name">{themed_icon} {themed_agent_name(agent['agent_id'])}</span>
            </div>
            <div class="agent-card-model">{agent.get('model','')}</div>
            <div class="agent-card-desc">{cfg.get('desc','')}</div>
        </div>
        """, unsafe_allow_html=True)

    active_now = []
    for agent_id in st.session_state.active_agents:
        agent_cfg = AGENT_CONFIG.get(agent_id, {})
        if agent_cfg:
            active_now.append(f"{themed_agent_icon(agent_id)} {themed_agent_name(agent_id)}")
        else:
            active_now.append(agent_id)

    st.markdown(f"""
    <div style="font-size:10px;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;
                color:{theme_cfg.get('text_muted', '#94a3b8')};padding:8px 4px 6px;margin-top:6px">
        Active Agents
    </div>
    """, unsafe_allow_html=True)
    if active_now:
        st.markdown("\n".join([f"- {name}" for name in active_now]))
    else:
        st.caption("No active agents yet. Send a message to see active names.")

    # Session stats
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:10px;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;
                color:{theme_cfg.get('text_primary', '#30303a')};padding:0 4px;margin-bottom:8px">
        Session Stats
    </div>
    <div class="stat-row">
        <div class="stat-chip">queries <span>{st.session_state.total_queries}</span></div>
        <div class="stat-chip">session <span>{st.session_state.session_id[:8]}…</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown(f"<hr style='border:0;border-top:1px solid {theme_cfg.get('card_border', '#dbe4ef')};margin:10px 0 12px 0'>", unsafe_allow_html=True)
    if st.button("🗑️  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.active_agents = set()
        st.rerun()

# ─────────────────────────────────────────────
# Main — chat title
# ─────────────────────────────────────────────

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    <div style="margin-bottom:6px">
        <span style="font-size:20px;font-weight:600;color:#334155">Multi-Agent Chat</span>
        <span style="font-size:12px;color:#64748b;margin-left:10px;
                     font-family:'IBM Plex Mono',monospace">
            A2A Protocol
        </span>
    </div>
    """, unsafe_allow_html=True)
with col2:
    selected_theme = st.selectbox(
        "Theme",
        options=list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme),
        format_func=lambda key: f"{THEMES[key]['icon']} {THEMES[key]['label']}",
        key="theme",
        label_visibility="collapsed",
    )
    st.markdown(
        f"""
        <div style="text-align:right;font-size:11px;color:#64748b;padding-top:3px;
                    font-family:'IBM Plex Mono',monospace">
            Mode: {THEMES[selected_theme]['label']}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ─────────────────────────────────────────────
# Empty state
# ─────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown(f"""
    <div class="hero">
        <div class="hero-icon">{theme_cfg["hero_icon"]}</div>
        <div class="hero-title">How can I help you today?</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Try asking:**")
    cols = st.columns(2)
    for i, starter in enumerate(STARTERS):
        with cols[i % 2]:
            if st.button(starter, key=f"starter_{i}", use_container_width=True):
                st.session_state["_pending_input"] = starter
                st.rerun()

# ─────────────────────────────────────────────
# Chat history
# ─────────────────────────────────────────────

for msg in st.session_state.messages:
    if msg["role"] == "user":
        safe_user_text = html.escape(msg["content"])
        st.markdown(f'<div class="msg-user">{safe_user_text}</div>', unsafe_allow_html=True)
    else:
        assistant_text = apply_theme_to_agent_headers(msg["content"])
        assistant_text = html.escape(assistant_text).replace("\n", "<br>")
        st.markdown(f'<div class="msg-agent">{assistant_text}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────────

# Consume pending input from starter chips
pending = st.session_state.pop("_pending_input", None)
prompt  = st.chat_input("Ask anything — your agents are ready...") or pending

if prompt:
    # Show user message immediately
    safe_prompt = html.escape(prompt)
    st.markdown(f'<div class="msg-user">{safe_prompt}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.history.append(prompt)
    st.session_state.total_queries += 1

    # Status + streaming
    status_box   = st.empty()
    response_box = st.empty()

    full_text   = ""
    agents_used = []
    intent      = ""
    reasoning   = ""
    error_msg   = None

    try:
        with requests.post(
            f"{API_URL}/tasks/send",
            json={"message": prompt, "session_id": st.session_state.session_id},
            stream=True,
            timeout=180,
        ) as resp:
            buf = ""
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8")
                if not line.startswith("data: "):
                    continue

                try:
                    event = json.loads(line[6:])
                except Exception:
                    continue

                etype = event.get("type")

                if etype == "status":
                    status_box.info(event.get("message", ""))
                    if "reasoning" in event:
                        reasoning = event["reasoning"]
                    if "intent" in event:
                        intent = event["intent"]

                elif etype == "chunk":
                    full_text += event.get("text", "")
                    streaming_text = apply_theme_to_agent_headers(full_text) + "▌"
                    streaming_text = html.escape(streaming_text).replace("\n", "<br>")
                    response_box.markdown(f'<div class="msg-agent">{streaming_text}</div>', unsafe_allow_html=True)

                elif etype == "done":
                    agents_used = event.get("agents_used", [])
                    reasoning   = event.get("reasoning", reasoning)
                    intent      = event.get("intent", intent)
                    st.session_state.active_agents = set(agents_used)
                    break

                elif etype == "error":
                    error_msg = event.get("message", "Unknown error")
                    break

    except requests.exceptions.ConnectionError:
        error_msg = (
            "Cannot connect to the server.\n\n"
            "Make sure the backend is running:\n```\npython server.py\n```"
        )
    except Exception as e:
        error_msg = str(e)

    # Clear status
    status_box.empty()

    # Render final pipeline card + response
    if error_msg:
        response_box.error(f"❌ {error_msg}")
        final_content = f"Error: {error_msg}"
        meta = {}
    else:
        themed_text = apply_theme_to_agent_headers(full_text)
        themed_text_html = html.escape(themed_text).replace("\n", "<br>")
        response_box.markdown(f'<div class="msg-agent">{themed_text_html}</div>', unsafe_allow_html=True)
        final_content = full_text
        meta = {"agents_used": agents_used, "intent": intent, "reasoning": reasoning}

    # Save to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": final_content,
        "meta":    meta,
    })
