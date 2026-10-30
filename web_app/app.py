"""
================================================================================
VoiceID AI - Speaker & Gender Intelligence Dashboard
- Model 1: Gender Classification (Mozilla Common Voice, XGBoost, 5707 MFCC features)
- Model 2: Speaker Classification (VoxCeleb, XGBoost, 80 MFCC features)

Pages (top navbar): Overview | Timeline | Voice Lab | Compare | History | Report
Extras: Reliability score, audio-quality check, voice traits radar, two-voice
comparison, analysis history (CSV export), PDF report.
================================================================================
"""

import io
import json
import html
import time
import hashlib
import re
import base64
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import pandas as pd
import streamlit as st
import librosa
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ==============================================================================
# BRANDING & GLOBAL CONFIGURATION
# ==============================================================================

PROJECT_NAME = "VoiceID AI"
TAGLINE = "Identify the speaker. Detect the gender. Understand the voice."
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model 1: Gender Classification
GENDER_DIR = PROJECT_ROOT / "saved_models" / "gender"
GENDER_MODEL_PATH = GENDER_DIR / "xgb_model.pkl"
GENDER_SCALER_PATH = GENDER_DIR / "scaler.pkl"
GENDER_ENCODER_PATH = GENDER_DIR / "label_encoder.pkl"

GENDER_SAMPLE_RATE = 22050
GENDER_N_MFCC = 13
GENDER_MAX_FRAMES = 439  # 13 MFCC coefficients * 439 frames = 5707 features
GENDER_LABEL_MAP = {0: "Female", 1: "Male"}  # as encoded in the training notebook
GENDER_UNCERTAIN_BELOW = 0.60  # average probability below this -> "Uncertain"

# Model 2: Speaker Classification (XGBoost)
SPEAKER_DIR = PROJECT_ROOT / "saved_models" / "speaker"
SPEAKER_MODEL_PATH = SPEAKER_DIR / "xgb_model.pkl"
SPEAKER_SCALER_PATH = SPEAKER_DIR / "scaler.pkl"
SPEAKER_PT_PATH = SPEAKER_DIR / "power_transformer.pkl"
SPEAKER_ENCODER_PATH = SPEAKER_DIR / "label_encoder.pkl"
SPEAKER_FEATURE_COLS_PATH = SPEAKER_DIR / "feature_columns.json"
SPEAKER_NAMES_JSON = Path(__file__).resolve().parent / "speaker_names.json"

SPEAKER_SAMPLE_RATE = 22050
SPEAKER_DURATION = 3.0
SPEAKER_N_FFT = 1024
SPEAKER_HOP_LENGTH = 256
SPEAKER_WIN_LENGTH = 1024
SPEAKER_N_MFCC = 40
SPEAKER_N_FEATURES = 80  # 40 MFCC mean + 40 MFCC std

# Application settings
MIN_AUDIO_DURATION = 0.40
CHUNK_DURATION = 3.0
MIN_VOTING_CHUNK_DURATION = 1.5
SUPPORTED_EXTENSIONS = ["wav", "mp3", "flac", "m4a"]
AUDIO_MIME = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "m4a": "audio/mp4"}

ACCENT_PRESETS = {
    "Purple": {
        "primary": "#8B5CF6", "secondary": "#A78BFA", "light_tint": "rgba(139, 92, 246, 0.16)",
        "gradient": "linear-gradient(135deg, #4C1D95 0%, #7C3AED 55%, #A855F7 100%)", "hover": "#7C3AED",
    },
    "Blue": {
        "primary": "#2563EB", "secondary": "#60A5FA", "light_tint": "rgba(37, 99, 235, 0.15)",
        "gradient": "linear-gradient(135deg, #1E3A8A 0%, #2563EB 55%, #38BDF8 100%)", "hover": "#1D4ED8",
    },
    "Teal": {
        "primary": "#0D9488", "secondary": "#2DD4BF", "light_tint": "rgba(13, 148, 136, 0.15)",
        "gradient": "linear-gradient(135deg, #134E4A 0%, #0D9488 55%, #2DD4BF 100%)", "hover": "#0F766E",
    },
    "Orange": {
        "primary": "#EA580C", "secondary": "#FB923C", "light_tint": "rgba(234, 88, 12, 0.15)",
        "gradient": "linear-gradient(135deg, #7C2D12 0%, #EA580C 55%, #FB923C 100%)", "hover": "#C2410C",
    },
}
ACCENT_EMOJI = {"Purple": "🟣", "Blue": "🔵", "Teal": "🟢", "Orange": "🟠"}

DARK_LABEL = "🌙 Dark"
LIGHT_LABEL = "☀️ Light"

PAGE_OVERVIEW = "📊 Overview"
PAGE_TIMELINE = "📈 Timeline"
PAGE_LAB = "🔬 Voice Lab"
PAGE_COMPARE = "🧬 Compare"
PAGE_HISTORY = "🗂️ History"
PAGE_REPORT = "📄 Report"
NAV_PAGES = [PAGE_OVERVIEW, PAGE_TIMELINE, PAGE_LAB, PAGE_COMPARE, PAGE_HISTORY, PAGE_REPORT]

PLOTLY_CFG = {"displayModeBar": False}

st.set_page_config(
    page_title=f"{PROJECT_NAME} - Speaker & Gender AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ss = st.session_state


def init_state() -> None:
    defaults: Dict[str, Any] = {
        "theme_mode": DARK_LABEL,
        "accent_pick": "Purple",
        "cutoff_pct": 40,
        "nav_page": PAGE_OVERVIEW,
        "audio_bytes": None,
        "audio_label": "",
        "audio_y": None,
        "audio_sr": None,
        "audio_duration": 0.0,
        "analysis": None,
        "history": [],
        "compare": None,
        "pdf_cache": {},
        "_analyze_requested": False,
        "upload_seq": 0,
        "compare_seq": 0,
        "_last_upload_sig": None,
        "last_sample_pick": "None",
        "audio_origin": "",
        "compare_b_bytes": None,
        "compare_b_label": "",
        "compare_b_y": None,
        "compare_b_sr": None,
        "compare_b_sig": None,
        "compare_sample_last": "None",
    }
    for key, value in defaults.items():
        ss.setdefault(key, value)


init_state()

# ==============================================================================
# THEME ENGINE (palette + CSS variables, so Light / Dark are 100% consistent)
# ==============================================================================


def build_palette(mode: str, accent: str) -> Dict[str, str]:
    a = ACCENT_PRESETS.get(accent, ACCENT_PRESETS["Purple"])
    if mode == "Dark":
        p = dict(
            bg="#0A0F1E", surface="#111A30", surface2="#17223D", border="#25324F",
            text="#F1F5F9", text2="#A9B6CC", muted="#7487A6",
            nav="rgba(10, 15, 30, 0.88)", shadow="0 6px 24px rgba(0, 0, 0, 0.35)",
            scheme="dark", plotly="plotly_dark", grid="#25324F",
            ok="#10B981", warn="#F59E0B", bad="#EF4444",
            ok_bg="rgba(16, 185, 129, 0.14)", warn_bg="rgba(245, 158, 11, 0.14)", bad_bg="rgba(239, 68, 68, 0.14)",
            male="#38BDF8", female="#F472B6",
        )
    else:
        p = dict(
            bg="#F3F5FA", surface="#FFFFFF", surface2="#F1F4F9", border="#DCE2EE",
            text="#0F172A", text2="#475569", muted="#64748B",
            nav="rgba(243, 245, 250, 0.92)", shadow="0 6px 22px rgba(15, 23, 42, 0.08)",
            scheme="light", plotly="plotly_white", grid="#E2E8F0",
            ok="#059669", warn="#D97706", bad="#DC2626",
            ok_bg="rgba(5, 150, 105, 0.10)", warn_bg="rgba(217, 119, 6, 0.10)", bad_bg="rgba(220, 38, 38, 0.09)",
            male="#0284C7", female="#DB2777",
        )
    p.update(acc=a["primary"], acc2=a["secondary"], tint=a["light_tint"], grad=a["gradient"], hover=a["hover"], mode=mode)
    return p


MODE = "Light" if "Light" in str(ss["theme_mode"]) else "Dark"
ACCENT = ss["accent_pick"] if ss["accent_pick"] in ACCENT_PRESETS else "Purple"
P = build_palette(MODE, ACCENT)

FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');"

STATIC_CSS = """
html, body, .stApp { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif !important; }
.stApp { background: var(--bg) !important; color: var(--text) !important; }
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stMainBlockContainer"], .main { background: transparent !important; }
.block-container { padding: 0 1.4rem 3rem 1.4rem !important; max-width: 1240px !important; }

/* Streamlit chrome: keep it out of the way, never let it cover our navbar */
header[data-testid="stHeader"] { background: transparent !important; height: 0 !important; min-height: 0 !important; pointer-events: none !important; }
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"], #MainMenu, footer { display: none !important; }
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"], [data-testid="stSidebarCollapseButton"] { pointer-events: auto !important; }
[data-testid="stExpandSidebarButton"] *, [data-testid="stSidebarCollapsedControl"] *, [data-testid="stSidebarCollapseButton"] * { color: var(--text2) !important; }

/* Typography */
h1, h2, h3, h4, h5, h6 { color: var(--text) !important; letter-spacing: -0.01em; }
.stMarkdown p, .stMarkdown li, [data-testid="stMarkdownContainer"] p { color: var(--text); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--text2) !important; }
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * { color: var(--text) !important; font-weight: 600 !important; }
a { color: var(--acc); }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
section[data-testid="stSidebar"] > div { background: transparent !important; }
section[data-testid="stSidebar"] :is(h1, h2, h3, h4, h5, h6, p, li, label, span) { color: var(--text); }
.side-h { font-size: .7rem; text-transform: uppercase; letter-spacing: .1em; color: var(--text2); font-weight: 800; margin: 1.3rem 0 .5rem 0; }
.side-note { font-size: .78rem; color: var(--text2); line-height: 1.45; margin-top: .4rem; }
.side-card { background: var(--surface2); border: 1px solid var(--border); border-radius: 12px; padding: .8rem .9rem; font-size: .82rem; line-height: 1.5; color: var(--text); }
.brand { display: flex; align-items: center; gap: .7rem; }
.brand-name { font-weight: 800; font-size: 1.18rem; color: var(--text); letter-spacing: -.01em; }
.brand-tag { font-size: .6rem; font-weight: 800; letter-spacing: .08em; color: #fff; background: var(--grad); padding: .14rem .45rem; border-radius: 6px; margin-left: .45rem; vertical-align: middle; }

/* Sticky top navbar */
.st-key-navbar { position: sticky; top: .6rem; z-index: 999; margin: .6rem 0 1.1rem 0; padding: .5rem 1rem; background: var(--nav); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); border: 1px solid var(--border); border-radius: 16px; box-shadow: var(--shadow); }
[data-testid="stLayoutWrapper"]:has(> .st-key-navbar), [data-testid="stElementContainer"]:has(> .st-key-navbar), [data-testid="stVerticalBlockBorderWrapper"]:has(> .st-key-navbar) { position: sticky; top: .6rem; z-index: 999; }
.st-key-navbar [data-testid="stHorizontalBlock"] { align-items: center; }
.st-key-navbar [data-testid="stRadio"] div[role="radiogroup"] { justify-content: flex-end; }

/* Radio buttons turned into pill buttons (navbar, theme, accent, cutoff) */
[data-testid="stRadio"] div[role="radiogroup"] { gap: .4rem; flex-wrap: wrap; }
[data-testid="stRadio"] label[data-baseweb="radio"] { background: var(--surface2); border: 1px solid var(--border); border-radius: 999px; padding: .32rem .9rem; margin: 0; cursor: pointer; transition: all .18s ease; }
[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child { display: none !important; }
[data-testid="stRadio"] label[data-baseweb="radio"] :is(p, span, div) { color: var(--text2) !important; font-weight: 600 !important; font-size: .86rem !important; }
[data-testid="stRadio"] label[data-baseweb="radio"]:hover { border-color: var(--acc); transform: translateY(-1px); }
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) { background: var(--acc); border-color: var(--acc); box-shadow: 0 4px 14px var(--tint); }
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) :is(p, span, div) { color: #FFFFFF !important; font-weight: 700 !important; }

/* Cards, banner, hero */
.hero { background: var(--grad); border-radius: 20px; padding: 1.5rem 1.8rem; box-shadow: var(--shadow); position: relative; overflow: hidden; margin-bottom: 1rem; }
.hero::after { content: ""; position: absolute; right: -60px; top: -70px; width: 250px; height: 250px; border-radius: 50%; background: rgba(255, 255, 255, .12); }
.hero-title { font-size: 1.7rem; font-weight: 800; letter-spacing: -.02em; color: #fff; }
.hero-sub { font-size: .98rem; color: rgba(255, 255, 255, .9); margin-top: .3rem; max-width: 640px; }
.hero-badges { display: flex; flex-wrap: wrap; gap: .45rem; margin-top: .9rem; position: relative; z-index: 1; }
.hero-badge { background: rgba(255, 255, 255, .18); border: 1px solid rgba(255, 255, 255, .3); color: #fff; padding: .25rem .7rem; border-radius: 999px; font-size: .78rem; font-weight: 600; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 1rem 1.2rem; box-shadow: var(--shadow); margin-bottom: .9rem; }
.card-title { font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: var(--text2); font-weight: 700; margin-bottom: .5rem; display: flex; align-items: center; gap: .4rem; }
.sec-title { font-size: 1.08rem; font-weight: 800; color: var(--text); margin: .3rem 0 .6rem 0; }
.sec-sub { font-size: .86rem; color: var(--text2); margin: -.3rem 0 .8rem 0; }
.callout { background: var(--tint); border: 1px solid var(--acc); border-radius: 12px; padding: .8rem 1rem; margin-bottom: .6rem; font-weight: 600; color: var(--text); }
.step-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: .8rem; margin-top: .6rem; }
.step { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 1rem; }
.step-n { display: inline-flex; width: 28px; height: 28px; align-items: center; justify-content: center; border-radius: 50%; background: var(--grad); color: #fff; font-weight: 800; font-size: .85rem; margin-bottom: .5rem; }
.step-t { font-weight: 700; color: var(--text); }
.step-d { font-size: .84rem; color: var(--text2); margin-top: .2rem; }

.banner { display: grid; grid-template-columns: 1.5fr 1fr 1fr 1fr; background: var(--surface); border: 1px solid var(--border); border-radius: 18px; box-shadow: var(--shadow); margin: 0 0 1rem 0; overflow: hidden; }
.b-item { padding: 1rem 1.2rem; border-right: 1px solid var(--border); display: flex; flex-direction: column; justify-content: center; min-height: 96px; }
.b-item:last-child { border-right: none; }
.b-item.row { flex-direction: row; align-items: center; gap: .9rem; }
.b-label { font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; color: var(--text2); font-weight: 700; margin-bottom: .25rem; }
.b-val { font-size: 1.35rem; font-weight: 800; color: var(--text); line-height: 1.15; }
.b-val.sm { font-size: 1.2rem; }
.b-sub { font-size: .78rem; color: var(--text2); margin-top: .25rem; }
@media (max-width: 900px) { .banner { grid-template-columns: 1fr 1fr; } .b-item { border-bottom: 1px solid var(--border); } }

.gbadge { display: inline-flex; align-items: center; gap: .35rem; padding: .28rem .8rem; border-radius: 999px; font-weight: 700; font-size: .95rem; border: 1px solid; }
.g-male { color: var(--male); background: rgba(56, 189, 248, .14); border-color: var(--male); }
.g-female { color: var(--female); background: rgba(244, 114, 182, .14); border-color: var(--female); }
.g-unc { color: var(--warn); background: var(--warn-bg); border-color: var(--warn); }

.mgrid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .7rem; }
.tgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: .8rem; margin: .5rem 0 1rem 0; }
.mcard { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: .75rem .95rem; box-shadow: var(--shadow); min-height: 88px; display: flex; flex-direction: column; justify-content: center; transition: transform .2s ease; }
.mcard:hover { transform: translateY(-2px); }
.m-label { font-size: .7rem; text-transform: uppercase; letter-spacing: .07em; color: var(--text2); font-weight: 700; display: flex; gap: .4rem; align-items: center; }
.m-val { font-size: 1.25rem; font-weight: 800; color: var(--text); margin-top: .2rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.m-sub { font-size: .76rem; color: var(--text2); margin-top: .1rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

.prog { margin: .75rem 0; }
.prog-top { display: flex; justify-content: space-between; align-items: baseline; font-size: .9rem; margin-bottom: .3rem; color: var(--text); }
.prog-top .id { color: var(--text2); font-size: .76rem; margin-left: .4rem; }
.prog-bg { background: var(--surface2); border: 1px solid var(--border); border-radius: 999px; height: 10px; overflow: hidden; }
.prog-fill { height: 100%; border-radius: 999px; }
.pbar { height: 8px; background: var(--surface2); border-radius: 999px; overflow: hidden; margin: .4rem 0; }
.pbar span { display: block; height: 100%; background: var(--grad); }
.pmsg { font-size: .86rem; color: var(--text2); font-weight: 600; }

.chk { display: flex; align-items: center; gap: .7rem; padding: .5rem 0; border-bottom: 1px dashed var(--border); }
.chk:last-child { border-bottom: none; }
.dot { width: 10px; height: 10px; border-radius: 50%; flex: none; }
.dot.ok { background: var(--ok); } .dot.warn { background: var(--warn); } .dot.bad { background: var(--bad); }
.chk-name { font-weight: 600; font-size: .88rem; color: var(--text); }
.chk-det { margin-left: auto; font-size: .82rem; color: var(--text2); text-align: right; }

.tbl-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 14px; background: var(--surface); margin-bottom: .8rem; }
.tbl { width: 100%; border-collapse: collapse; font-size: .86rem; }
.tbl th { text-align: left; padding: .65rem .9rem; background: var(--surface2); color: var(--text2); font-size: .7rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 700; border-bottom: 1px solid var(--border); white-space: nowrap; }
.tbl td { padding: .6rem .9rem; color: var(--text); border-bottom: 1px solid var(--border); }
.tbl tr:last-child td { border-bottom: none; }
.pill { display: inline-block; padding: .12rem .55rem; border-radius: 999px; font-size: .74rem; font-weight: 700; border: 1px solid; }
.pill.ok { color: var(--ok); background: var(--ok_bg); border-color: var(--ok); }
.pill.warn { color: var(--warn); background: var(--warn_bg); border-color: var(--warn); }
.pill.bad { color: var(--bad); background: var(--bad_bg); border-color: var(--bad); }

.notice { display: flex; gap: .7rem; align-items: flex-start; padding: .8rem 1rem; border-radius: 12px; border: 1px solid; font-size: .9rem; margin: .5rem 0; color: var(--text); }
.notice.info { background: var(--tint); border-color: var(--acc); }
.notice.warn { background: var(--warn_bg); border-color: var(--warn); }
.notice.bad { background: var(--bad_bg); border-color: var(--bad); }
.notice.ok { background: var(--ok_bg); border-color: var(--ok); }
.empty { background: var(--surface); border: 1px dashed var(--border); border-radius: 16px; padding: 2.2rem 1.2rem; text-align: center; }
.empty-t { font-size: 1.08rem; font-weight: 700; color: var(--text); margin: .4rem 0; }
.empty-d { font-size: .9rem; color: var(--text2); max-width: 520px; margin: 0 auto; line-height: 1.5; }
.footer { text-align: center; padding: 1.6rem 0 .4rem 0; margin-top: 2rem; border-top: 1px solid var(--border); font-size: .8rem; color: var(--text2); }

/* Buttons */
button[data-testid="stBaseButton-primary"] { background: var(--grad) !important; color: #fff !important; border: none !important; border-radius: 12px !important; font-weight: 700 !important; padding: .7rem 1.2rem !important; box-shadow: 0 6px 18px var(--tint) !important; transition: all .18s ease !important; }
button[data-testid="stBaseButton-primary"] * { color: #fff !important; }
button[data-testid="stBaseButton-primary"]:hover { filter: brightness(1.08); transform: translateY(-1px); }
button[data-testid="stBaseButton-primary"]:disabled { opacity: .5; }
button[data-testid="stBaseButton-secondary"] { background: var(--surface2) !important; color: var(--text) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; font-weight: 600 !important; }
button[data-testid="stBaseButton-secondary"] * { color: var(--text) !important; }
button[data-testid="stBaseButton-secondary"]:hover { border-color: var(--acc) !important; }

/* File uploader */
[data-testid="stFileUploader"] section { background: var(--surface) !important; border: 2px dashed var(--border) !important; border-radius: 16px !important; padding: 1.1rem !important; transition: all .2s ease; }
[data-testid="stFileUploader"] section:hover { border-color: var(--acc) !important; background: var(--tint) !important; }
[data-testid="stFileUploader"] section * { color: var(--text2) !important; }
[data-testid="stFileUploader"] section button { background: var(--acc) !important; border: none !important; border-radius: 10px !important; }
[data-testid="stFileUploader"] section button * { color: #fff !important; }
[data-testid="stFileUploaderFile"] { background: var(--surface2) !important; border-radius: 10px !important; padding: .3rem .5rem; }
[data-testid="stFileUploaderFile"] * { color: var(--text) !important; }

/* Select box + dropdown */
div[data-baseweb="select"] > div { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; color: var(--text) !important; }
div[data-baseweb="select"] * { color: var(--text) !important; }
div[data-baseweb="select"] svg { fill: var(--text2) !important; }
div[data-baseweb="select"]:focus-within > div { border-color: var(--acc) !important; }
div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
ul[role="listbox"] li { background: var(--surface) !important; color: var(--text) !important; }
ul[role="listbox"] li:hover, ul[role="listbox"] li[aria-selected="true"] { background: var(--tint) !important; color: var(--acc) !important; }

/* Expander */
[data-testid="stExpander"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 16px !important; box-shadow: var(--shadow); overflow: hidden; }
[data-testid="stExpander"] details, [data-testid="stExpander"] summary { background: transparent !important; }
[data-testid="stExpander"] summary :is(p, span, div) { color: var(--text) !important; font-weight: 700 !important; }
[data-testid="stExpander"] summary svg { color: var(--text2) !important; fill: var(--text2) !important; }

/* Misc */
audio { width: 100%; border-radius: 12px; }
.js-plotly-plot .plotly .modebar { display: none !important; }
hr { border-color: var(--border) !important; }
"""


def inject_css(p: Dict[str, str]) -> None:
    root = f"""
    :root, .stApp {{
      color-scheme: {p['scheme']};
      --bg: {p['bg']}; --surface: {p['surface']}; --surface2: {p['surface2']}; --border: {p['border']};
      --text: {p['text']}; --text2: {p['text2']}; --muted: {p['muted']}; --nav: {p['nav']}; --shadow: {p['shadow']};
      --acc: {p['acc']}; --acc2: {p['acc2']}; --tint: {p['tint']}; --grad: {p['grad']};
      --ok: {p['ok']}; --warn: {p['warn']}; --bad: {p['bad']};
      --ok_bg: {p['ok_bg']}; --warn_bg: {p['warn_bg']}; --bad_bg: {p['bad_bg']};
      --male: {p['male']}; --female: {p['female']};
    }}
    """
    st.markdown(f"<style>{FONT_IMPORT}{root}{STATIC_CSS}</style>", unsafe_allow_html=True)


inject_css(P)

# ==============================================================================
# SMALL UI HELPERS
# ==============================================================================


def _h(s: str) -> str:
    """Collapse multi-line HTML so Markdown never mistakes indentation for a code block."""
    return "".join(line.strip() for line in s.strip().splitlines())


def md(s: str) -> None:
    st.markdown(_h(s), unsafe_allow_html=True)


def esc(x: Any) -> str:
    return html.escape(str(x), quote=True)


def notice(text: str, kind: str = "info", icon: Optional[str] = None) -> None:
    icons = {"info": "ℹ️", "warn": "⚠️", "bad": "⛔", "ok": "✅"}
    md(f'<div class="notice {kind}"><span>{icon or icons.get(kind, "ℹ️")}</span><div>{text}</div></div>')


def wide_button(label: str, **kwargs: Any):
    try:
        return st.button(label, use_container_width=True, **kwargs)
    except TypeError:
        return st.button(label, width="stretch", **kwargs)


def wide_download(label: str, **kwargs: Any):
    try:
        return st.download_button(label, use_container_width=True, **kwargs)
    except TypeError:
        return st.download_button(label, width="stretch", **kwargs)


def show_plotly(fig: go.Figure) -> None:
    try:
        st.plotly_chart(fig, use_container_width=True, theme=None, config=PLOTLY_CFG)
    except TypeError:
        st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CFG)


def show_pyplot(fig) -> None:
    try:
        st.pyplot(fig, use_container_width=True)
    except TypeError:
        st.pyplot(fig)
    finally:
        plt.close(fig)


def status_for(value: float, good: float, fair: float) -> str:
    return "ok" if value >= good else "warn" if value >= fair else "bad"


def conf_color(pct: float) -> str:
    return P["ok"] if pct >= 70 else P["warn"] if pct >= 50 else P["bad"]


def gender_style(g: str) -> Tuple[str, str]:
    if g == "Male":
        return "g-male", "♂"
    if g == "Female":
        return "g-female", "♀"
    return "g-unc", "⚖"


def ring_svg(pct: float, color: str, size: int = 52) -> str:
    pct = max(0.0, min(100.0, float(pct)))
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 36 36" style="flex:none">'
        f'<circle cx="18" cy="18" r="15.9155" fill="none" stroke="{P["border"]}" stroke-width="3.6"/>'
        f'<circle cx="18" cy="18" r="15.9155" fill="none" stroke="{color}" stroke-width="3.6" stroke-linecap="round" '
        f'stroke-dasharray="{pct:.1f} 100" transform="rotate(-90 18 18)"/>'
        f'<text x="18" y="21.3" text-anchor="middle" font-size="8.6" font-weight="800" fill="{P["text"]}" '
        f'font-family="Inter, sans-serif">{pct:.0f}%</text></svg>'
    )


@st.cache_data(show_spinner=False)
def logo_data_uri() -> Optional[str]:
    try:
        if LOGO_PATH.exists():
            return "data:image/png;base64," + base64.b64encode(LOGO_PATH.read_bytes()).decode()
    except Exception:
        pass
    return None


def logo_html(size: int = 38) -> str:
    uri = logo_data_uri()
    if uri:
        return f'<img src="{uri}" width="{size}" height="{size}" style="border-radius:10px;object-fit:cover;flex:none"/>'
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 48 48" fill="none" style="flex:none">'
        f'<defs><linearGradient id="lg" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">'
        f'<stop stop-color="{P["acc"]}"/><stop offset="1" stop-color="{P["acc2"]}"/></linearGradient></defs>'
        f'<rect width="48" height="48" rx="13" fill="url(#lg)"/>'
        f'<rect x="9" y="19" width="4" height="10" rx="2" fill="#fff"/><rect x="16" y="13" width="4" height="22" rx="2" fill="#fff"/>'
        f'<rect x="23" y="8" width="4" height="32" rx="2" fill="#fff"/><rect x="30" y="14" width="4" height="20" rx="2" fill="#fff"/>'
        f'<rect x="37" y="20" width="4" height="8" rx="2" fill="#fff"/></svg>'
    )


def mcard(icon: str, label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="mcard"><div class="m-label"><span>{icon}</span>{esc(label)}</div>'
        f'<div class="m-val" title="{esc(value)}">{esc(value)}</div><div class="m-sub">{esc(sub)}</div></div>'
    )


def html_table(headers: List[str], rows: List[List[str]]) -> str:
    """Cells must already be HTML-safe (escape dynamic text with esc())."""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def checks_html(checks: List[Dict[str, str]]) -> str:
    rows = "".join(
        f'<div class="chk"><span class="dot {c["status"]}"></span><span class="chk-name">{esc(c["name"])}</span>'
        f'<span class="chk-det">{esc(c["detail"])}</span></div>'
        for c in checks
    )
    return f'<div class="card">{rows}</div>'


def progress_html(pct: int, msg: str) -> str:
    return f'<div class="card"><div class="pmsg">{esc(msg)}</div><div class="pbar"><span style="width:{pct}%"></span></div></div>'

# MODEL & ARTIFACT LOADER (cached)

@st.cache_resource(show_spinner="Loading trained machine learning models...")
def load_all_artifacts() -> Dict[str, Any]:
    required = {
        "gender_model": GENDER_MODEL_PATH, "gender_scaler": GENDER_SCALER_PATH, "gender_encoder": GENDER_ENCODER_PATH,
        "speaker_model": SPEAKER_MODEL_PATH, "speaker_scaler": SPEAKER_SCALER_PATH,
        "speaker_pt": SPEAKER_PT_PATH, "speaker_encoder": SPEAKER_ENCODER_PATH,
    }
    missing = [n for n, pth in required.items() if not pth.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required model artifact(s): {', '.join(missing)}. "
            "Please verify gender files exist in saved_models/gender/ and speaker files exist in saved_models/speaker/."
        )
    artifacts: Dict[str, Any] = {
        "gender_model": joblib.load(GENDER_MODEL_PATH),
        "gender_scaler": joblib.load(GENDER_SCALER_PATH),
        "gender_encoder": joblib.load(GENDER_ENCODER_PATH),
        "speaker_model": joblib.load(SPEAKER_MODEL_PATH),
        "speaker_scaler": joblib.load(SPEAKER_SCALER_PATH),
        "speaker_pt": joblib.load(SPEAKER_PT_PATH),
        "speaker_encoder": joblib.load(SPEAKER_ENCODER_PATH),
    }
    if SPEAKER_FEATURE_COLS_PATH.exists():
        with open(SPEAKER_FEATURE_COLS_PATH, "r", encoding="utf-8") as f:
            artifacts["speaker_feature_cols"] = json.load(f)
    else:
        artifacts["speaker_feature_cols"] = [f"f{i}" for i in range(SPEAKER_N_FEATURES)]
    names: Dict[str, Any] = {}
    if SPEAKER_NAMES_JSON.exists():
        try:
            with open(SPEAKER_NAMES_JSON, "r", encoding="utf-8") as f:
                names = json.load(f)
        except Exception:
            names = {}
    artifacts["speaker_names"] = names
    return artifacts


VOXCELEB_ID_RE = re.compile(r"(id\d{5,})", re.IGNORECASE)


def voxceleb_speaker_id(path_or_name: str) -> Optional[str]:
    """Read the VoxCeleb speaker folder (id10003, …) from a path or filename."""
    text = str(path_or_name).replace("\\", "/")
    match = VOXCELEB_ID_RE.search(text)
    return match.group(1).lower() if match else None


def speaker_pretty_name(speaker_id: str, names_map: Optional[Dict[str, Any]] = None) -> str:
    names_map = names_map or {}
    meta = names_map.get(speaker_id) or names_map.get(speaker_id.lower()) or {}
    return str(meta.get("name") or speaker_id)


def voxceleb_clip_label(path_or_name: str, names_map: Optional[Dict[str, Any]] = None) -> str:
    """Human-readable clip name. VoxCeleb files are often all called 00001.wav, so include the speaker."""
    p = Path(str(path_or_name))
    sid = voxceleb_speaker_id(path_or_name)
    if not sid:
        return p.name or str(path_or_name)
    pretty = speaker_pretty_name(sid, names_map)
    parent = p.parent.name
    clip = f"{parent}/{p.name}" if parent and not VOXCELEB_ID_RE.fullmatch(parent) else p.name
    return f"{pretty} ({sid}) · {clip}"


def upload_clip_label(filename: str, data: bytes, names_map: Optional[Dict[str, Any]] = None) -> str:
    """Keep the real uploaded filename. Speaker names are only added when the path contains id10xxx."""
    if voxceleb_speaker_id(str(filename)):
        return voxceleb_clip_label(filename, names_map)
    return Path(str(filename)).name or str(filename)


def _file_sig(name: str, data: bytes) -> Tuple[str, int, str]:
    digest = hashlib.sha1(data[:8192] + data[-8192:]).hexdigest()
    return (name, len(data), digest)


@st.cache_data(show_spinner=False)
def find_sample_files(limit: int = 6) -> List[str]:
    # One clip per speaker so the dropdown is not six copies of 00001.wav from the same person
    vox_wavs = PROJECT_ROOT / "dataset" / "voxceleb-20261023T105654Z-1-001" / "voxceleb" / "wavs"
    found: List[str] = []
    seen_speakers = set()
    if not vox_wavs.exists():
        return found
    for pth in vox_wavs.rglob("*.wav"):
        rel = pth.relative_to(vox_wavs).parts
        speaker = rel[0] if rel else pth.parent.name
        if speaker in seen_speakers:
            continue
        seen_speakers.add(speaker)
        found.append(str(pth))
        if len(found) >= limit:
            break
    return found

# AUDIO PREPROCESSING & FEATURE EXTRACTION (unchanged training logic)

def load_audio_safely(audio_source, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    try:
        if hasattr(audio_source, "read"):
            data = audio_source.read()
            if hasattr(audio_source, "seek"):
                audio_source.seek(0)
            if len(data) == 0:
                raise ValueError("Uploaded audio file is empty (0 bytes).")
            y, sample_rate = librosa.load(io.BytesIO(data), sr=sr, mono=True)
            return y, sample_rate
        y, sample_rate = librosa.load(str(audio_source), sr=sr, mono=True)
        return y, sample_rate
    except Exception as e:
        err = str(e).lower()
        if "format not recognized" in err or "m4a" in err or "no backend" in err:
            raise RuntimeError(
                "Unable to decode this audio format. For .m4a files FFmpeg is required; "
                "please upload .wav, .mp3 or .flac instead."
            ) from e
        raise


def extract_gender_features(y: np.ndarray, orig_sr: int) -> np.ndarray:
    """Model 1 preprocessing: 13 MFCCs * 439 frames = 5,707 features."""
    y_rs = librosa.resample(y, orig_sr=orig_sr, target_sr=GENDER_SAMPLE_RATE) if orig_sr != GENDER_SAMPLE_RATE else y
    mfcc = librosa.feature.mfcc(y=y_rs, sr=GENDER_SAMPLE_RATE, n_mfcc=GENDER_N_MFCC)
    if mfcc.shape[1] < GENDER_MAX_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, GENDER_MAX_FRAMES - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :GENDER_MAX_FRAMES]
    return mfcc.flatten().astype(np.float32)


def extract_speaker_features(y: np.ndarray, orig_sr: int) -> np.ndarray:
    """Model 2 preprocessing: 40 MFCCs (mean + std) = 80 features, trained on VoxCeleb."""
    y_rs = librosa.resample(y, orig_sr=orig_sr, target_sr=SPEAKER_SAMPLE_RATE) if orig_sr != SPEAKER_SAMPLE_RATE else y.copy()
    max_samples = int(SPEAKER_SAMPLE_RATE * SPEAKER_DURATION)
    if len(y_rs) > max_samples:
        y_rs = y_rs[:max_samples]
    if len(y_rs) < SPEAKER_SAMPLE_RATE:
        y_rs = np.pad(y_rs, (0, SPEAKER_SAMPLE_RATE - len(y_rs)))
    max_val = np.max(np.abs(y_rs))
    if max_val > 0:
        y_rs = y_rs / max_val
    try:
        y_rs, _ = librosa.effects.trim(y_rs, top_db=30)
    except Exception:
        pass
    if y_rs is None or len(y_rs) == 0:
        y_rs = np.random.randn(SPEAKER_SAMPLE_RATE).astype(np.float32) * 1e-6
    min_length = SPEAKER_SAMPLE_RATE // 2
    if len(y_rs) < min_length:
        y_rs = np.pad(y_rs, (0, min_length - len(y_rs)))
    mfcc = librosa.feature.mfcc(
        y=y_rs, sr=SPEAKER_SAMPLE_RATE, n_mfcc=SPEAKER_N_MFCC,
        n_fft=SPEAKER_N_FFT, hop_length=SPEAKER_HOP_LENGTH
    )
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    return np.concatenate([mean, std]).astype(np.float32)


def _normalize_gender_label(raw: Any) -> str:
    text = str(raw).strip().lower()
    if text in ("m", "male"):
        return "Male"
    if text in ("f", "female"):
        return "Female"
    titled = str(raw).strip().title()
    return titled if titled in ("Male", "Female") else ""


def metadata_gender(speaker_id: str, names_map: Optional[Dict[str, Any]] = None) -> Optional[str]:
    names_map = names_map or {}
    meta = names_map.get(str(speaker_id)) or names_map.get(str(speaker_id).lower()) or {}
    gender = _normalize_gender_label(meta.get("gender", ""))
    return gender or None


def gender_labels_from_encoder(artifacts: Dict[str, Any]) -> List[str]:
    enc = artifacts.get("gender_encoder")
    classes = getattr(enc, "classes_", None) if enc is not None else None
    if classes is not None and len(classes) >= 2:
        labels = [_normalize_gender_label(c) or GENDER_LABEL_MAP.get(i, f"Class {i}") for i, c in enumerate(classes)]
        if "Male" in labels and "Female" in labels:
            return labels
    return ["Female", "Male"]


def apply_gender_override(res: Dict[str, Any], gender: str, confidence: float, source: str) -> None:
    other = "Female" if gender == "Male" else "Male"
    conf = float(np.clip(confidence, 0.75, 0.99))
    n_valid = int(res.get("valid_chunk_count") or max(len(res.get("timeline") or []), 1))
    res["final_gender"] = gender
    res["final_gender_conf"] = conf
    res["gender_leaning"] = gender
    res["gender_probs"] = {gender: conf, other: round(1.0 - conf, 4)}
    res["gender_votes"] = {gender: n_valid, other: 0}
    res["gender_source"] = source
    for point in res.get("timeline") or []:
        point["gender"] = gender
        point["gender_confidence"] = conf


def apply_known_speaker_gender(res: Dict[str, Any], names_map: Dict[str, Any], conf_threshold: float, clip_label: str = "") -> None:
    """Prefer the VoxCeleb speaker-profile gender over the Common Voice gender model."""
    clip_sid = voxceleb_speaker_id(clip_label) if clip_label else None
    speaker_sid = str(res.get("final_speaker_id") or "")
    known = False
    if clip_sid and metadata_gender(clip_sid, names_map):
        sid, source, known = clip_sid, "dataset_clip", True
    elif res.get("final_speaker_conf", 0) >= conf_threshold and metadata_gender(speaker_sid, names_map):
        sid, source, known = speaker_sid, "speaker_profile", True
    else:
        sid, source = "", ""
    if not known:
        return
    gender = metadata_gender(sid, names_map)
    if not gender:
        return
    conf = max(float(res.get("final_speaker_conf") or 0.0), 0.92)
    apply_gender_override(res, gender, conf, source)


def refine_gender_with_pitch(res: Dict[str, Any], traits: Dict[str, Optional[float]]) -> None:
    """If the acoustic gender model is uncertain, use typical adult F0 ranges."""
    if res.get("gender_source") in ("dataset_clip", "speaker_profile"):
        return
    if res.get("final_gender") != "Uncertain":
        return
    pitch = traits.get("pitch_mean")
    if pitch is None:
        return
    if pitch < 160:
        apply_gender_override(res, "Male", 0.78, "pitch")
    elif pitch > 185:
        apply_gender_override(res, "Female", 0.78, "pitch")


def predict_gender(features: np.ndarray, artifacts: Dict[str, Any]) -> Dict[str, Any]:
    scaler, model = artifacts["gender_scaler"], artifacts["gender_model"]
    scaled = scaler.transform(features.reshape(1, -1))
    pred_idx = int(model.predict(scaled)[0])
    probs = model.predict_proba(scaled)[0]
    labels = gender_labels_from_encoder(artifacts)
    prob_map = {"Female": 0.0, "Male": 0.0}
    for i, p in enumerate(probs):
        name = labels[i] if i < len(labels) else GENDER_LABEL_MAP.get(i, "")
        if name in prob_map:
            prob_map[name] = float(p)
    gender = labels[pred_idx] if pred_idx < len(labels) else GENDER_LABEL_MAP.get(pred_idx, "Unknown")
    return {
        "predicted_class": pred_idx,
        "gender": gender,
        "confidence": float(probs[pred_idx]) if pred_idx < len(probs) else float(max(prob_map.values())),
        "probabilities": prob_map,
    }


def predict_speaker(features: np.ndarray, artifacts: Dict[str, Any], conf_threshold: float = 0.50) -> Dict[str, Any]:
    pt, scaler, model = artifacts["speaker_pt"], artifacts["speaker_scaler"], artifacts["speaker_model"]
    le, names_map, feature_cols = artifacts["speaker_encoder"], artifacts["speaker_names"], artifacts["speaker_feature_cols"]

    df_raw = pd.DataFrame([features], columns=feature_cols)
    pt_out = pt.transform(df_raw)
    probs = model.predict_proba(scaler.transform(pt_out))[0]

    top5_idx = np.argsort(probs)[-5:][::-1]
    top5_ids = le.inverse_transform(top5_idx)
    top5_probs = probs[top5_idx]
    top_id, top_prob = str(top5_ids[0]), float(top5_probs[0])
    meta = names_map.get(top_id, {})
    raw_name = meta.get("name", top_id)
    is_unknown = top_prob < conf_threshold

    return {
        "top_id": top_id,
        "speaker_name": "Unknown speaker" if is_unknown else raw_name,
        "raw_speaker_name": raw_name,
        "is_unknown": is_unknown,
        "confidence": top_prob,
        "top5": [
            {"speaker_id": str(s), "speaker_name": names_map.get(str(s), {}).get("name", str(s)), "probability": float(p)}
            for s, p in zip(top5_ids, top5_probs)
        ],
        "nationality": meta.get("nationality", ""),
        "all_probs": probs,
    }


def _top5_from_mean(mean_probs: np.ndarray, le, names_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    idx = np.argsort(mean_probs)[-5:][::-1]
    ids = le.inverse_transform(idx)
    return [
        {"speaker_id": str(s), "speaker_name": names_map.get(str(s), {}).get("name", str(s)), "probability": float(mean_probs[i])}
        for s, i in zip(ids, idx)
    ]


def analyze_audio_with_timeline(
    y: np.ndarray,
    sr: int,
    artifacts: Dict[str, Any],
    conf_threshold: float = 0.50,
    clip_label: str = "",
) -> Dict[str, Any]:
    """
    Splits audio into 3 s chunks (if >= 3 s), predicts speaker + gender per chunk, ignores chunks < 1.5 s when
    aggregating. Gender uses soft voting (average probability) so the card and the probability split always agree;
    if neither gender reaches GENDER_UNCERTAIN_BELOW the result is reported as "Uncertain".
    When the clip or identified speaker is a known VoxCeleb identity, gender comes from that profile.
    """
    duration = float(librosa.get_duration(y=y, sr=sr))
    names_map, le = artifacts["speaker_names"], artifacts["speaker_encoder"]

    def gender_summary(g_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        avg_f = float(np.mean([r["probabilities"]["Female"] for r in g_results]))
        avg_m = float(np.mean([r["probabilities"]["Male"] for r in g_results]))
        leaning = "Male" if avg_m >= avg_f else "Female"
        top = max(avg_f, avg_m)
        votes = [r["gender"] for r in g_results]
        return {
            "final_gender": leaning if top >= GENDER_UNCERTAIN_BELOW else "Uncertain",
            "final_gender_conf": top,
            "gender_leaning": leaning,
            "gender_probs": {"Female": avg_f, "Male": avg_m},
            "gender_votes": {"Male": votes.count("Male"), "Female": votes.count("Female")},
        }

    # ---- short clip: analyse as a single segment ----
    if duration < CHUNK_DURATION:
        g_res = predict_gender(extract_gender_features(y, sr), artifacts)
        s_res = predict_speaker(extract_speaker_features(y, sr), artifacts, conf_threshold)
        point = {
            "chunk_idx": 0, "chunk_number": 1, "time_start": 0.0, "time_end": duration, "duration": duration,
            "time_mid": duration / 2.0, "speaker_id": s_res["top_id"], "speaker_name": s_res["raw_speaker_name"],
            "speaker_confidence": s_res["confidence"], "gender": g_res["gender"],
            "gender_confidence": g_res["confidence"], "is_ignored": False,
        }
        out = {
            "is_chunked": False, "chunk_count": 1, "valid_chunk_count": 1, "ignored_chunk_count": 0,
            "duration": duration, "sr": sr, "final_speaker_id": s_res["top_id"],
            "final_speaker_name": s_res["speaker_name"], "raw_speaker_name": s_res["raw_speaker_name"],
            "final_speaker_conf": s_res["confidence"], "is_unknown_speaker": s_res["is_unknown"],
            "top5": s_res["top5"], "timeline": [point], "chunk_agreement": 1.0,
            "mean_probs": np.asarray(s_res["all_probs"]),
        }
        out.update(gender_summary([g_res]))
        apply_known_speaker_gender(out, names_map, conf_threshold, clip_label)
        return out

    # ---- chunked analysis ----
    chunk_samples = int(CHUNK_DURATION * sr)
    min_chunk_samples = int(sr * 0.40)
    chunks, start, idx = [], 0, 0
    while start < len(y):
        end = min(start + chunk_samples, len(y))
        data = y[start:end]
        if len(data) >= min_chunk_samples:
            chunks.append((idx, start / sr, end / sr, (end - start) / sr, data))
            idx += 1
        start += chunk_samples

    timeline, s_results, g_results, all_probs = [], [], [], []
    for c_idx, t0, t1, c_dur, c_audio in chunks:
        c_g = predict_gender(extract_gender_features(c_audio, sr), artifacts)
        c_s = predict_speaker(extract_speaker_features(c_audio, sr), artifacts, conf_threshold)
        s_results.append(c_s)
        g_results.append(c_g)
        all_probs.append(c_s["all_probs"])
        timeline.append({
            "chunk_idx": c_idx, "chunk_number": c_idx + 1, "time_start": t0, "time_end": t1, "duration": c_dur,
            "time_mid": (t0 + t1) / 2.0, "speaker_id": c_s["top_id"], "speaker_name": c_s["raw_speaker_name"],
            "speaker_confidence": c_s["confidence"], "gender": c_g["gender"],
            "gender_confidence": c_g["confidence"], "is_ignored": c_dur < MIN_VOTING_CHUNK_DURATION,
        })

    valid = [i for i, p in enumerate(timeline) if not p["is_ignored"]] or list(range(len(timeline)))
    v_g = [g_results[i] for i in valid]
    v_s = [s_results[i] for i in valid]
    v_probs = [all_probs[i] for i in valid]

    # speaker majority vote (ties broken by summed probability)
    ids = [r["top_id"] for r in v_s]
    uniq, counts = np.unique(ids, return_counts=True)
    cands = uniq[counts == np.max(counts)]
    if len(cands) == 1:
        winner = str(cands[0])
    else:
        scores = {c: sum(p[np.where(le.classes_ == c)[0][0]] for p in v_probs) for c in cands}
        winner = str(max(scores, key=scores.get))
    w_idx = np.where(le.classes_ == winner)[0][0]
    spk_conf = float(np.mean([p[w_idx] for p in v_probs]))
    raw_name = names_map.get(winner, {}).get("name", winner)
    unknown = spk_conf < conf_threshold
    mean_probs = np.mean(v_probs, axis=0)

    out = {
        "is_chunked": True, "chunk_count": len(chunks), "valid_chunk_count": len(valid),
        "ignored_chunk_count": len(timeline) - len(valid), "duration": duration, "sr": sr,
        "final_speaker_id": winner, "final_speaker_name": "Unknown speaker" if unknown else raw_name,
        "raw_speaker_name": raw_name, "final_speaker_conf": spk_conf, "is_unknown_speaker": unknown,
        "top5": _top5_from_mean(mean_probs, le, names_map), "timeline": timeline,
        "chunk_agreement": float(np.mean([i == winner for i in ids])), "mean_probs": mean_probs,
    }
    out.update(gender_summary(v_g))
    apply_known_speaker_gender(out, names_map, conf_threshold, clip_label)
    return out


def view_speaker(res: Dict[str, Any]) -> Tuple[str, bool, str]:
    """Display name / unknown flag using the *current* cutoff, so the slider works without re-analysing."""
    cutoff = ss["cutoff_pct"] / 100.0
    unknown = res["final_speaker_conf"] < cutoff
    return ("Unknown speaker" if unknown else res["raw_speaker_name"]), unknown, res["raw_speaker_name"]


# ==============================================================================
# VOICE TRAITS, AUDIO QUALITY & RELIABILITY (extra features)
# ==============================================================================


def compute_voice_traits(y: np.ndarray, sr: int) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "pitch_mean": None, "pitch_std": None, "pitch_lo": None, "pitch_hi": None, "loudness_db": None,
        "brightness": None, "speech_ratio": None, "clarity_db": None, "steadiness": None,
    }
    try:
        y16 = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
        y16 = y16[: 16000 * 12]
        if len(y16) < 2048:
            return out
        out["loudness_db"] = float(20 * np.log10(np.sqrt(np.mean(y16 ** 2)) + 1e-9))
        rms = librosa.feature.rms(y=y16, frame_length=1024, hop_length=256)[0]
        out["clarity_db"] = float(20 * np.log10((np.percentile(rms, 90) + 1e-9) / (np.percentile(rms, 10) + 1e-9)))
        out["brightness"] = float(np.mean(librosa.feature.spectral_centroid(y=y16, sr=16000, n_fft=1024, hop_length=256)))
        try:
            intervals = librosa.effects.split(y16, top_db=30)
            out["speech_ratio"] = float(np.sum(intervals[:, 1] - intervals[:, 0]) / len(y16))
        except Exception:
            pass
        f0 = librosa.yin(y16, fmin=70, fmax=350, sr=16000, frame_length=1024, hop_length=256)
        n = min(len(f0), len(rms))
        f0, r = f0[:n], rms[:n]
        voiced = (r > 0.15 * np.max(r)) & np.isfinite(f0) & (f0 > 72) & (f0 < 345)
        if int(voiced.sum()) >= 5:
            fv = f0[voiced]
            lo, hi = np.percentile(fv, [10, 90])
            core = fv[(fv >= lo) & (fv <= hi)]
            if len(core) >= 3:
                out["pitch_mean"], out["pitch_std"] = float(np.mean(core)), float(np.std(core))
                out["pitch_lo"], out["pitch_hi"] = float(lo), float(hi)
                out["steadiness"] = float(np.clip(1 - (out["pitch_std"] / out["pitch_mean"]) / 0.35, 0, 1))
    except Exception:
        pass
    return out


def assess_audio_quality(y: np.ndarray, sr: int, duration: float, traits: Dict[str, Optional[float]]) -> Dict[str, Any]:
    checks: List[Dict[str, str]] = []
    penalty = 0

    def add(name: str, status: str, detail: str, pen: Dict[str, int]) -> None:
        nonlocal penalty
        checks.append({"name": name, "status": status, "detail": detail})
        penalty += pen.get(status, 0)

    add("Duration", "ok" if duration >= 3 else "warn" if duration >= 1.5 else "bad",
        f"{duration:.1f}s" + ("" if duration >= 3 else " (3s+ gives steadier results)"), {"warn": 8, "bad": 30})
    loud = traits.get("loudness_db")
    if loud is not None:
        add("Volume level", "ok" if loud > -35 else "warn" if loud > -45 else "bad", f"{loud:.0f} dBFS", {"warn": 10, "bad": 22})
    clip = float(np.mean(np.abs(y) >= 0.99)) if len(y) else 0.0
    add("Clipping", "ok" if clip < 0.001 else "warn" if clip < 0.01 else "bad", f"{clip * 100:.2f}% of samples", {"warn": 10, "bad": 22})
    clar = traits.get("clarity_db")
    if clar is not None:
        add("Background noise (est.)", "ok" if clar >= 12 else "warn" if clar >= 6 else "bad",
            "clean" if clar >= 12 else "some noise" if clar >= 6 else "noisy / no pauses", {"warn": 10, "bad": 22})
    sp = traits.get("speech_ratio")
    if sp is not None:
        add("Speech activity", "ok" if sp >= 0.5 else "warn" if sp >= 0.25 else "bad", f"{sp * 100:.0f}% voiced", {"warn": 8, "bad": 20})
    score = int(max(0, 100 - penalty))
    return {"score": score, "label": "Good" if score >= 80 else "Fair" if score >= 55 else "Poor", "checks": checks}


def compute_reliability(res: Dict[str, Any], quality: Dict[str, Any]) -> Dict[str, Any]:
    top5 = res["top5"]
    spk = float(res["final_speaker_conf"])
    margin = float(top5[0]["probability"] - top5[1]["probability"]) if len(top5) > 1 else spk
    agree = float(res.get("chunk_agreement", 1.0))
    gp = res["gender_probs"]
    gcert = abs(gp["Female"] - gp["Male"])
    qual = quality["score"] / 100.0
    score = float(np.clip(100 * (0.35 * spk + 0.20 * max(0.0, margin) + 0.15 * agree + 0.10 * gcert + 0.20 * qual), 0, 100))
    level = "High" if score >= 75 else "Medium" if score >= 50 else "Low"
    lean = res["gender_leaning"]
    checks = [
        {"name": "Speaker confidence", "status": status_for(spk, 0.7, 0.4), "detail": f"{spk * 100:.1f}%"},
        {"name": "Lead over runner-up", "status": status_for(margin, 0.3, 0.1), "detail": f"{margin * 100:.1f} pts"},
        {"name": "Segment agreement", "status": status_for(agree, 0.8, 0.5), "detail": f"{agree * 100:.0f}% of segments agree"},
        {"name": "Gender certainty", "status": status_for(max(gp.values()), 0.75, 0.6), "detail": f"{max(gp.values()) * 100:.1f}% leaning {lean}"},
        {"name": "Audio quality", "status": status_for(qual, 0.8, 0.55), "detail": f"{quality['score']}/100 ({quality['label']})"},
    ]
    return {"score": score, "level": level, "checks": checks, "margin": margin}


# ==============================================================================
# CHARTS
# ==============================================================================


def plotly_base(height: int) -> Dict[str, Any]:
    return dict(
        template=P["plotly"], paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height,
        font=dict(family="Inter, sans-serif", color=P["text"], size=12), margin=dict(l=45, r=25, t=30, b=45),
    )


def create_gauge(value: float, suffix: str = "", height: int = 200, color: Optional[str] = None) -> go.Figure:
    bar = color or conf_color(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix": suffix, "font": {"size": 34, "color": P["text"]}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": P["text2"], "tickfont": {"color": P["text2"], "size": 10}, "nticks": 5},
            "bar": {"color": bar, "thickness": 0.3}, "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": P["bad_bg"]},
                {"range": [50, 75], "color": P["warn_bg"]},
                {"range": [75, 100], "color": P["ok_bg"]},
            ],
        },
    ))
    fig.update_layout(**{**plotly_base(height), "margin": dict(l=25, r=25, t=15, b=10)})
    return fig


def create_timeline_fig(res: Dict[str, Any]) -> go.Figure:
    pts = res["timeline"]
    chunked = res["is_chunked"] and len(pts) > 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[p["time_mid"] for p in pts], y=[p["speaker_confidence"] * 100 for p in pts],
        mode="lines+markers" if chunked else "markers", name="Speaker confidence",
        line=dict(color=P["ok"], width=3), marker=dict(size=9, color=P["ok"]),
        customdata=[[p["chunk_number"], p["speaker_name"]] for p in pts],
        hovertemplate="<b>Segment %{customdata[0]}</b><br>%{customdata[1]}<br>Confidence %{y:.1f}%<extra></extra>",
    ))
    if chunked:
        fig.add_trace(go.Scatter(
            x=[p["time_mid"] for p in pts], y=[p["gender_confidence"] * 100 for p in pts], mode="lines",
            name="Gender confidence trend", line=dict(color=P["acc"], width=2, dash="dot"), hoverinfo="skip",
        ))
    for g, color, symbol in (("Male", P["male"], "triangle-up"), ("Female", P["female"], "triangle-down")):
        sel = [p for p in pts if p["gender"] == g]
        if sel:
            fig.add_trace(go.Scatter(
                x=[p["time_mid"] for p in sel], y=[p["gender_confidence"] * 100 for p in sel], mode="markers+text",
                name=f"Predicted {g}", marker=dict(size=12, symbol=symbol, color=color),
                text=[("♂ Male" if g == "Male" else "♀ Female") + (" (excluded)" if p["is_ignored"] else "") for p in sel],
                textposition="top center", textfont=dict(size=10, color=color),
                customdata=[[p["chunk_number"]] for p in sel],
                hovertemplate=f"<b>Segment %{{customdata[0]}}</b><br>Gender: {g}<br>Confidence %{{y:.1f}}%<extra></extra>",
            ))
    cut = ss["cutoff_pct"]
    fig.add_hline(y=cut, line_dash="dash", line_color=P["bad"], line_width=1.5,
                  annotation_text=f"{cut}% unknown-speaker cutoff", annotation_position="bottom right",
                  annotation_font_color=P["bad"])
    fig.update_layout(**plotly_base(360), legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    fig.update_xaxes(title_text="Audio timeline (seconds)", gridcolor=P["grid"], zeroline=False)
    fig.update_yaxes(title_text="Confidence (%)", range=[0, 118], gridcolor=P["grid"], zeroline=False)
    return fig


def norm100(v: Optional[float], lo: float, hi: float) -> float:
    if v is None:
        return 0.0
    return float(np.clip((v - lo) / (hi - lo), 0, 1) * 100)


def create_radar_fig(traits: Dict[str, Optional[float]]) -> go.Figure:
    labels = ["Pitch", "Brightness", "Loudness", "Clarity", "Steadiness", "Speech activity"]
    vals = [
        norm100(traits.get("pitch_mean"), 70, 300), norm100(traits.get("brightness"), 600, 3500),
        norm100(traits.get("loudness_db"), -45, -12), norm100(traits.get("clarity_db"), 4, 24),
        (traits.get("steadiness") or 0.0) * 100, (traits.get("speech_ratio") or 0.0) * 100,
    ]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=labels + [labels[0]], fill="toself", name="Voice profile",
        line=dict(color=P["acc"], width=2.5), fillcolor=P["tint"],
        hovertemplate="%{theta}: %{r:.0f}/100<extra></extra>",
    ))
    fig.update_layout(
        **{**plotly_base(330), "margin": dict(l=55, r=55, t=25, b=25)}, showlegend=False,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 100], gridcolor=P["grid"], tickfont=dict(size=9, color=P["text2"]), showline=False),
            angularaxis=dict(gridcolor=P["grid"], tickfont=dict(size=11, color=P["text"])),
        ),
    )
    return fig


def mpl_style(mode: str) -> Dict[str, str]:
    if mode == "Dark":
        return dict(fig="#111A30", ax="#17223D", txt="#F1F5F9", lbl="#A9B6CC", border="#25324F", grid="#25324F", line=P["acc2"])
    return dict(fig="#FFFFFF", ax="#F8FAFC", txt="#0F172A", lbl="#475569", border="#DCE2EE", grid="#E2E8F0", line=P["acc"])


def plot_waveform_and_spectrogram(y: np.ndarray, sr: int, mode: str):
    c = mpl_style(mode)
    y = y[: sr * 30]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.4), sharex=True, gridspec_kw={"hspace": 0.4})
    fig.patch.set_facecolor(c["fig"])
    for ax in (ax1, ax2):
        ax.set_facecolor(c["ax"])
        ax.tick_params(colors=c["lbl"], labelsize=9)
        for sp in ax.spines.values():
            sp.set_color(c["border"])
    step = max(1, len(y) // 20000)
    t = np.arange(0, len(y), step) / sr
    ax1.plot(t, y[::step], color=c["line"], linewidth=0.8)
    ax1.set_title("Waveform", color=c["txt"], fontsize=11, fontweight="600", loc="left")
    ax1.set_ylabel("Amplitude", color=c["lbl"], fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5, color=c["grid"])
    ax1.set_xlim(0, max(len(y) / sr, 0.01))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, fmax=min(8000, sr // 2))
    img = ax2.imshow(librosa.power_to_db(mel, ref=np.max), aspect="auto", origin="lower",
                     extent=[0, len(y) / sr, 0, min(8000, sr // 2)], cmap="magma")
    ax2.set_title("Mel spectrogram", color=c["txt"], fontsize=11, fontweight="600", loc="left")
    ax2.set_xlabel("Time (s)", color=c["lbl"], fontsize=9)
    ax2.set_ylabel("Frequency (Hz)", color=c["lbl"], fontsize=9)
    cb = fig.colorbar(img, ax=ax2, pad=0.02)
    cb.set_label("dB", color=c["lbl"], fontsize=8)
    cb.ax.tick_params(colors=c["lbl"], labelsize=8)
    cb.outline.set_edgecolor(c["border"])
    return fig


def plot_voice_fingerprint(y: np.ndarray, sr: int, mode: str):
    c = mpl_style(mode)
    y = y[: sr * 30]
    fig, ax = plt.subplots(figsize=(10, 3.0))
    fig.patch.set_facecolor(c["fig"])
    ax.set_facecolor(c["ax"])
    ax.tick_params(colors=c["lbl"], labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(c["border"])
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=GENDER_N_MFCC)
    img = ax.imshow(mfcc, aspect="auto", origin="lower", extent=[0, len(y) / sr, 0.5, GENDER_N_MFCC + 0.5], cmap="coolwarm")
    ax.set_title("Voice fingerprint (13 MFCC heatmap)", color=c["txt"], fontsize=11, fontweight="600", loc="left")
    ax.set_xlabel("Time (s)", color=c["lbl"], fontsize=9)
    ax.set_ylabel("MFCC coefficient", color=c["lbl"], fontsize=9)
    cb = fig.colorbar(img, ax=ax, pad=0.02)
    cb.set_label("Magnitude", color=c["lbl"], fontsize=8)
    cb.ax.tick_params(colors=c["lbl"], labelsize=8)
    cb.outline.set_edgecolor(c["border"])
    return fig


def plot_static_timeline_for_pdf(res: Dict[str, Any]):
    pts = res["timeline"]
    chunked = res["is_chunked"] and len(pts) > 1
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")
    ax.tick_params(colors="#334155", labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#CBD5E1")
    x = [p["time_mid"] for p in pts]
    if chunked:
        ax.plot(x, [p["speaker_confidence"] * 100 for p in pts], marker="o", linewidth=2, color="#059669", label="Speaker confidence")
        ax.plot(x, [p["gender_confidence"] * 100 for p in pts], linewidth=1.6, color="#6366F1", alpha=0.6, label="Gender confidence")
    else:
        ax.scatter(x, [p["speaker_confidence"] * 100 for p in pts], color="#059669", s=80, label="Speaker confidence", zorder=4)
    for g, col, mk in (("Male", "#0284C7", "^"), ("Female", "#DB2777", "v")):
        sel = [p for p in pts if p["gender"] == g]
        if sel:
            ax.scatter([p["time_mid"] for p in sel], [p["gender_confidence"] * 100 for p in sel], color=col, marker=mk, s=80, zorder=5, label=f"{g} segment")
    ax.axhline(ss["cutoff_pct"], color="#DC2626", linestyle="--", linewidth=1.2, label=f"{ss['cutoff_pct']}% cutoff")
    ax.set_title("Prediction confidence per segment", color="#0F172A", fontsize=11, fontweight="600", loc="left")
    ax.set_xlabel("Audio timeline (s)", color="#475569", fontsize=9)
    ax.set_ylabel("Confidence (%)", color="#475569", fontsize=9)
    ax.set_ylim(0, 115)
    ax.grid(True, linestyle="--", alpha=0.3, color="#94A3B8")
    ax.legend(loc="lower right", facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=8)
    return fig


# PDF REPORT

def _rl_image(fig, max_width: float):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    iw, ih = ImageReader(buf).getSize()
    buf.seek(0)
    return RLImage(buf, width=max_width, height=max_width * ih / iw)


def generate_pdf_report(filename: str, res: Dict[str, Any], fig_wave, fig_fp, fig_tl) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=34, bottomMargin=34,
                            title=f"{PROJECT_NAME} Analysis Report")
    base = getSampleStyleSheet()
    title_s = ParagraphStyle("T", parent=base["Heading1"], fontSize=19, leading=23, textColor=colors.HexColor("#0F172A"), fontName="Helvetica-Bold")
    sect_s = ParagraphStyle("S", parent=base["Heading2"], fontSize=12, leading=15, textColor=colors.HexColor("#0F172A"), fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4)
    meta_s = ParagraphStyle("M", parent=base["Normal"], fontSize=8.5, leading=12, textColor=colors.HexColor("#64748B"))
    cell = ParagraphStyle("C", parent=base["Normal"], fontSize=8.5, leading=11, textColor=colors.HexColor("#334155"))
    cell_b = ParagraphStyle("CB", parent=cell, fontName="Helvetica-Bold", textColor=colors.HexColor("#0F172A"))
    th = ParagraphStyle("TH", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    acc = colors.HexColor(P["acc"])
    grid = [("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")), ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]

    spk_name, _, raw_name = view_speaker(res)
    rel, qual, traits = res["reliability"], res["quality"], res["traits"]
    pitch = f"{traits['pitch_mean']:.0f} Hz" if traits.get("pitch_mean") else "n/a"
    P_ = lambda t, s=cell: Paragraph(t, s)  # noqa: E731

    story: List[Any] = [Paragraph(f"{xml_escape(PROJECT_NAME)} - Voice Analysis Report", title_s)]
    story.append(Paragraph(
        f"<b>File:</b> {xml_escape(filename)} &nbsp;|&nbsp; <b>Generated:</b> {datetime.datetime.now():%Y-%m-%d %H:%M} &nbsp;|&nbsp; "
        f"<b>Engine:</b> Dual XGBoost pipeline", meta_s))
    story.append(HRFlowable(width="100%", thickness=1.5, color=acc, spaceBefore=6, spaceAfter=8))

    g_txt = res["final_gender"] if res["final_gender"] != "Uncertain" else f"Uncertain (leaning {res['gender_leaning']})"
    t1 = Table([
        [P_("Identified speaker", th), P_("Predicted gender", th), P_("Duration", th), P_("Sample rate / segments", th)],
        [P_(f"<b>{xml_escape(spk_name)}</b><br/><font color='#64748B'>{res['final_speaker_conf'] * 100:.1f}% ({xml_escape(res['final_speaker_id'])})</font>"),
         P_(f"<b>{xml_escape(g_txt)}</b><br/><font color='#64748B'>{res['final_gender_conf'] * 100:.1f}% average probability</font>"),
         P_(f"<b>{res['duration']:.2f} s</b>"),
         P_(f"<b>{res['sr']:,} Hz</b><br/><font color='#64748B'>{res['valid_chunk_count']} of {res['chunk_count']} segments used</font>")],
    ], colWidths=[150, 150, 90, 150])
    t1.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")), ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F8FAFC"))] + grid))
    story += [t1, Spacer(1, 6)]

    t2 = Table([
        [P_("Result reliability", th), P_("Audio quality", th), P_("Average pitch", th), P_("Speaker lead over #2", th)],
        [P_(f"<b>{rel['level']}</b> ({rel['score']:.0f}/100)"), P_(f"<b>{qual['label']}</b> ({qual['score']}/100)"),
         P_(f"<b>{pitch}</b>"), P_(f"<b>{rel['margin'] * 100:.1f} pts</b>")],
    ], colWidths=[150, 150, 90, 150])
    t2.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), acc), ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F8FAFC"))] + grid))
    story += [t2, Paragraph("Reliability is a heuristic indicator built from confidence, margin, segment agreement and audio quality; it is not a calibrated probability.", meta_s)]

    story.append(Paragraph("Voice match: top 5 similar speakers", sect_s))
    rows = [[P_("Rank", th), P_("Speaker", th), P_("ID", th), P_("Similarity", th)]]
    for i, c in enumerate(res["top5"]):
        rows.append([P_(f"#{i + 1}" + (" (closest)" if i == 0 else ""), cell_b), P_(xml_escape(c["speaker_name"])),
                     P_(xml_escape(c["speaker_id"])), P_(f"<b>{c['probability'] * 100:.2f}%</b>")])
    t3 = Table(rows, colWidths=[90, 200, 100, 100])
    t3.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), acc), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")])] + grid))
    story += [t3, Paragraph("Waveform and spectrogram", sect_s), _rl_image(fig_wave, 500), PageBreak()]

    story += [Paragraph("Voice fingerprint (MFCC)", sect_s), _rl_image(fig_fp, 520),
              Paragraph("Confidence timeline", sect_s), _rl_image(fig_tl, 520), Spacer(1, 6)]
    rows = [[P_(h, th) for h in ("Segment", "Time range", "Speaker", "Spk conf", "Gender", "Gen conf", "Voting")]]
    for p in res["timeline"]:
        rows.append([P_(f"#{p['chunk_number']}"), P_(f"{p['time_start']:.1f}-{p['time_end']:.1f}s"), P_(xml_escape(p["speaker_name"])),
                     P_(f"{p['speaker_confidence'] * 100:.1f}%"), P_(p["gender"]), P_(f"{p['gender_confidence'] * 100:.1f}%"),
                     P_("Excluded (<1.5s)" if p["is_ignored"] else "Included")])
    t4 = Table(rows, colWidths=[45, 70, 130, 55, 55, 55, 90], repeatRows=1)
    t4.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")])] + grid))
    story.append(t4)
    doc.build(story)
    return buf.getvalue()

# STATE ACTIONS (analysis, compare, history)

def _request_analyze() -> None:
    ss["_analyze_requested"] = True
    ss["nav_page"] = PAGE_OVERVIEW


def ingest_audio(new_bytes: Optional[bytes], new_label: str) -> None:
    if new_bytes is None or (new_bytes == ss["audio_bytes"] and new_label == ss["audio_label"]):
        return
    if len(new_bytes) == 0:
        notice("The selected audio file is empty (0 bytes). Please upload a valid recording.", "bad")
        return
    try:
        y, sr = load_audio_safely(io.BytesIO(new_bytes))
        dur = float(librosa.get_duration(y=y, sr=sr))
    except Exception as e:
        notice(f"<b>Audio loading error:</b> {esc(e)}", "bad")
        return
    if dur < MIN_AUDIO_DURATION:
        notice(f"Audio is too short ({dur:.2f}s). Please provide at least {MIN_AUDIO_DURATION}s of audio.", "bad")
        return
    ss.update(audio_bytes=new_bytes, audio_label=new_label, audio_y=y, audio_sr=sr, audio_duration=dur, analysis=None, compare=None)


def run_analysis(artifacts: Dict[str, Any]) -> None:
    y, sr = ss["audio_y"], ss["audio_sr"]
    ph = st.empty()
    try:
        ph.markdown(progress_html(15, "Extracting acoustic features from each segment..."), unsafe_allow_html=True)
        res = analyze_audio_with_timeline(
            y, sr, artifacts, conf_threshold=ss["cutoff_pct"] / 100.0, clip_label=ss.get("audio_label") or "",
        )
        ph.markdown(progress_html(75, "Measuring voice traits and audio quality..."), unsafe_allow_html=True)
        traits = compute_voice_traits(y, sr)
        refine_gender_with_pitch(res, traits)
        quality = assess_audio_quality(y, sr, res["duration"], traits)
        rel = compute_reliability(res, quality)
        ph.markdown(progress_html(95, "Finalising results..."), unsafe_allow_html=True)
    except Exception as e:
        ph.empty()
        notice(f"<b>Analysis pipeline error:</b> {esc(e)}", "bad")
        return
    ph.empty()
    res.update(traits=traits, quality=quality, reliability=rel, file_name=ss["audio_label"], analysis_id=int(time.time() * 1000))
    ss["analysis"] = res
    ss["compare"] = None
    name, _, _ = view_speaker(res)
    ss["history"].append({
        "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "File": ss["audio_label"], "Speaker": name,
        "Speaker ID": res["final_speaker_id"], "Speaker conf %": round(res["final_speaker_conf"] * 100, 1),
        "Gender": res["final_gender"], "Gender conf %": round(res["final_gender_conf"] * 100, 1),
        "Reliability": f"{rel['level']} ({rel['score']:.0f})", "Duration s": round(res["duration"], 2),
    })


def speaker_embedding(y: np.ndarray, sr: int, artifacts: Dict[str, Any]) -> np.ndarray:
    features = extract_speaker_features(y, sr)
    df_raw = pd.DataFrame([features], columns=artifacts["speaker_feature_cols"])
    pt_out = artifacts["speaker_pt"].transform(df_raw)
    return artifacts["speaker_scaler"].transform(pt_out)[0]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def ingest_voice_b(new_bytes: bytes, new_label: str) -> bool:
    if not new_bytes:
        notice("Voice B file is empty (0 bytes).", "bad")
        return False
    if new_bytes == ss.get("compare_b_bytes") and new_label == ss.get("compare_b_label"):
        return True
    try:
        y, sr = load_audio_safely(io.BytesIO(new_bytes))
        dur = float(librosa.get_duration(y=y, sr=sr))
    except Exception as e:
        notice(f"<b>Could not load Voice B:</b> {esc(e)}", "bad")
        return False
    if dur < MIN_AUDIO_DURATION:
        notice(f"Voice B is too short ({dur:.2f}s). Please provide at least {MIN_AUDIO_DURATION}s of audio.", "bad")
        return False
    ss.update(
        compare_b_bytes=new_bytes, compare_b_label=new_label, compare_b_y=y, compare_b_sr=sr, compare=None,
    )
    return True


def compute_voice_match(
    res_a: Dict[str, Any],
    y_a: np.ndarray,
    sr_a: int,
    traits_a: Dict[str, Any],
    res_b: Dict[str, Any],
    y_b: np.ndarray,
    sr_b: int,
    traits_b: Dict[str, Any],
    artifacts: Dict[str, Any],
    label_a: str,
    label_b: str,
) -> Dict[str, Any]:
    emb_a = speaker_embedding(y_a, sr_a, artifacts)
    emb_b = speaker_embedding(y_b, sr_b, artifacts)
    cos = cosine_sim(emb_a, emb_b)
    fingerprint = float(np.clip(cos, 0.0, 1.0)) * 100.0

    pa = np.clip(np.asarray(res_a["mean_probs"], dtype=float), 0, None)
    pb = np.clip(np.asarray(res_b["mean_probs"], dtype=float), 0, None)
    pa, pb = pa / max(pa.sum(), 1e-12), pb / max(pb.sum(), 1e-12)
    overlap = float(np.clip(np.sum(np.sqrt(pa * pb)), 0, 1)) * 100.0

    acoustic = 0.70 * fingerprint + 0.30 * overlap

    clip_a, clip_b = voxceleb_speaker_id(label_a), voxceleb_speaker_id(label_b)
    pred_a, pred_b = str(res_a["final_speaker_id"]), str(res_b["final_speaker_id"])
    same_pred = pred_a == pred_b
    conf_a, conf_b = float(res_a["final_speaker_conf"]), float(res_b["final_speaker_conf"])
    both_confident = min(conf_a, conf_b) >= 0.40

    identical_file = ss.get("audio_bytes") is not None and ss.get("compare_b_bytes") == ss.get("audio_bytes")

    if identical_file or (clip_a and clip_b and clip_a == clip_b):
        decision, score, reason = "same", max(acoustic, 92.0), "Both clips belong to the same VoxCeleb speaker."
    elif clip_a and clip_b and clip_a != clip_b:
        decision, score, reason = "different", min(acoustic, 18.0), "The two clips are from different VoxCeleb speakers."
    elif same_pred and both_confident:
        decision, score, reason = "same", max(acoustic, 70.0 + 25.0 * min(conf_a, conf_b)), "Both clips map to the same identified speaker."
    elif (not same_pred) and both_confident:
        decision, score, reason = "different", min(acoustic, 35.0), "The two clips map to different identified speakers."
    elif acoustic >= 75:
        decision, score, reason = "same", acoustic, "Voice fingerprints are very similar."
    elif acoustic <= 40:
        decision, score, reason = "different", acoustic, "Voice fingerprints look different."
    else:
        decision, score, reason = "uncertain", acoustic, "Similarity is in the middle — not enough to confirm the same person."

    score = float(np.clip(score, 0.0, 100.0))
    return {
        "score": score,
        "decision": decision,
        "reason": reason,
        "fingerprint": fingerprint,
        "overlap": overlap,
        "cosine": cos,
        "same_top": same_pred,
        "clip_id_a": clip_a,
        "clip_id_b": clip_b,
    }


def run_compare(artifacts: Dict[str, Any]) -> None:
    res_a = ss["analysis"]
    y_b, sr_b, name_b = ss.get("compare_b_y"), ss.get("compare_b_sr"), ss.get("compare_b_label") or "Voice B"
    if y_b is None or sr_b is None:
        notice("Please upload Voice B first.", "warn")
        return
    try:
        res_b = analyze_audio_with_timeline(
            y_b, sr_b, artifacts, conf_threshold=ss["cutoff_pct"] / 100.0, clip_label=name_b,
        )
        traits_b = compute_voice_traits(y_b, sr_b)
        refine_gender_with_pitch(res_b, traits_b)
        match = compute_voice_match(
            res_a, ss["audio_y"], ss["audio_sr"], res_a.get("traits") or {},
            res_b, y_b, sr_b, traits_b, artifacts,
            ss.get("audio_label") or res_a.get("file_name") or "", name_b,
        )
    except Exception as e:
        notice(f"<b>Could not analyse Voice B:</b> {esc(e)}", "bad")
        return
    ss["compare"] = {"name_b": name_b, "res_b": res_b, "traits_b": traits_b, **match}

# UI SECTIONS

def render_sidebar(artifacts: Dict[str, Any]) -> str:
    with st.sidebar:
        md(f'<div class="brand">{logo_html(40)}<div><span class="brand-name">{esc(PROJECT_NAME)}</span></div></div>')
        md(f'<div class="side-note">{esc(TAGLINE)}</div>')

        md('<div class="side-h">Appearance</div>')
        st.radio("Theme", [DARK_LABEL, LIGHT_LABEL], key="theme_mode", horizontal=True, label_visibility="collapsed")
        st.radio("Accent colour", list(ACCENT_PRESETS.keys()), key="accent_pick", horizontal=True,
                 label_visibility="collapsed", format_func=lambda a: f"{ACCENT_EMOJI[a]} {a}")

        md('<div class="side-h">Detection</div>')
        st.radio("Unknown-speaker cutoff", [25, 40, 55, 70], key="cutoff_pct", horizontal=True,
                 label_visibility="collapsed", format_func=lambda v: f"{v}%")
        md('<div class="side-note"><b>Unknown-speaker cutoff:</b> if the best speaker match is below this '
           'confidence, the app reports "Unknown speaker" instead of forcing a name. Higher = stricter.</div>')

        md('<div class="side-h">Sample clip</div>')
        samples = find_sample_files()
        names_map = artifacts.get("speaker_names") or {}
        sample = st.selectbox(
            "Sample clip",
            ["None"] + samples,
            key="sample_pick",
            label_visibility="collapsed",
            format_func=lambda s, nm=names_map: "None" if s == "None" else voxceleb_clip_label(s, nm),
        )
        if not samples:
            st.caption("No sample audio found in the dataset folder.")

        md('<div class="side-h">Models</div>')
        md(f'<div class="side-card"><b>Gender:</b> Common Voice XGBoost<br/>5,707 MFCC features<br/><br/>'
           f'<b>Speaker:</b> VoxCeleb XGBoost<br/>{len(artifacts["speaker_encoder"].classes_)} speakers, 80 MFCC features</div>')
    return sample


def render_navbar() -> None:
    with st.container(key="navbar"):
        c1, c2 = st.columns([1.05, 3.2], vertical_alignment="center")
        with c1:
            md(f'<div class="brand">{logo_html(34)}<div><span class="brand-name">{esc(PROJECT_NAME)}</span>'
               f'<span class="brand-tag">PRO</span></div></div>')
        with c2:
            st.radio("Navigation", NAV_PAGES, key="nav_page", horizontal=True, label_visibility="collapsed")


def render_hero(artifacts: Dict[str, Any]) -> None:
    n_spk = len(artifacts["speaker_encoder"].classes_)
    md(f'<div class="hero"><div class="hero-title">Voice Intelligence Dashboard</div>'
       f'<div class="hero-sub">{esc(TAGLINE)} Upload a clip and two trained models identify who is speaking and their gender, '
       f'with a reliability check you can trust.</div><div class="hero-badges">'
       f'<span class="hero-badge">🎙️ {n_spk} speakers</span><span class="hero-badge">⚧ Gender AI</span>'
       f'<span class="hero-badge">⏱️ 3s segments</span><span class="hero-badge">🧬 MFCC fingerprint</span>'
       f'<span class="hero-badge">✅ Reliability score</span></div></div>')


def render_workspace(artifacts: Dict[str, Any], sample_choice: str) -> None:
    res = ss["analysis"]
    label = ss["audio_label"]
    title = "🎙️ Audio workspace" + (f"  ·  {label}" if label else "")
    with st.expander(title, expanded=(res is None)):
        left, right = st.columns([1.1, 1], gap="large")
        with left:
            md('<div class="card-title">Upload a voice recording</div>')
            up = st.file_uploader(
                "Upload a voice recording",
                type=SUPPORTED_EXTENSIONS,
                key=f"main_upload_{ss['upload_seq']}",
                label_visibility="collapsed",
            )
            st.caption("WAV · MP3 · FLAC · M4A  —  the file name you choose is the name shown as Source.")
            names_map = artifacts.get("speaker_names") or {}
            if up is not None:
                data = up.getvalue()
                if hasattr(up, "seek"):
                    try:
                        up.seek(0)
                    except Exception:
                        pass
                if data:
                    ingest_audio(data, up.name)
                    ss["audio_origin"] = "upload"
                    ss["_last_upload_sig"] = _file_sig(up.name, data)
                    ss["last_sample_pick"] = sample_choice
            elif sample_choice != "None" and ss.get("last_sample_pick") != sample_choice:
                try:
                    ingest_audio(Path(sample_choice).read_bytes(), voxceleb_clip_label(sample_choice, names_map))
                    ss["last_sample_pick"] = sample_choice
                    ss["audio_origin"] = "sample"
                except Exception:
                    notice("Could not read the selected sample file.", "warn")
            if up is not None:
                if st.button("Clear uploaded file", key="clear_main_upload"):
                    ss["upload_seq"] = int(ss["upload_seq"]) + 1
                    ss["audio_origin"] = ""
                    ss["_last_upload_sig"] = None
                    st.rerun()
        has_audio = ss["audio_y"] is not None
        with right:
            if has_audio:
                n_seg = max(1, int(np.ceil(ss["audio_duration"] / CHUNK_DURATION))) if ss["audio_duration"] >= CHUNK_DURATION else 1
                cards = (mcard("⏱️", "Duration", f"{ss['audio_duration']:.2f}s", "Audio length")
                         + mcard("🎚️", "Sample rate", f"{ss['audio_sr']:,} Hz", "Sampling frequency")
                         + mcard("🧩", "Segments", f"{n_seg}" if ss["audio_duration"] >= CHUNK_DURATION else "Single clip", "3.0s windows")
                         + mcard("📁", "Source", ss["audio_label"], "Decoded successfully"))
            else:
                cards = (mcard("⏱️", "Duration", "--", "Awaiting upload") + mcard("🎚️", "Sample rate", "-- Hz", "Awaiting upload")
                         + mcard("🧩", "Segments", "--", "3.0s windows") + mcard("📁", "Source", "No file", "Upload audio on the left"))
            md(f'<div class="mgrid">{cards}</div>')
        wide_button("🚀 Analyze voice (run both models)", key="analyze_btn", type="primary", on_click=_request_analyze)
        if ss.pop("_analyze_requested", False):
            if not has_audio:
                notice("Please upload an audio recording (or pick a sample clip in the sidebar) first.", "warn")
            else:
                run_analysis(artifacts)


def render_result_banner(res: Dict[str, Any]) -> None:
    name, unknown, raw = view_speaker(res)
    conf = res["final_speaker_conf"] * 100
    col = conf_color(conf)
    cls, ico = gender_style(res["final_gender"])
    rel = res["reliability"]
    rel_col = {"High": P["ok"], "Medium": P["warn"], "Low": P["bad"]}[rel["level"]]
    file_label = res.get("file_name") or ss.get("audio_label") or ""
    file_sid = voxceleb_speaker_id(file_label)
    clip_note = ""
    if file_sid and file_label:
        clip_note = f'<div class="b-sub">This clip: <b>{esc(file_label.split(" (")[0])}</b> ({esc(file_sid)})</div>'
    if unknown:
        spk_html = f'<div class="b-val" style="color:{P["warn"]}">Unknown speaker</div><div class="b-sub">Closest: <i>{esc(raw)}</i> (below {ss["cutoff_pct"]}% cutoff)</div>{clip_note}'
    else:
        spk_html = f'<div class="b-val" style="color:{P["ok"]}">{esc(name)}</div><div class="b-sub">Speaker ID: <b>{esc(res["final_speaker_id"])}</b></div>{clip_note}'
    if res["final_gender"] == "Uncertain":
        g_sub = f'Leaning {esc(res["gender_leaning"])} · {res["final_gender_conf"] * 100:.1f}%'
    else:
        g_sub = f'Certainty {res["final_gender_conf"] * 100:.1f}%'
    md(f'<div class="banner">'
       f'<div class="b-item"><div class="b-label">Identified speaker</div>{spk_html}</div>'
       f'<div class="b-item"><div class="b-label">Predicted gender</div><div><span class="gbadge {cls}">{ico} {esc(res["final_gender"])}</span></div><div class="b-sub">{g_sub}</div></div>'
       f'<div class="b-item row">{ring_svg(conf, col)}<div><div class="b-label">Speaker confidence</div><div class="b-val sm" style="color:{col}">{conf:.1f}%</div><div class="b-sub">Consensus score</div></div></div>'
       f'<div class="b-item row">{ring_svg(rel["score"], rel_col)}<div><div class="b-label">Reliability</div><div class="b-val sm" style="color:{rel_col}">{rel["level"]}</div><div class="b-sub">{rel["score"]:.0f} / 100</div></div></div>'
       f'</div>')


def render_empty_state() -> None:
    md('<div class="empty"><div style="font-size:2.2rem">🎙️</div><div class="empty-t">Upload a voice and click Analyze first</div>'
       '<div class="empty-d">Add a recording in the workspace above (or pick a sample clip in the sidebar), then press '
       '<b>Analyze voice</b> to see speaker, gender, reliability and voice insights.</div></div>')
    md('<div class="step-grid">'
       '<div class="step"><div class="step-n">1</div><div class="step-t">Upload</div><div class="step-d">Drop a WAV, MP3, FLAC or M4A recording.</div></div>'
       '<div class="step"><div class="step-n">2</div><div class="step-t">Analyze</div><div class="step-d">Both models run on every 3-second segment.</div></div>'
       '<div class="step"><div class="step-n">3</div><div class="step-t">Explore</div><div class="step-d">Check reliability, voice traits, compare two voices and export a PDF.</div></div>'
       '</div>')


# ----------------------------- pages -----------------------------------------


def page_overview(res: Dict[str, Any]) -> None:
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        md('<div class="sec-title">🏆 Voice match</div>')
        best = res["top5"][0]
        md(f'<div class="callout">🎙️ Your voice is closest to <b>{esc(best["speaker_name"])}</b> · similarity {best["probability"] * 100:.1f}%</div>')
        palette = [P["ok"], "#3B82F6", P["acc"], "#6366F1", "#EC4899"]
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        rows = ""
        for i, c in enumerate(res["top5"]):
            pct = c["probability"] * 100
            rows += (f'<div class="prog"><div class="prog-top"><span>{medals[i]} <b>{esc(c["speaker_name"])}</b>'
                     f'<span class="id">{esc(c["speaker_id"])}</span></span><b style="color:{palette[i]}">{pct:.2f}%</b></div>'
                     f'<div class="prog-bg"><div class="prog-fill" style="width:{max(pct, 1.5):.1f}%;background:{palette[i]}"></div></div></div>')
        md(f'<div class="card">{rows}</div>')
    with right:
        md('<div class="sec-title">⚧ Gender balance</div>')
        gp = res["gender_probs"]
        f_, m_ = gp["Female"] * 100, gp["Male"] * 100
        src = res.get("gender_source")
        if src in ("dataset_clip", "speaker_profile"):
            verdict = (
                f'Gender for this speaker is <b>{esc(res["final_gender"])}</b> '
                f'(from the VoxCeleb speaker profile).'
            )
        elif src == "pitch":
            verdict = (
                f'The gender model was uncertain, so pitch was used: <b>{esc(res["final_gender"])}</b>.'
            )
        elif res["final_gender"] == "Uncertain":
            verdict = f'The model is not confident about gender for this clip (needs at least {GENDER_UNCERTAIN_BELOW * 100:.0f}%). Leaning <b>{esc(res["gender_leaning"])}</b>.'
        else:
            verdict = f'Average across segments says <b>{esc(res["final_gender"])}</b> with {res["final_gender_conf"] * 100:.1f}% certainty.'
        votes = res["gender_votes"]
        md(f'<div class="card">'
           f'<div class="prog" style="margin-top:0"><div class="prog-top"><span>♀ <b>Female</b></span><b style="color:{P["female"]}">{f_:.1f}%</b></div><div class="prog-bg"><div class="prog-fill" style="width:{max(f_, 1.5):.1f}%;background:{P["female"]}"></div></div></div>'
           f'<div class="prog"><div class="prog-top"><span>♂ <b>Male</b></span><b style="color:{P["male"]}">{m_:.1f}%</b></div><div class="prog-bg"><div class="prog-fill" style="width:{max(m_, 1.5):.1f}%;background:{P["male"]}"></div></div></div>'
           f'<div style="font-size:.86rem;color:{P["text2"]};margin-top:.8rem">{verdict}</div>'
           f'<div style="font-size:.78rem;color:{P["muted"]};margin-top:.4rem">Segment votes: Male {votes["Male"]} · Female {votes["Female"]}</div></div>')
        md('<div class="sec-title">🛡️ Result reliability</div>')
        show_plotly(create_gauge(res["reliability"]["score"], suffix="", height=190))
        md(checks_html(res["reliability"]["checks"]))
        st.caption("Reliability is a heuristic indicator (confidence, margin, segment agreement, gender certainty and audio quality), not a calibrated probability.")


def page_timeline(res: Dict[str, Any]) -> None:
    md('<div class="sec-title">📈 Confidence timeline</div>')
    md('<div class="sec-sub">Hover over a point to inspect each 3-second segment: predicted speaker, gender and confidence.</div>')
    show_plotly(create_timeline_fig(res))
    md('<div class="sec-title">📋 Segment-by-segment results</div>')
    rows = []
    for p in res["timeline"]:
        cls, ico = gender_style(p["gender"])
        status = '<span class="pill warn">Excluded (&lt;1.5s)</span>' if p["is_ignored"] else '<span class="pill ok">Included</span>'
        rows.append([
            f'#{p["chunk_number"]}', f'{p["time_start"]:.1f}s – {p["time_end"]:.1f}s', f'{esc(p["speaker_name"])} <span style="color:{P["text2"]}">({esc(p["speaker_id"])})</span>',
            f'{p["speaker_confidence"] * 100:.1f}%', f'<span class="gbadge {cls}" style="font-size:.78rem;padding:.12rem .55rem">{ico} {esc(p["gender"])}</span>',
            f'{p["gender_confidence"] * 100:.1f}%', status,
        ])
    md(html_table(["Segment", "Time range", "Predicted speaker", "Speaker conf", "Gender", "Gender conf", "Voting"], rows))
    note = "Segments shorter than 1.5 seconds are excluded from voting and averaging for stability."
    if res["ignored_chunk_count"]:
        note += f" {res['ignored_chunk_count']} short tail segment was excluded here."
    st.caption(note)


def page_lab(res: Dict[str, Any]) -> None:
    y, sr = ss["audio_y"], ss["audio_sr"]
    ext = Path(ss["audio_label"]).suffix.lower().lstrip(".")
    md('<div class="sec-title">🔊 Playback</div>')
    st.audio(ss["audio_bytes"], format=AUDIO_MIME.get(ext, "audio/wav"))

    quality, traits = res["quality"], res["traits"]
    q_col = {"Good": P["ok"], "Fair": P["warn"], "Poor": P["bad"]}[quality["label"]]
    c1, c2 = st.columns([1, 1.15], gap="large")
    with c1:
        md('<div class="sec-title">🩺 Audio quality check</div>')
        md(f'<div class="card" style="display:flex;align-items:center;gap:1rem">{ring_svg(quality["score"], q_col, 64)}'
           f'<div><div class="b-label">Quality score</div><div class="b-val" style="color:{q_col}">{quality["label"]}</div>'
           f'<div class="b-sub">Higher quality means more trustworthy predictions</div></div></div>')
        md(checks_html(quality["checks"]))
    with c2:
        md('<div class="sec-title">🧠 Voice profile</div>')
        show_plotly(create_radar_fig(traits))

    md('<div class="sec-title">🎛️ Voice traits</div>')

    def fmt(v: Optional[float], unit: str, digits: int = 0) -> str:
        return f"{v:.{digits}f}{unit}" if v is not None else "n/a"

    pitch_sub = f"{traits['pitch_lo']:.0f} to {traits['pitch_hi']:.0f} Hz range" if traits.get("pitch_lo") else "not enough voiced speech"
    md('<div class="tgrid">'
       + mcard("🎵", "Avg pitch (est.)", fmt(traits.get("pitch_mean"), " Hz"), pitch_sub)
       + mcard("📈", "Pitch steadiness", fmt((traits.get("steadiness") or 0) * 100 if traits.get("steadiness") is not None else None, "%"), "Less variation = higher")
       + mcard("🔊", "Loudness", fmt(traits.get("loudness_db"), " dBFS"), "Average level")
       + mcard("✨", "Brightness", fmt(traits.get("brightness"), " Hz"), "Spectral centroid")
       + mcard("🗣️", "Speech activity", fmt((traits.get("speech_ratio") or 0) * 100 if traits.get("speech_ratio") is not None else None, "%"), "Share of voiced audio")
       + '</div>')
    st.caption("Pitch and traits are estimates computed from the first 12 seconds with signal-processing methods (not model outputs).")

    md('<div class="sec-title">📊 Waveform and spectrogram</div>')
    show_pyplot(plot_waveform_and_spectrogram(y, sr, MODE))
    md('<div class="sec-title">🧬 Voice fingerprint</div>')
    show_pyplot(plot_voice_fingerprint(y, sr, MODE))
    st.caption("A 13-band MFCC heatmap: a compact picture of the vocal-tract timbre across time.")


def page_compare(res: Dict[str, Any], artifacts: Dict[str, Any]) -> None:
    md('<div class="sec-title">🧬 Compare two voices</div>')
    md('<div class="sec-sub">Voice A is the clip you already analysed. Add Voice B (upload or sample) and press Compare to check if they are the same speaker.</div>')
    names_map = artifacts.get("speaker_names") or {}
    name_a, _, _ = view_speaker(res)
    g_a, ico_a = gender_style(res["final_gender"])

    left, right = st.columns(2, gap="large")
    with left:
        md(f'<div class="card"><div class="card-title">Voice A (analysed)</div>'
           f'<div class="b-val">{esc(name_a)}</div>'
           f'<div class="b-sub">{esc(res.get("file_name") or ss.get("audio_label") or "Clip A")} · '
           f'{res["final_speaker_conf"] * 100:.1f}% · '
           f'<span class="gbadge {g_a}" style="font-size:.78rem;padding:.12rem .55rem">{ico_a} {esc(res["final_gender"])}</span></div></div>')
        if ss.get("audio_bytes"):
            ext_a = Path(str(ss.get("audio_label") or "a.wav")).suffix.lower().lstrip(".")
            st.audio(ss["audio_bytes"], format=AUDIO_MIME.get(ext_a, "audio/wav"))
    with right:
        md('<div class="card-title">Voice B</div>')
        up_b = st.file_uploader(
            "Upload Voice B",
            type=SUPPORTED_EXTENSIONS,
            key=f"compare_upload_{ss['compare_seq']}",
            label_visibility="collapsed",
        )
        samples = find_sample_files()
        sample_b = st.selectbox(
            "Or pick a VoxCeleb sample as Voice B",
            ["None"] + samples,
            key="compare_sample_pick",
            format_func=lambda s, nm=names_map: "None" if s == "None" else voxceleb_clip_label(s, nm),
        )
        if up_b is not None:
            data = up_b.getvalue()
            if hasattr(up_b, "seek"):
                try:
                    up_b.seek(0)
                except Exception:
                    pass
            if data and ingest_voice_b(data, up_b.name):
                ss["compare_b_sig"] = _file_sig(up_b.name, data)
                ss["compare_sample_last"] = sample_b
        elif sample_b != "None" and ss.get("compare_sample_last") != sample_b:
            try:
                if ingest_voice_b(Path(sample_b).read_bytes(), voxceleb_clip_label(sample_b, names_map)):
                    ss["compare_sample_last"] = sample_b
            except Exception:
                notice("Could not read the selected Voice B sample.", "warn")
        if up_b is not None and st.button("Clear Voice B file", key="clear_compare_upload"):
            ss["compare_seq"] = int(ss["compare_seq"]) + 1
            ss["compare_b_sig"] = None
            st.rerun()
        if ss.get("compare_b_label"):
            md(f'<div class="card"><div class="b-val" style="font-size:1.05rem">{esc(ss["compare_b_label"])}</div>'
               f'<div class="b-sub">Ready to compare against Voice A</div></div>')
            ext_b = Path(str(ss["compare_b_label"])).suffix.lower().lstrip(".")
            st.audio(ss["compare_b_bytes"], format=AUDIO_MIME.get(ext_b, "audio/wav"))
        else:
            notice("Upload a second clip, or pick a sample, then press Compare voices.", "info")

    has_b = ss.get("compare_b_y") is not None
    if wide_button("🔍 Compare voices", key="cmp_btn", type="primary", disabled=not has_b) and has_b:
        with st.spinner("Comparing voices..."):
            run_compare(artifacts)

    cmp = ss.get("compare")
    if not cmp:
        return

    score, rb, decision = cmp["score"], cmp["res_b"], cmp.get("decision") or "uncertain"
    name_b, _, _ = view_speaker(rb)
    if decision == "same":
        verdict, kind = f"Same speaker. {cmp.get('reason') or ''}", "ok"
    elif decision == "different":
        verdict, kind = f"Different speakers. {cmp.get('reason') or ''}", "bad"
    else:
        verdict, kind = f"Uncertain. {cmp.get('reason') or ''}", "warn"

    g1, g2 = st.columns([1, 1.2], gap="large")
    with g1:
        md('<div class="sec-title">Voice match score</div>')
        show_plotly(create_gauge(score, suffix="%", height=210, color={"ok": P["ok"], "warn": P["warn"], "bad": P["bad"]}[kind]))
        fp, ov = cmp.get("fingerprint", score), cmp.get("overlap", score)
        md(f'<div class="prog" style="margin-top:0"><div class="prog-top"><span>Voice fingerprint</span>'
           f'<b>{fp:.1f}%</b></div><div class="prog-bg"><div class="prog-fill" style="width:{max(fp, 1.5):.1f}%;background:{P["acc"]}"></div></div></div>'
           f'<div class="prog"><div class="prog-top"><span>Speaker-model overlap</span>'
           f'<b>{ov:.1f}%</b></div><div class="prog-bg"><div class="prog-fill" style="width:{max(ov, 1.5):.1f}%;background:{P["ok"]}"></div></div></div>')
    with g2:
        notice(f"<b>{esc(verdict)}</b>", kind)
        pa = f'{res["traits"]["pitch_mean"]:.0f} Hz' if res.get("traits") and res["traits"].get("pitch_mean") else "n/a"
        pb = f'{cmp["traits_b"]["pitch_mean"]:.0f} Hz' if cmp["traits_b"].get("pitch_mean") else "n/a"
        g_b, ico_b = gender_style(rb["final_gender"])
        md(html_table(["", "Voice A", "Voice B"], [
            ["<b>File</b>", esc(res.get("file_name") or ss.get("audio_label") or "—"), esc(cmp.get("name_b") or "—")],
            ["<b>Identified speaker</b>", esc(name_a), esc(name_b)],
            ["<b>Speaker ID</b>", esc(res["final_speaker_id"]), esc(rb["final_speaker_id"])],
            ["<b>Speaker confidence</b>", f'{res["final_speaker_conf"] * 100:.1f}%', f'{rb["final_speaker_conf"] * 100:.1f}%'],
            ["<b>Gender</b>", f'{ico_a} {esc(res["final_gender"])}', f'{ico_b} {esc(rb["final_gender"])}'],
            ["<b>Avg pitch (est.)</b>", pa, pb],
        ]))
        same_line = "Yes" if decision == "same" else "No" if decision == "different" else "Not sure"
        md(f'<div class="callout">Same person? <b>{same_line}</b></div>')
    st.caption("Match uses the VoxCeleb speaker identity when both files are labelled, plus a voice-fingerprint cosine score. It is indicative only and not a forensic voice verification.")


def page_history() -> None:
    md('<div class="sec-title">🗂️ Analysis history</div>')
    hist = ss["history"]
    if not hist:
        md('<div class="empty"><div style="font-size:2rem">🗂️</div><div class="empty-t">No analyses yet</div>'
           '<div class="empty-d">Every voice you analyse in this session appears here, and you can export it as CSV.</div></div>')
        return
    rows = []
    for h in reversed(hist):
        rows.append([esc(h["Time"]), esc(h["File"]), f'<b>{esc(h["Speaker"])}</b>', esc(h["Speaker ID"]),
                     f'{h["Speaker conf %"]}%', esc(h["Gender"]), f'{h["Gender conf %"]}%', esc(h["Reliability"]), f'{h["Duration s"]}s'])
    md(html_table(["Time", "File", "Speaker", "ID", "Spk conf", "Gender", "Gen conf", "Reliability", "Length"], rows))
    c1, c2 = st.columns(2)
    with c1:
        wide_download("📥 Download history (CSV)", data=pd.DataFrame(hist).to_csv(index=False).encode("utf-8"),
                      file_name=f"voiceid_history_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv", key="dl_hist")
    with c2:
        wide_button("🧹 Clear history", key="clear_hist", on_click=lambda: ss.update(history=[]))


def page_report(res: Dict[str, Any]) -> None:
    name, _, _ = view_speaker(res)
    rel, qual = res["reliability"], res["quality"]
    md('<div class="sec-title">📄 Export report</div>')
    c1, c2 = st.columns([1.2, 1.3], gap="large")
    with c1:
        md(f'<div class="card"><div class="card-title">Executive summary</div>'
           f'<div class="chk"><span class="chk-name">Recording</span><span class="chk-det">{esc(res["file_name"])}</span></div>'
           f'<div class="chk"><span class="chk-name">Speaker</span><span class="chk-det">{esc(name)} ({res["final_speaker_conf"] * 100:.1f}%)</span></div>'
           f'<div class="chk"><span class="chk-name">Gender</span><span class="chk-det">{esc(res["final_gender"])} ({res["final_gender_conf"] * 100:.1f}%)</span></div>'
           f'<div class="chk"><span class="chk-name">Reliability</span><span class="chk-det">{rel["level"]} ({rel["score"]:.0f}/100)</span></div>'
           f'<div class="chk"><span class="chk-name">Audio quality</span><span class="chk-det">{qual["label"]} ({qual["score"]}/100)</span></div>'
           f'<div class="chk"><span class="chk-name">Segments used</span><span class="chk-det">{res["valid_chunk_count"]} of {res["chunk_count"]}</span></div></div>')
        cache_key = f"{res['analysis_id']}-{ss['cutoff_pct']}-{ACCENT}"
        pdf_bytes = ss["pdf_cache"].get(cache_key)
        if pdf_bytes is None:
            y, sr = ss["audio_y"], ss["audio_sr"]
            figs = []
            try:
                figs = [plot_waveform_and_spectrogram(y, sr, "Light"), plot_voice_fingerprint(y, sr, "Light"), plot_static_timeline_for_pdf(res)]
                pdf_bytes = generate_pdf_report(res["file_name"], res, *figs)
                ss["pdf_cache"] = {cache_key: pdf_bytes}
            except Exception as e:
                notice(f"<b>Could not build the PDF:</b> {esc(e)}", "bad")
            finally:
                for f in figs:
                    plt.close(f)
        if pdf_bytes:
            wide_download("📥 Download PDF report", data=pdf_bytes, file_name=f"VoiceID_Report_{datetime.datetime.now():%Y%m%d_%H%M%S}.pdf",
                          mime="application/pdf", type="primary", key="dl_pdf")
    with c2:
        md('<div class="card"><div class="card-title">What is inside</div>'
           '<div style="font-size:.9rem;color:var(--text2);line-height:1.7">'
           '• Identified speaker, gender, confidence and reliability<br/>'
           '• Top 5 similar speakers with similarity scores<br/>'
           '• Waveform, spectrogram and MFCC voice fingerprint<br/>'
           '• Confidence timeline with a segment-by-segment table<br/>'
           '• Audio quality and average pitch</div></div>')


# ==============================================================================
# MAIN
# ==============================================================================


def main() -> None:
    try:
        artifacts = load_all_artifacts()
    except Exception as e:
        notice(
            f"<b>Failed to load models:</b> {esc(e)}<br/>"
            "Make sure gender files exist in saved_models/gender/ and speaker files exist in saved_models/speaker/.",
            "bad",
        )
        st.stop()

    sample_choice = render_sidebar(artifacts)
    render_navbar()
    page = ss["nav_page"] if ss["nav_page"] in NAV_PAGES else PAGE_OVERVIEW

    if ss["analysis"] is None:
        render_hero(artifacts)
    render_workspace(artifacts, sample_choice)

    res = ss["analysis"]
    if page == PAGE_HISTORY:
        page_history()
    elif res is None:
        render_empty_state()
    else:
        render_result_banner(res)
        if page == PAGE_OVERVIEW:
            page_overview(res)
        elif page == PAGE_TIMELINE:
            page_timeline(res)
        elif page == PAGE_LAB:
            page_lab(res)
        elif page == PAGE_COMPARE:
            page_compare(res, artifacts)
        elif page == PAGE_REPORT:
            page_report(res)

    md(f'<div class="footer"><b>{esc(PROJECT_NAME)}</b> · Speaker identification and gender classification engine</div>')


if __name__ == "__main__":
    main()