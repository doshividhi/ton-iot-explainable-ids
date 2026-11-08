"""Helpers shared by every page of the Streamlit demo."""
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
ASSETS = APP_DIR / "assets"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ids_common import RULE_COLUMNS, SPOOF_RULES  # noqa: E402,F401

FAMILY_LABELS = {
    "normal": "Normal",
    "backdoor": "Backdoor",
    "ddos": "DDoS",
    "dos": "DoS",
    "injection": "Injection",
    "mitm": "Man-in-the-middle",
    "password": "Password attack",
    "ransomware": "Ransomware",
    "scanning": "Scanning",
    "xss": "Cross-site scripting",
}
SEVERITY = {"mitm": "High", "ransomware": "High", "backdoor": "High", "injection": "Medium", "password": "Medium",
            "xss": "Medium", "ddos": "Medium", "dos": "Medium", "scanning": "Low", "normal": "—"}
FEATURE_LABELS = {
    "proto": "Protocol", "service": "Service", "conn_state": "Connection state", "duration": "Duration (s)",
    "src_bytes": "Bytes sent", "dst_bytes": "Bytes received", "src_pkts": "Packets sent", "dst_pkts": "Packets received",
    "src_ip_bytes": "IP bytes sent", "dst_ip_bytes": "IP bytes received", "missed_bytes": "Missed bytes",
    "dns_query": "DNS query", "dns_qtype": "DNS query type", "dns_rcode": "DNS response code", "dns_AA": "DNS authoritative flag",
    "dns_RD": "DNS recursion-desired flag", "dns_RA": "DNS recursion-available flag", "dns_rejected": "DNS rejected",
    "dns_qclass": "DNS query class", "ssl_version": "TLS version", "ssl_cipher": "TLS cipher", "ssl_established": "TLS established",
    "ssl_resumed": "TLS resumed", "ssl_subject": "TLS certificate subject", "ssl_issuer": "TLS certificate issuer",
    "http_method": "HTTP method", "http_uri": "HTTP URI", "http_status_code": "HTTP status", "http_user_agent": "HTTP user agent",
    "http_trans_depth": "HTTP depth", "http_version": "HTTP version", "http_request_body_len": "HTTP request size",
    "http_response_body_len": "HTTP response size", "http_orig_mime_types": "HTTP request MIME", "http_resp_mime_types": "HTTP response MIME",
    "weird_name": "Protocol anomaly", "weird_addl": "Anomaly detail", "weird_notice": "Anomaly notice",
    "rule_icmp_redirect": "Spoof rule: ICMP redirect", "rule_icmp_ttl_exceeded": "Spoof rule: TTL exceeded",
    "rule_dns_spoof": "Spoof rule: rogue DNS", "rule_tls_strip": "Spoof rule: TLS stripping",
}


def assets_ready() -> bool:
    return (ASSETS / "manifest.json").exists()


def require_assets():
    if not assets_ready():
        st.error("Demo assets not found. Run this once from the project folder, then refresh:")
        st.code("python app/build_demo_assets.py", language="bash")
        st.stop()


@st.cache_data(show_spinner=False)
def manifest() -> dict:
    return json.loads((ASSETS / "manifest.json").read_text())


@st.cache_data(show_spinner="Loading replay traffic…")
def replay_flows() -> pd.DataFrame:
    return pd.read_pickle(ASSETS / "replay_flows.pkl")


@st.cache_resource(show_spinner="Loading detection models…")
def fold_models() -> dict:
    return {int(p.stem.split("_")[-1]): joblib.load(p) for p in sorted(ASSETS.glob("fold_model_*.joblib"))}


@st.cache_resource(show_spinner="Loading simulator models…")
def simulator_models() -> dict:
    return {"standard": joblib.load(ASSETS / "sim_standard.joblib"), "adversarial": joblib.load(ASSETS / "sim_adversarial.joblib")}


@st.cache_data(show_spinner=False)
def simulator_pool() -> pd.DataFrame:
    return pd.read_pickle(ASSETS / "sim_pool.pkl")


@st.cache_data(show_spinner=False)
def drift_reference() -> dict:
    return json.loads((ASSETS / "drift_reference.json").read_text())


@st.cache_data(show_spinner=False)
def results_csv(relative: str, **kwargs) -> pd.DataFrame | None:
    path = PROJECT_ROOT / relative
    return pd.read_csv(path, **kwargs) if path.exists() else None


@st.cache_resource(show_spinner=False)
def explainer_for(fold: int):
    import shap

    return shap.TreeExplainer(fold_models()[fold]["model"].named_steps["model"])


def explain_flow(row: pd.Series, top_k: int = 6) -> pd.DataFrame:
    """SHAP contribution of each feature to the predicted class for one flow."""
    bundle = fold_models()[int(row["fold"])]
    pipe, cols = bundle["model"], bundle["features"]
    X = pipe.named_steps["pre"].transform(row[cols].to_frame().T)
    values = explainer_for(int(row["fold"])).shap_values(X)
    classes = list(pipe.classes_)
    target = row["predicted"] if row["predicted"] in classes else "normal"
    k = classes.index(target)
    if isinstance(values, list):
        contrib = values[k][0]
    else:
        contrib = values[0, :, k] if values.ndim == 3 else values[0]
    names = [n.split("__", 1)[1] for n in pipe.named_steps["pre"].get_feature_names_out()]
    out = pd.DataFrame({"feature": names, "contribution": contrib})
    out["value"] = [row[n] for n in names]
    out["label"] = out["feature"].map(FEATURE_LABELS).fillna(out["feature"])
    return out.reindex(out["contribution"].abs().sort_values(ascending=False).index).head(top_k)


def rule_hits(row: pd.Series) -> list[dict]:
    return [SPOOF_RULES[c.replace("rule_", "")] | {"column": c} for c in RULE_COLUMNS if int(row[c]) == 1]


def style_page():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1250px;}
        div[data-testid="stMetric"] {background: rgba(42,120,214,0.06); border: 1px solid rgba(42,120,214,0.15);
            border-radius: 10px; padding: 10px 14px;}
        div[data-testid="stMetricLabel"] p {font-size: 0.85rem;}
        .pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.8rem; font-weight:600;}
        .pill-high {background:#fde8e8; color:#9b1c1c;} .pill-medium {background:#fdf3d7; color:#7a4d00;}
        .pill-low {background:#e3f1fb; color:#0c447c;} .pill-ok {background:#e5f5ec; color:#14532d;}
        .card {border:1px solid rgba(120,120,120,0.25); border-radius:12px; padding:14px 16px; margin-bottom:10px;}
        .muted {color: rgba(120,120,120,1); font-size: 0.9rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def pill(text: str, kind: str) -> str:
    return f'<span class="pill pill-{kind}">{text}</span>'


def severity_kind(family: str) -> str:
    return {"High": "high", "Medium": "medium", "Low": "low"}.get(SEVERITY.get(family, "—"), "ok")
