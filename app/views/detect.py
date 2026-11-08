import hashlib
import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from demo_lib import (APP_DIR, FAMILY_LABELS, FEATURE_LABELS, SEVERITY, explain_flow, fold_models, manifest,
                      replay_flows, require_assets, style_page)
from ids_common import RULE_COLUMNS, decide, spoof_rule_flags

style_page()
st.title("IoT Attack Detector")
st.write("Upload a file of network connections. The detector checks every connection and tells you which ones look like attacks, and why.")
require_assets()

man = manifest()
FOLD = man["deploy_fold"]
bundle = fold_models()[FOLD]
MODEL, COLS, THRESHOLD = bundle["model"], bundle["features"], bundle["threshold"]
RAW = man["raw_features"]
REQUIRED = ["proto", "service", "conn_state", "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "src_ip_bytes", "dst_ip_bytes"]
reference = replay_flows()
NUMERIC = {c for c in RAW if c in reference.columns and pd.api.types.is_numeric_dtype(reference[c])} | {"src_port", "dst_port"}
SAMPLE = APP_DIR / "sample_data" / "unseen_flows_with_labels.xlsx"
SIMPLE_RULES = {
    "rule_icmp_redirect": "Fake route message (ICMP redirect)",
    "rule_icmp_ttl_exceeded": "Extra hop in the path (TTL exceeded)",
    "rule_dns_spoof": "Fake DNS answer",
    "rule_tls_strip": "Encryption removed (TLS stripping)",
}
RISK_ICON = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟡 Low", "—": "🟢 None"}
state = st.session_state

# ------------------------------------------------------------------ step 1: get a file
st.subheader("1. Choose a file")
c1, c2 = st.columns([3, 1.2])
uploaded = c1.file_uploader("Excel (.xlsx) or CSV with TON_IoT columns", type=["xlsx", "csv"], label_visibility="collapsed")
if c2.button("Try a sample file", icon=":material/science:", type="primary", width="stretch"):
    state.use_sample = True
if SAMPLE.exists():
    c2.download_button("Download sample", SAMPLE.read_bytes(), file_name=SAMPLE.name, icon=":material/download:", width="stretch")
if uploaded is not None:
    state.use_sample = False

if uploaded is not None:
    data, name = uploaded.getvalue(), uploaded.name
elif state.get("use_sample") and SAMPLE.exists():
    data, name = SAMPLE.read_bytes(), "sample file (devices the model has never seen)"
else:
    st.info("Upload a file, or click **Try a sample file**.")
    st.stop()


@st.cache_data(show_spinner="Reading file…")
def read_file(raw: bytes, is_csv: bool) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw), low_memory=False) if is_csv else pd.read_excel(io.BytesIO(raw))


frame = read_file(data, name.lower().endswith(".csv"))
canonical = {c.lower(): c for c in RAW + ["src_ip", "dst_ip", "label", "type"]}
frame.columns = [canonical.get(str(c).strip().lower(), str(c).strip()) for c in frame.columns]
missing = [c for c in REQUIRED if c not in frame.columns]
if missing or frame.empty:
    st.error("This file doesn't look like network-flow data. It needs these columns: " + ", ".join(REQUIRED) + "."
             + (f" Missing: **{', '.join(missing)}**." if missing else " The file has no rows."))
    st.stop()
for c in RAW:
    if c not in frame.columns:
        frame[c] = 0 if c in NUMERIC else "-"
    elif c in NUMERIC:
        frame[c] = pd.to_numeric(frame[c], errors="coerce").fillna(0)
    else:
        frame[c] = frame[c].fillna("-").astype(str)

key = hashlib.md5(data).hexdigest()
if state.get("key") != key:
    with st.spinner(f"Checking {len(frame):,} connections…"):
        rules = spoof_rule_flags(frame)
        X = pd.concat([frame[[c for c in COLS if c not in RULE_COLUMNS]], rules], axis=1)[COLS]
        state.key, state.X, state.rules, state.decision = key, X, rules, decide(MODEL, X, rules, THRESHOLD)
X, rules, decision = state.X, state.rules, state.decision
pred = decision["predicted"]
is_attack = pred != "normal"

# ------------------------------------------------------------------ step 2: verdict
st.subheader("2. Result")
n, n_attack = len(frame), int(is_attack.sum())
if n_attack:
    st.error(f"### ⚠️ {n_attack:,} of {n:,} connections look like attacks ({n_attack / n:.0%})")
else:
    st.success(f"### ✅ No attacks found in {n:,} connections")
st.caption(f"File: {name}")

left, right = st.columns([1.3, 1])
with left:
    counts = pred.map(FAMILY_LABELS).value_counts().rename_axis("Result").reset_index(name="Connections")
    counts["color"] = np.where(counts["Result"] == "Normal", "Normal", "Attack")
    fig = px.bar(counts, x="Connections", y="Result", orientation="h", text="Connections", color="color",
                 color_discrete_map={"Normal": "#1d9e75", "Attack": "#d0453b"})
    fig.update_layout(height=320, showlegend=False, yaxis=dict(categoryorder="total ascending", title=None), xaxis_title=None, margin=dict(t=10, l=0, r=10))
    st.plotly_chart(fig, key="counts")
with right:
    st.markdown("**Man-in-the-middle (spoofing) check**")
    st.caption("Signs that someone is secretly sitting between devices and the network:")
    for col, label in SIMPLE_RULES.items():
        k = int(rules[col].sum())
        st.markdown(f"{'🔴' if k else '⚪'} {label}: **{k:,}**")
    st.markdown(f"**{int((pred == 'mitm').sum()):,} connections flagged as man-in-the-middle**")

truth = None
if "type" in frame.columns:
    truth = frame["type"].astype(str).str.strip().str.lower()
elif "label" in frame.columns:
    truth = pd.Series(np.where(pd.to_numeric(frame["label"], errors="coerce").fillna(0).astype(int) == 1, "attack", "normal"), index=frame.index)
if truth is not None:
    real_attack = truth != "normal"
    caught = int((real_attack & is_attack).sum())
    false_alarms = int((~real_attack & is_attack).sum())
    st.success(f"**Accuracy check** (this file includes the true answers): caught **{caught:,} of {int(real_attack.sum()):,} real attacks "
               f"({caught / max(int(real_attack.sum()), 1):.0%})**, with **{false_alarms:,} false alarms** out of {int((~real_attack).sum()):,} normal connections.")

# ------------------------------------------------------------------ step 3: details
st.subheader("3. Details")
only_attacks = st.toggle("Show attacks only", value=True)


def why(i):
    hits = [SIMPLE_RULES[c] for c in RULE_COLUMNS if rules.at[i, c] == 1]
    if hits:
        return ", ".join(hits)
    return "Normal-looking traffic" if pred.at[i] == "normal" else f"Traffic pattern matches {FAMILY_LABELS[pred.at[i]]}"


table = pd.DataFrame({
    "Row": frame.index + 2,
    "Device": frame["src_ip"] if "src_ip" in frame.columns else "",
    "Result": pred.map(FAMILY_LABELS),
    "Risk": pred.map(SEVERITY).map(RISK_ICON),
    "Why": [why(i) for i in frame.index],
}, index=frame.index)
if truth is not None:
    table["Actual"] = truth.map(lambda v: FAMILY_LABELS.get(v, v.title()))
view = table[is_attack] if only_attacks else table
st.dataframe(view.head(1000), hide_index=True, height=360)

out = io.BytesIO()
pd.concat([frame, table[["Result", "Risk", "Why"]]], axis=1).to_excel(out, index=False)
st.download_button("Download results (Excel)", out.getvalue(), file_name="detection_results.xlsx", icon=":material/download:")

if n_attack:
    st.markdown("**Look closer at one alert**")
    order = table[is_attack].assign(m=pred[is_attack] == "mitm").sort_values("m", ascending=False).index[:300]
    pick = st.selectbox("Alert", order, label_visibility="collapsed",
                        format_func=lambda i: f"Row {i + 2} · {table.at[i, 'Result']} · {table.at[i, 'Device']}")
    row = X.loc[pick].copy()
    row["fold"], row["predicted"] = FOLD, pred.at[pick]
    with st.spinner("Working out the reasons…"):
        reasons = explain_flow(row, top_k=8)
    reasons = reasons[(reasons["contribution"] > 0) & (~reasons["feature"].isin(RULE_COLUMNS))].head(3)
    st.markdown(f"Row {pick + 2} was flagged as **{table.at[pick, 'Result']}** ({table.at[pick, 'Risk']} risk). Main reasons:")
    for r in reasons.itertuples():
        st.markdown(f"- **{r.label}** was `{str(r.value)[:40]}`")
    rule_list = [SIMPLE_RULES[c] for c in RULE_COLUMNS if rules.at[pick, c] == 1]
    if rule_list:
        st.markdown("- Spoofing check: **" + ", ".join(rule_list) + "**")
