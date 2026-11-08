import hashlib
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from demo_lib import (APP_DIR, FAMILY_LABELS, SEVERITY, explain_flow, fold_models, manifest, pill, replay_flows,
                      require_assets, rule_hits, severity_kind, style_page)
from ids_common import RULE_COLUMNS, SPOOF_RULES, decide, spoof_rule_flags

style_page()
st.title("Upload & detect")
st.caption("Upload a file of network flows the model has never seen. Every flow is checked for attacks and MITM spoofing, and each alert comes with its reasons.")
require_assets()

man = manifest()
FOLD = man["deploy_fold"]
bundle = fold_models()[FOLD]
MODEL, COLS, THRESHOLD = bundle["model"], bundle["features"], bundle["threshold"]
RAW = man["raw_features"]
REQUIRED = ["proto", "service", "conn_state", "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "src_ip_bytes", "dst_ip_bytes"]
reference = replay_flows()
NUMERIC = {c for c in RAW if c in reference.columns and pd.api.types.is_numeric_dtype(reference[c])} | {"src_port", "dst_port"}
SAMPLES = APP_DIR / "sample_data"
state = st.session_state

# ---------------------------------------------------------------- file input
with st.expander("What file can I upload?", expanded=False):
    st.markdown(
        "- An **Excel (.xlsx) or CSV** file with one row per network flow, using the TON_IoT / Zeek column names "
        "(for example `proto`, `service`, `conn_state`, `duration`, `src_bytes`, `dst_bytes`, `src_pkts`, `dst_pkts`).\n"
        f"- **Required columns:** {', '.join(REQUIRED)}. Missing optional columns (DNS, TLS, HTTP fields) are filled with "
        "\"not used\" values, and the app tells you which ones.\n"
        "- If the file also has a `type` or `label` column (the true answer), the app **grades its own detection** on that file.\n"
        "- The model in this page never saw the devices in the sample files below, so they are a fair test."
    )
    c1, c2, c3 = st.columns(3)
    for col, name, label in ((c1, "unseen_flows_with_labels.xlsx", "Sample with true labels (.xlsx)"),
                             (c2, "unseen_flows_no_labels.xlsx", "Sample without labels (.xlsx)"),
                             (c3, "unseen_flows_no_labels.csv", "Sample without labels (.csv)")):
        path = SAMPLES / name
        if path.exists():
            col.download_button(label, path.read_bytes(), file_name=name, icon=":material/download:", width="stretch")

up_col, sample_col = st.columns([3, 1])
uploaded = up_col.file_uploader("Upload network flows", type=["xlsx", "csv"])
if sample_col.button("Use the sample unseen file", icon=":material/science:", width="stretch"):
    state.use_sample = True
if uploaded is not None:
    state.use_sample = False

if uploaded is not None:
    data, file_name = uploaded.getvalue(), uploaded.name
elif state.get("use_sample") and (SAMPLES / "unseen_flows_with_labels.xlsx").exists():
    data, file_name = (SAMPLES / "unseen_flows_with_labels.xlsx").read_bytes(), "unseen_flows_with_labels.xlsx (sample)"
else:
    st.info("Upload an Excel or CSV file, or click **Use the sample unseen file** to try it.")
    st.stop()


@st.cache_data(show_spinner="Reading file…")
def read_file(raw: bytes, name: str) -> pd.DataFrame:
    if name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw), low_memory=False)
    return pd.read_excel(io.BytesIO(raw))


frame = read_file(data, file_name.replace(" (sample)", ""))
canonical = {c.lower(): c for c in RAW + ["src_ip", "dst_ip", "label", "type"]}
frame.columns = [canonical.get(str(c).strip().lower(), str(c).strip()) for c in frame.columns]

missing_required = [c for c in REQUIRED if c not in frame.columns]
if missing_required:
    st.error(f"This file is missing required columns: **{', '.join(missing_required)}**. Check that it uses the TON_IoT column names.")
    st.stop()
if frame.empty:
    st.error("The file has no rows.")
    st.stop()

filled = [c for c in RAW if c not in frame.columns]
for c in RAW:
    if c not in frame.columns:
        frame[c] = 0 if c in NUMERIC else "-"
    elif c in NUMERIC:
        frame[c] = pd.to_numeric(frame[c], errors="coerce").fillna(0)
    else:
        frame[c] = frame[c].fillna("-").astype(str)
if filled:
    st.warning(f"{len(filled)} optional columns were not in the file and were filled as \"not used\": {', '.join(filled[:12])}{'…' if len(filled) > 12 else ''}.")

# ---------------------------------------------------------------- scoring
key = hashlib.md5(data).hexdigest()
if state.get("upload_key") != key:
    with st.spinner(f"Scoring {len(frame):,} flows…"):
        rules = spoof_rule_flags(frame)
        X = pd.concat([frame[[c for c in COLS if c not in RULE_COLUMNS]], rules], axis=1)[COLS]
        decision = decide(MODEL, X, rules, THRESHOLD)
    state.upload_key, state.upload_X, state.upload_rules, state.upload_decision = key, X, rules, decision
X, rules, decision = state.upload_X, state.upload_rules, state.upload_decision
pred = decision["predicted"]
is_attack = pred != "normal"
rule_names = rules.apply(lambda r: ", ".join(SPOOF_RULES[c.replace("rule_", "")]["title"] for c in RULE_COLUMNS if r[c] == 1), axis=1)

st.success(f"Scored **{len(frame):,} flows** from *{file_name}* with the hybrid detector (calibrated alert threshold {THRESHOLD:.2f}).")
m = st.columns(5)
m[0].metric("Flows checked", f"{len(frame):,}")
m[1].metric(f"Attacks detected ({is_attack.mean():.0%})", f"{int(is_attack.sum()):,}")
m[2].metric("MITM detected", f"{int((pred == 'mitm').sum()):,}")
m[3].metric("Spoofing rules fired", f"{int((rules.sum(axis=1) > 0).sum()):,}")
m[4].metric("High-severity alerts", f"{int(pred.map(SEVERITY).eq('High').sum()):,}")

c1, c2 = st.columns([1.4, 1])
with c1:
    counts = pred[is_attack].map(FAMILY_LABELS).value_counts().rename_axis("Attack type").reset_index(name="Flows")
    if not counts.empty:
        fig = px.bar(counts, x="Flows", y="Attack type", orientation="h", color_discrete_sequence=["#2a78d6"], text="Flows")
        fig.update_layout(title="Detected attacks by type", height=340, yaxis=dict(categoryorder="total ascending", title=None), margin=dict(t=40, l=0))
        st.plotly_chart(fig, key="up_types")
    else:
        st.info("No attacks detected in this file.")
with c2:
    rule_counts = pd.DataFrame({"Detector": [SPOOF_RULES[c.replace("rule_", "")]["title"] for c in RULE_COLUMNS], "Flows": [int(rules[c].sum()) for c in RULE_COLUMNS]})
    fig = px.bar(rule_counts, x="Flows", y="Detector", orientation="h", color_discrete_sequence=["#1d9e75"], text="Flows")
    fig.update_layout(title="MITM spoofing detectors", height=340, yaxis_title=None, margin=dict(t=40, l=0))
    st.plotly_chart(fig, key="up_rules")

# ---------------------------------------------------------------- grading (if labels are present)
truth = None
if "type" in frame.columns:
    truth = frame["type"].astype(str).str.strip().str.lower()
elif "label" in frame.columns:
    truth = np.where(pd.to_numeric(frame["label"], errors="coerce").fillna(0).astype(int) == 1, "attack", "normal")
    truth = pd.Series(truth, index=frame.index)
if truth is not None:
    st.subheader("How well did it do on this file?")
    st.caption("The file includes the true answers, so the app can grade itself. The model never saw these labels.")
    t_attack, p_attack = truth != "normal", is_attack
    tp, fp = int((t_attack & p_attack).sum()), int((~t_attack & p_attack).sum())
    fn, tn = int((t_attack & ~p_attack).sum()), int((~t_attack & ~p_attack).sum())
    g = st.columns(4)
    g[0].metric(f"Attacks caught ({tp:,} of {tp + fn:,})", f"{tp / max(tp + fn, 1):.1%}")
    g[1].metric(f"False alarms ({fp:,} of {fp + tn:,} normal)", f"{fp / max(fp + tn, 1):.1%}")
    g[2].metric("Alert precision", f"{tp / max(tp + fp, 1):.1%}")
    if "type" in frame.columns and (truth == "mitm").any():
        g[3].metric("MITM caught", f"{((truth == 'mitm') & (pred == 'mitm')).sum() / (truth == 'mitm').sum():.1%}")
    if "type" in frame.columns:
        labels = sorted(set(truth) | set(pred))
        cm = pd.crosstab(truth, pred).reindex(index=labels, columns=labels, fill_value=0)
        norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
        names = [FAMILY_LABELS.get(c, c) for c in labels]
        fig = go.Figure(go.Heatmap(z=norm.values, x=names, y=names, colorscale="Blues", zmin=0, zmax=1,
                                   text=[[f"{v:.0%}" if v >= 0.02 else "" for v in r] for r in norm.values], texttemplate="%{text}",
                                   hovertemplate="True %{y}<br>Predicted %{x}<br>%{z:.1%}<extra></extra>"))
        fig.update_layout(title="True vs. detected (row %)", height=470, yaxis=dict(autorange="reversed", title="True"), xaxis_title="Detected")
        st.plotly_chart(fig, key="up_cm")

# ---------------------------------------------------------------- results table + download
st.subheader("Results")
show = st.radio("Show", ["Attacks only", "MITM only", "All flows"], horizontal=True)
results = pd.DataFrame({
    "Row": frame.index + 2,
    "Device": frame["src_ip"] if "src_ip" in frame.columns else "",
    "Target": frame["dst_ip"] if "dst_ip" in frame.columns else "",
    "Traffic": frame["proto"].astype(str) + " / " + frame["service"].astype(str).replace("-", "other"),
    "Detected as": pred.map(FAMILY_LABELS),
    "Severity": pred.map(SEVERITY),
    "Attack score": decision["attack_score"].round(4),
    "Spoofing rules": rule_names,
})
if truth is not None:
    results["True label"] = truth.map(lambda v: FAMILY_LABELS.get(v, v.title()))
view = results[is_attack] if show == "Attacks only" else results[pred == "mitm"] if show == "MITM only" else results
st.dataframe(view.head(1000), hide_index=True, height=380, column_config={"Attack score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f")})
st.caption(f"Showing {min(len(view), 1000):,} of {len(view):,} rows. 'Row' is the row number in your file.")

out = io.BytesIO()
pd.concat([frame, results[["Detected as", "Severity", "Attack score", "Spoofing rules"]]], axis=1).to_excel(out, index=False)
d1, d2 = st.columns(2)
d1.download_button("Download results (.xlsx)", out.getvalue(), file_name="detection_results.xlsx", icon=":material/download:", width="stretch")
d2.download_button("Download results (.csv)", pd.concat([frame, results[["Detected as", "Severity", "Attack score", "Spoofing rules"]]], axis=1).to_csv(index=False).encode(),
                   file_name="detection_results.csv", icon=":material/download:", width="stretch")

# ---------------------------------------------------------------- explanation
st.subheader("Why was this flow flagged?")
attacks = results[is_attack]
if attacks.empty:
    st.caption("No attacks to explain.")
    st.stop()
order = attacks.assign(mitm=pred[is_attack] == "mitm").sort_values(["mitm", "Attack score"], ascending=[False, False]).index[:300]
choice = st.selectbox("Alert", order, format_func=lambda i: f"Row {i + 2} · {results.at[i, 'Detected as']} · score {results.at[i, 'Attack score']:.2f} · {results.at[i, 'Traffic']}")
row = X.loc[choice].copy()
row["fold"], row["predicted"] = FOLD, pred.loc[choice]
fam = pred.loc[choice]

a, b = st.columns([1.1, 1.4])
with a:
    st.markdown(f"#### {FAMILY_LABELS[fam]} {pill(SEVERITY[fam] + ' severity', severity_kind(fam))}", unsafe_allow_html=True)
    st.metric("Attack score", f"{decision.at[choice, 'attack_score']:.3f}")
    hits = rule_hits(row)
    if hits:
        st.markdown("**Spoofing detectors that fired**")
        for h in hits:
            st.markdown(f"<div class='card'><b>{h['title']}</b> {pill(h['layer'], 'low')}<br><span class='muted'>{h['why']}</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='muted'>No spoofing detector fired; this alert comes from the ML model.</span>", unsafe_allow_html=True)
with b:
    with st.spinner("Explaining…"):
        exp = explain_flow(row)
    exp["direction"] = exp["contribution"].map(lambda v: "pushes towards this alert" if v > 0 else "pushes away")
    exp["shown"] = exp["label"] + " = " + exp["value"].astype(str).str.slice(0, 28)
    fig = px.bar(exp.iloc[::-1], x="contribution", y="shown", orientation="h", color="direction",
                 color_discrete_map={"pushes towards this alert": "#eb6834", "pushes away": "#2a78d6"})
    fig.update_layout(title="Top reasons (SHAP)", height=320, margin=dict(l=0, r=10, t=40, b=0), legend=dict(orientation="h", y=-0.25, title=None), yaxis_title=None, xaxis_title="Contribution to the detected class")
    st.plotly_chart(fig, key="up_shap")
