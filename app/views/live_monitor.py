import pandas as pd
import plotly.express as px
import streamlit as st

from demo_lib import (FAMILY_LABELS, SEVERITY, explain_flow, pill, replay_flows, require_assets, rule_hits,
                      severity_kind, style_page)

style_page()
st.title("Live monitor")
st.caption("Replays traffic from devices the model never saw in training. Each flow is scored by the model that held its device out.")
require_assets()

flows = replay_flows()
state = st.session_state
state.setdefault("pos", 0)
state.setdefault("running", False)
state.setdefault("feedback", {})

controls = st.columns([2.2, 1.2, 1, 1, 1])
mode = controls[0].radio("Alert threshold", ["Calibrated (1% false-alarm budget)", "Default 0.5"], horizontal=True, key="mode")
speed = controls[1].selectbox("Flows per second", [10, 25, 50, 100], index=1, key="speed")
if controls[2].button("Start", icon=":material/play_arrow:", type="primary", width="stretch"):
    state.running = True
if controls[3].button("Pause", icon=":material/pause:", width="stretch"):
    state.running = False
if controls[4].button("Reset", icon=":material/restart_alt:", width="stretch"):
    state.pos, state.running = 0, False
PRED = "predicted" if mode.startswith("Calibrated") else "predicted_default"


@st.fragment(run_every=1.0)
def live_panel():
    if state.running:
        state.pos = min(state.pos + state.speed, len(flows))
        if state.pos >= len(flows):
            state.running = False
    seen = flows.iloc[: state.pos]
    is_alert = seen[PRED] != "normal"
    truth_attack = seen["type"] != "normal"
    normal_n = int((~truth_attack).sum())

    m = st.columns(5)
    m[0].metric("Flows scored", f"{len(seen):,}")
    m[1].metric("Alerts raised", f"{int(is_alert.sum()):,}")
    m[2].metric("MITM alerts", f"{int((seen[PRED] == 'mitm').sum()):,}")
    fa, missed = int((is_alert & ~truth_attack).sum()), int((~is_alert & truth_attack).sum())
    fa_pct = f" ({fa / normal_n:.1%})" if normal_n else ""
    miss_pct = f" ({missed / int(truth_attack.sum()):.0%})" if truth_attack.any() else ""
    m[3].metric(f"False alarms{fa_pct}", f"{fa:,}", help="Normal flows that raised an alert (share of normal flows so far)")
    m[4].metric(f"Attacks missed{miss_pct}", f"{missed:,}", help="Attack flows that raised no alert")
    st.progress(state.pos / len(flows), text=f"{'Streaming' if state.running else 'Paused'} · {state.pos:,} of {len(flows):,} flows")

    left, right = st.columns([2.3, 1])
    with left:
        st.markdown("##### Latest alerts")
        alerts = seen[is_alert].tail(12).iloc[::-1]
        if alerts.empty:
            st.info("No alerts yet. Press Start.")
        else:
            table = pd.DataFrame({
                "Flow": alerts["flow_id"],
                "Device": alerts["src_ip"],
                "Traffic": alerts["proto"] + " / " + alerts["service"].replace("-", "other"),
                "Detected as": alerts[PRED].map(FAMILY_LABELS),
                "Severity": alerts[PRED].map(SEVERITY),
                "Attack score": alerts["attack_score"],
                "Spoofing rule": alerts[[c for c in alerts.columns if c.startswith("rule_")]].sum(axis=1).gt(0).map({True: "Yes", False: ""}),
                "Ground truth": alerts["type"].map(FAMILY_LABELS),
            })
            st.dataframe(table, hide_index=True, height=420, column_config={
                "Attack score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
            })
    with right:
        st.markdown("##### Alerts by type")
        counts = seen.loc[is_alert, PRED].map(FAMILY_LABELS).value_counts().rename_axis("type").reset_index(name="alerts")
        if not counts.empty:
            fig = px.bar(counts, x="alerts", y="type", orientation="h", color_discrete_sequence=["#2a78d6"])
            fig.update_layout(height=420, margin=dict(l=0, r=10, t=10, b=0), yaxis=dict(categoryorder="total ascending", title=None), xaxis_title=None)
            st.plotly_chart(fig, key="alerts_by_type")


live_panel()

st.divider()
st.subheader("Investigate an alert")
seen = flows.iloc[: state.pos]
alerts = seen[seen[PRED] != "normal"].tail(60).iloc[::-1]
if alerts.empty:
    st.caption("Start the replay, then pick an alert here. Tip: pause first so the list stays still.")
    st.stop()

labels = {r.flow_id: f"Flow {r.flow_id} · {FAMILY_LABELS[getattr(r, PRED)]} · score {r.attack_score:.2f} · {r.src_ip}" for r in alerts.itertuples()}
mitm_first = sorted(labels, key=lambda f: (alerts.set_index("flow_id").loc[f, PRED] != "mitm"))
choice = st.selectbox("Alert", mitm_first, format_func=labels.get)
row = flows.set_index("flow_id").loc[choice]
row["fold"] = int(row["fold"])

a, b = st.columns([1.1, 1.4])
with a:
    fam = row[PRED]
    st.markdown(f"#### {FAMILY_LABELS[fam]} {pill(SEVERITY[fam] + ' severity', severity_kind(fam))}", unsafe_allow_html=True)
    st.markdown(f"<span class='muted'>Device {row['src_ip']} → {row['dst_ip']} · {row['proto']}/{row['service']} · "
                f"connection state {row['conn_state']}</span>", unsafe_allow_html=True)
    st.metric("Attack score", f"{row['attack_score']:.3f}")
    hits = rule_hits(row)
    if hits:
        st.markdown("**Spoofing detectors that fired**")
        for h in hits:
            st.markdown(f"<div class='card'><b>{h['title']}</b> {pill(h['layer'], 'low')}<br><span class='muted'>{h['why']}</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='muted'>No spoofing detector fired; this alert comes from the ML model.</span>", unsafe_allow_html=True)
    verdict = "correct" if (row["type"] == fam) else ("attack, different family" if row["type"] != "normal" else "false alarm")
    st.caption(f"Lab ground truth: **{FAMILY_LABELS[row['type']]}** ({verdict}). In a real SOC this is what the analyst decides.")

with b:
    st.markdown("**Why the model flagged it (SHAP)**")
    with st.spinner("Explaining…"):
        exp = explain_flow(row)
    exp["direction"] = exp["contribution"].map(lambda v: "pushes towards this alert" if v > 0 else "pushes away")
    exp["shown"] = exp["label"] + " = " + exp["value"].astype(str).str.slice(0, 28)
    fig = px.bar(exp.iloc[::-1], x="contribution", y="shown", orientation="h", color="direction",
                 color_discrete_map={"pushes towards this alert": "#eb6834", "pushes away": "#2a78d6"})
    fig.update_layout(height=300, margin=dict(l=0, r=10, t=10, b=0), legend=dict(orientation="h", y=-0.2, title=None), yaxis_title=None, xaxis_title="Contribution to the predicted class")
    st.plotly_chart(fig, key="shap_bar")

    f1, f2, f3 = st.columns(3)
    if f1.button("Confirm attack", icon=":material/check_circle:", width="stretch"):
        state.feedback[choice] = "confirmed"
    if f2.button("Mark false alarm", icon=":material/cancel:", width="stretch"):
        state.feedback[choice] = "false alarm"
    if f3.button("Isolate device", icon=":material/block:", width="stretch"):
        state.feedback[choice] = "isolation requested (needs human sign-off)"
    if state.feedback:
        fb = pd.Series(state.feedback).value_counts()
        st.caption("Analyst feedback this session: " + ", ".join(f"{v} {k}" for k, v in fb.items())
                   + ". Confirmed and dismissed alerts join the weekly retraining set.")
