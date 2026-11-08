import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demo_lib import drift_reference, replay_flows, require_assets, style_page

style_page()
st.title("Drift monitor")
st.caption("Is today's traffic still like the traffic the model learned from? If not, its alerts can't be trusted.")
require_assets()

ref = drift_reference()
flows = replay_flows()
NAMES = {"duration": "Duration", "src_pkts": "Packets sent", "dst_pkts": "Packets received", "src_ip_bytes": "IP bytes sent",
         "dst_ip_bytes": "IP bytes received", "src_bytes": "Payload bytes sent", "dst_bytes": "Payload bytes received"}


def psi_from_reference(feature, values):
    edges = np.array(ref["features"][feature]["edges"])
    expected = np.clip(np.array(ref["features"][feature]["props"]), 1e-6, None)
    actual = np.clip(np.histogram(np.log1p(np.asarray(values, float)), edges)[0] / max(len(values), 1), 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


options = ["Replay traffic seen so far (live monitor)", "Unseen TON_IoT devices (all replay traffic)"]
if ref.get("unsw_psi"):
    options.append("UNSW-NB15 (a different network)")
source = st.radio("Compare training traffic with", options, horizontal=True)

if source.startswith("UNSW"):
    psi = pd.Series(ref["unsw_psi"])
else:
    n = st.session_state.get("pos", 0) if source.startswith("Replay") else len(flows)
    if n < 200:
        st.info("Fewer than 200 flows replayed yet, so using the first 1,000 replay flows. Start the live monitor for a live view.")
        n = 1000
    window = flows.iloc[:n]
    psi = pd.Series({f: psi_from_reference(f, window[f]) for f in ref["features"]})

table = pd.DataFrame({"feature": psi.index.map(NAMES), "psi": psi.values})
table["status"] = pd.cut(table["psi"], [-np.inf, 0.1, 0.25, np.inf], labels=["Stable", "Moderate shift", "Major shift"])
colors = table["status"].map({"Stable": "#1d9e75", "Moderate shift": "#ba7517", "Major shift": "#a32d2d"}).tolist()

major = int((table["status"] == "Major shift").sum())
if major >= 2:
    st.error(f"**Retraining review triggered:** {major} features show a major shift (PSI > 0.25). The ML owner should collect a small labelled "
             "sample from this traffic and retrain; until then, treat alerts as lower confidence.")
elif major == 1 or (table["status"] == "Moderate shift").any():
    st.warning("Some features are shifting. Keep watching; investigate if it lasts three days.")
else:
    st.success("Traffic looks like the training data. No action needed.")

fig = go.Figure(go.Bar(x=table["psi"], y=table["feature"], orientation="h", marker_color=colors, text=table["psi"].round(2), textposition="outside"))
fig.add_vline(x=0.1, line_dash="dot", line_color="#ba7517", annotation_text="0.1 watch", annotation_position="bottom left")
fig.add_vline(x=0.25, line_dash="dash", line_color="#a32d2d", annotation_text="0.25 act", annotation_position="top right")
fig.update_layout(height=380, xaxis_title="Population Stability Index (higher = more different from training)", yaxis_title=None, margin=dict(t=40))
st.plotly_chart(fig, key="psi")

st.markdown("##### Response workflow")
st.dataframe(pd.DataFrame([
    {"Signal": "Feature drift (PSI)", "Trigger": "PSI > 0.25 on 2+ features for 3 days", "Who acts": "ML owner", "What happens": "Label a small sample, retrain, recalibrate threshold"},
    {"Signal": "Alert quality", "Trigger": "Estimated false alarms > 1%, or a device at 5× its usual alert rate", "Who acts": "SOC analyst", "What happens": "Triage, mute noisy pattern, open ticket"},
    {"Signal": "High-risk alert", "Trigger": "MITM, ransomware or backdoor above threshold", "Who acts": "On-call analyst (15 min)", "What happens": "Investigate; isolate the device only after human confirmation"},
]), hide_index=True)
