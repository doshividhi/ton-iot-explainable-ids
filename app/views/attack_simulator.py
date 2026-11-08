import numpy as np
import pandas as pd
import streamlit as st

from demo_lib import FAMILY_LABELS, manifest, pill, require_assets, results_csv, simulator_models, simulator_pool, style_page
from ids_common import perturb

style_page()
st.title("Attack simulator")
st.caption("Play the attacker: change your own traffic and see whether the detector still catches you.")
require_assets()

pool = simulator_pool()
models = simulator_models()
cols = manifest()["features"]
state = st.session_state

st.markdown(
    "**Threat model.** The attacker controls only their own packets: they can **pad payloads** with junk bytes and "
    "**slow down** to look like ordinary traffic. They can't change how the victim replies, and they never see our training data. "
    "Raw ports aren't model inputs, so swapping the source port does nothing."
)

c1, c2, c3 = st.columns([1.3, 1, 1])
families = sorted(pool["type"].unique(), key=lambda f: FAMILY_LABELS[f])
family = c1.selectbox("Attack type", families, index=families.index("password") if "password" in families else 0, format_func=FAMILY_LABELS.get)
only_evadable = c2.toggle("Only flows that can evade the standard model", value=True)
candidates = pool[pool["type"] == family]
if only_evadable and candidates["evades_standard"].any():
    candidates = candidates[candidates["evades_standard"]]
if c3.button("Pick another flow", icon=":material/shuffle:", width="stretch") or state.get("sim_family") != family or "sim_idx" not in state or state.sim_idx not in candidates.index:
    state.sim_idx = int(np.random.default_rng().choice(candidates.index))
    state.sim_family = family
flow = pool.loc[[state.sim_idx]]

state.setdefault("pad", 0)
state.setdefault("slow", 1.0)
b1, b2, _ = st.columns([1, 1, 2])
if b1.button("Apply maximum evasion", icon=":material/bolt:", width="stretch"):
    state.pad, state.slow = 1024, 10.0
if b2.button("Reset changes", icon=":material/undo:", width="stretch"):
    state.pad, state.slow = 0, 1.0
s1, s2 = st.columns(2)
pad = s1.slider("Payload padding (bytes added)", 0, 1024, step=64, key="pad")
slow = s2.slider("Slow-down factor (duration ×)", 1.0, 10.0, step=0.5, key="slow")
moved = perturb(flow, pad, slow)


def verdict(model, frame):
    proba = model.predict_proba(frame[cols])[0]
    classes = list(model.classes_)
    pred = classes[int(np.argmax(proba))]
    return pred, 1 - proba[classes.index("normal")]


r1, r2 = st.columns(2)
for col, key, title in ((r1, "standard", "Standard model"), (r2, "adversarial", "Adversarially trained model")):
    pred, score = verdict(models[key], moved)
    caught = pred != "normal"
    with col:
        st.markdown(f"<div class='card'><b>{title}</b><br><br>"
                    f"{pill('Caught: ' + FAMILY_LABELS[pred], 'high') if caught else pill('Evaded: looks normal', 'ok')}"
                    f"<br><br><span style='font-size:1.8rem;font-weight:600'>{score:.2f}</span> <span class='muted'>attack score</span></div>",
                    unsafe_allow_html=True)
        st.progress(float(score))

st.markdown("##### What the attacker changed")
fields = ["src_bytes", "src_pkts", "src_ip_bytes", "duration"]
def fmt(field, value):
    return f"{value:.5f}" if field == "duration" else f"{value:,.0f}"


diff = pd.DataFrame({"Field": ["Bytes sent", "Packets sent", "IP bytes sent", "Duration (s)"],
                     "Original": [fmt(f, flow[f].iloc[0]) for f in fields], "Modified": [fmt(f, moved[f].iloc[0]) for f in fields]})
st.dataframe(diff, hide_index=True)
service = flow["service"].iloc[0]
st.caption(f"Flow from the held-out test set · {flow['proto'].iloc[0]}/{'other' if service == '-' else service} · lab label: {FAMILY_LABELS[family]}")

adv = results_csv("results/status_report/metrics/adversarial_evasion.csv")
if adv is not None:
    with st.expander("Results from the report (all detected test attacks, random perturbations)"):
        show = adv[["model", "evasion_pad", "evasion_slow", "evasion_sport", "evasion_combined", "clean_macro_f1"]].rename(columns={
            "model": "Model", "evasion_pad": "Padding", "evasion_slow": "Slow-down", "evasion_sport": "Port change", "evasion_combined": "All combined", "clean_macro_f1": "Clean macro-F1"})
        st.dataframe(show.style.format({c: "{:.2%}" for c in ["Padding", "Slow-down", "Port change", "All combined"]} | {"Clean macro-F1": "{:.3f}"}), hide_index=True)
        st.caption("Evasion rate = share of detected attacks that become 'normal' after the change. Adversarial training "
                   "(adding perturbed copies of 30% of training attacks) closes the gap at almost no cost on clean traffic.")
