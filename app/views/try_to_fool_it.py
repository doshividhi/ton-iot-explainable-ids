import numpy as np
import streamlit as st

from demo_lib import FAMILY_LABELS, manifest, require_assets, simulator_models, simulator_pool, style_page
from ids_common import perturb

style_page()
st.title("Try to fool it")
st.write("Attackers try to hide by **adding junk data** and **slowing down** so their traffic looks normal. "
         "Pick an attack, raise the disguise, and see which detector gets fooled.")
require_assets()

pool, models, cols = simulator_pool(), simulator_models(), manifest()["features"]
state = st.session_state

families = sorted(pool["type"].unique(), key=lambda f: FAMILY_LABELS[f])
c1, c2 = st.columns([2, 1])
family = c1.selectbox("Attack", families, index=families.index("password") if "password" in families else 0, format_func=FAMILY_LABELS.get)
candidates = pool[(pool["type"] == family) & pool["evades_standard"]]
if candidates.empty:
    candidates = pool[pool["type"] == family]
if c2.button("Another example", icon=":material/shuffle:", width="stretch") or state.get("fool_family") != family or state.get("fool_idx") not in candidates.index:
    state.fool_idx, state.fool_family = int(np.random.default_rng().choice(candidates.index)), family

level = st.select_slider("Disguise level", options=["None", "Low", "Medium", "High", "Maximum"], value="None")
pad, slow = {"None": (0, 1), "Low": (256, 2.5), "Medium": (512, 5), "High": (768, 7.5), "Maximum": (1024, 10)}[level]
flow = perturb(pool.loc[[state.fool_idx]], pad, slow)
st.caption(f"Disguise adds {pad} bytes of junk data and makes the connection {slow:g}× slower.")

r1, r2 = st.columns(2)
for col, key, title, note in ((r1, "standard", "Normal detector", "Trained only on real attacks"),
                              (r2, "adversarial", "Hardened detector", "Also trained on disguised attacks")):
    pred = models[key].predict(flow[cols])[0]
    with col:
        if pred == "normal":
            st.error(f"#### {title}\n❌ **Fooled** — thinks it's normal traffic")
        else:
            st.success(f"#### {title}\n✅ **Caught it** — {FAMILY_LABELS[pred]}")
        st.caption(note)

st.info("**Takeaway:** a detector that has only seen ordinary attacks can be fooled by simple disguises. "
        "Training it on disguised examples too (\"adversarial training\") closes that gap at almost no cost.")
