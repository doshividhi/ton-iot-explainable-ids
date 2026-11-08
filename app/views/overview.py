import streamlit as st

from demo_lib import manifest, require_assets, results_csv, style_page

style_page()
st.title("Explainable intrusion detection for IoT networks")
st.caption("TON_IoT network flows · tested on devices the model has never seen · spoofing-aware MITM detection")
require_assets()

pooled = results_csv("results/spoofing/metrics/spoofing_comparison_pooled.csv")
if pooled is not None:
    final = pooled[pooled["approach"].str.contains("final system")].iloc[0]
    default = pooled[pooled["approach"].str.startswith("ML only (RF")].iloc[0]
    ml_cal = pooled[pooled["approach"].str.startswith("ML only, calibrated")].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Attacks caught (unseen devices)", f"{final['attack_recall']:.1%}")
    c2.metric("False alarms on normal traffic", f"{final['fpr']:.1%}", f"{(final['fpr'] - default['fpr']) * 100:+.1f} pts vs. default threshold", delta_color="inverse")
    c3.metric("MITM F1 with spoofing detectors", f"{final['mitm_f1']:.2f}", f"{final['mitm_f1'] - ml_cal['mitm_f1']:+.2f} vs. ML alone")
    c4.metric("Flows analysed", f"{manifest()['flows_total']:,}")

st.subheader("How the system works")
st.graphviz_chart(
    """
    digraph G {
      rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#9aa4b1", fillcolor="#f4f6f9"];
      edge [color="#7b8794"];
      flows [label="Network flows\\n(Zeek conn/DNS/TLS logs)"];
      prep  [label="Pre-processing\\nde-duplicate · drop IPs and ports"];
      ml    [label="Random Forest\\n10 traffic classes", fillcolor="#e6f1fb", color="#2a78d6"];
      rules [label="Spoofing detectors\\nICMP redirect · rogue DNS\\nTLS stripping · TTL anomaly", fillcolor="#e1f5ee", color="#1d9e75"];
      thr   [label="Calibrated threshold\\n1% false-alarm budget", fillcolor="#fdf3d7", color="#ba7517"];
      alert [label="Alert + explanation\\n(SHAP reasons, rule hits)", fillcolor="#fde8e8", color="#a32d2d"];
      human [label="Analyst decides\\nconfirm · dismiss · isolate"];
      flows -> prep; prep -> ml; prep -> rules; ml -> thr; rules -> thr [label=" escalate", fontsize=9];
      thr -> alert; alert -> human; human -> ml [style=dashed, label=" feedback\\n retrains", fontsize=9];
    }
    """,
    width="stretch",
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("#### Honest testing")
    st.write("Every number comes from **host-grouped** evaluation: the devices in the test set were never seen in training. "
             "A normal random split looked almost perfect (0.95 macro-F1) but hid device memorisation.")
with c2:
    st.markdown("#### Spoofing-aware MITM")
    st.write("Four detectors look for the footprints of a man-in-the-middle attack chain: forged routes (ICMP redirect), "
             "forged DNS answers, and stripped encryption. They catch 83% of MITM flows on their own.")
with c3:
    st.markdown("#### Every alert is explained")
    st.write("Each alert shows which spoofing rules fired and the top features behind the model's decision (SHAP), "
             "so an analyst can check the reasoning before acting.")

with st.expander("Suggested 5-minute demo"):
    st.markdown(
        """
1. **Live monitor**: press *Start*, then switch the threshold from *Calibrated* to *Default 0.5* and watch false alarms jump.
2. Open an alert in **Investigate an alert** to show the SHAP reasons and the analyst buttons.
3. **MITM spoofing lab**: walk through the attack chain and show an ICMP-redirect flow being caught.
4. **Attack simulator**: pick an evadable flow, drag the padding slider, and show the standard model missing it
   while the adversarially trained model still catches it.
5. **Drift monitor**: switch to UNSW-NB15 to show when the system would ask for retraining.
"""
    )
