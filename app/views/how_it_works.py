import streamlit as st

from demo_lib import results_csv, style_page

style_page()
st.title("How it works")

st.graphviz_chart(
    """
    digraph G {
      rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=13, color="#9aa4b1", fillcolor="#f4f6f9", margin="0.25,0.15"];
      a [label="1. Network\\nconnections"];
      b [label="2. Machine-learning model\\nlearns what each attack\\nlooks like", fillcolor="#e6f1fb", color="#2a78d6"];
      c [label="3. Spoofing check\\nlooks for signs of a\\nman-in-the-middle", fillcolor="#e1f5ee", color="#1d9e75"];
      d [label="4. Alert + reason\\nshown to a person", fillcolor="#fde8e8", color="#a32d2d"];
      a -> b -> d; a -> c -> d;
    }
    """,
    width="stretch",
)

pooled = results_csv("results/spoofing/metrics/spoofing_comparison_pooled.csv")
if pooled is not None:
    final = pooled[pooled["approach"].str.contains("final system")].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Attacks caught", f"{final['attack_recall']:.0%}")
    c2.metric("False alarms", f"{final['fpr']:.1%}")
    c3.metric("Connections studied", "183,570")
    st.caption("Tested on devices the model had never seen before, which is the fair way to test an intrusion detector.")

st.markdown("#### In plain words")
st.markdown(
    "- **The data:** TON_IoT, a lab network of smart devices (sensors, cameras, servers) where researchers ran 9 kinds of attacks.\n"
    "- **The model:** a Random Forest that learned the difference between normal traffic and each attack type.\n"
    "- **The spoofing check:** in a man-in-the-middle attack, the attacker pretends to be someone trusted. That leaves traces: "
    "fake *route* messages, fake *DNS* answers, and *encryption that was stripped away*. Four simple rules look for those traces.\n"
    "- **The alert:** each alert says what attack it looks like, how risky it is, and the main reasons, so a person can check before acting.\n"
    "- **Fair testing:** we never test on a device the model learned from. Testing that way showed earlier scores were too optimistic, so all the numbers here are the honest ones."
)
