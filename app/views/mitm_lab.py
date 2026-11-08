import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demo_lib import FAMILY_LABELS, RULE_COLUMNS, SPOOF_RULES, pill, replay_flows, require_assets, results_csv, style_page

style_page()
st.title("MITM spoofing lab")
st.caption("How the system spots a man-in-the-middle attack from the traces its spoofing leaves behind.")
require_assets()

st.markdown(
    "A man-in-the-middle attacker has to **put themselves in the path** of a victim's traffic and then **keep the victim "
    "from noticing**. Tools such as Ettercap and bettercap do this in layers. Each layer leaves a footprint in network "
    "flow logs, and each detector below looks for one footprint using behaviour only (never IP addresses)."
)
st.graphviz_chart(
    """
    digraph G {
      rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
      atk [label="Attacker", fillcolor="#fde8e8", color="#a32d2d"];
      r1 [label="1. Re-route the victim\\nICMP Redirect (type 5)\\nTTL-exceeded bursts", fillcolor="#f4f6f9", color="#9aa4b1"];
      r2 [label="2. Fake the phone book\\nRogue DNS answers\\n(recursion flag the real\\nresolver never sets)", fillcolor="#f4f6f9", color="#9aa4b1"];
      r3 [label="3. Remove the lock\\nTLS stripping: a 'secure'\\nsession with no handshake", fillcolor="#f4f6f9", color="#9aa4b1"];
      vic [label="Victim traffic\\nread or altered", fillcolor="#fde8e8", color="#a32d2d"];
      det [label="Our detectors flag\\neach footprint", fillcolor="#e1f5ee", color="#1d9e75"];
      atk -> r1 -> r2 -> r3 -> vic;
      r1 -> det [style=dashed, color="#1d9e75"]; r2 -> det [style=dashed, color="#1d9e75"]; r3 -> det [style=dashed, color="#1d9e75"];
    }
    """,
    width="stretch",
)

hits = results_csv("results/spoofing/metrics/spoof_rule_hits.csv")
if hits is not None:
    st.subheader("What each detector catches (all 183,570 flows)")
    cols = st.columns(4)
    for col, (key, meta) in zip(cols, SPOOF_RULES.items()):
        row = hits[hits["rule"] == meta["title"]].iloc[0]
        with col:
            st.markdown(f"<div class='card'><b>{meta['title']}</b><br>{pill(meta['layer'], 'low')}<br><br>"
                        f"<span style='font-size:1.6rem;font-weight:600'>{int(row['mitm_caught'])}</span> MITM flows caught<br>"
                        f"<span class='muted'>{row['mitm_recall']:.1%} of MITM · {row['normal_fpr']:.2%} of normal flagged</span><br><br>"
                        f"<span class='muted'>{meta['why']}</span></div>", unsafe_allow_html=True)
    any_row = hits[hits["rule"] == "Any spoofing rule"].iloc[0]
    st.success(f"Together the four detectors catch **{any_row['mitm_recall']:.0%} of all MITM flows** while flagging "
               f"only **{any_row['normal_fpr']:.1%} of normal traffic**. ICMP Redirect never fires on normal traffic.")

pooled = results_csv("results/spoofing/metrics/spoofing_comparison_pooled.csv")
if pooled is not None:
    st.subheader("Rules vs. machine learning vs. both (unseen devices)")
    short = {
        "ML only (RF, no ports), threshold 0.5": "ML only (t=0.5)",
        "Spoofing rules only": "Rules only",
        "Hybrid (RF + rule features), threshold 0.5": "Hybrid (t=0.5)",
        "ML only, calibrated threshold": "ML only (calibrated)",
        "Hybrid, calibrated threshold (final system)": "Hybrid (calibrated) ★",
    }
    p = pooled.assign(name=pooled["approach"].map(short))
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for metric, color, label in (("mitm_precision", "#9aa4b1", "Precision"), ("mitm_recall", "#1d9e75", "Recall"), ("mitm_f1", "#2a78d6", "F1")):
            fig.add_bar(x=p["name"], y=p[metric], name=label, marker_color=color, text=p[metric].round(2), textposition="outside")
        fig.update_layout(title="MITM detection quality", barmode="group", height=360, yaxis=dict(range=[0, 1.05]), margin=dict(t=50, b=0), legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, key="mitm_quality")
    with c2:
        fig = go.Figure(go.Bar(x=p["name"], y=p["fpr"] * 100, marker_color=["#9aa4b1", "#1d9e75", "#2a78d6", "#9aa4b1", "#2a78d6"],
                               text=(p["fpr"] * 100).round(1).astype(str) + "%", textposition="outside"))
        fig.update_layout(title="False alarms on normal traffic (%)", height=360, margin=dict(t=50, b=0), yaxis_title="% of normal flows")
        st.plotly_chart(fig, key="mitm_fpr")
    st.caption("Rules alone catch most MITM flows but also fire on other attacks and cannot detect non-MITM attacks. "
               "The final system feeds the rule flags into the Random Forest, lets the two zero-false-alarm routing rules raise "
               "alerts on their own, and keeps false alarms near 1% with a calibrated threshold.")

st.subheader("Inspect a spoofing footprint")
flows = replay_flows()
rule_titles = {SPOOF_RULES[c.replace("rule_", "")]["title"]: c for c in RULE_COLUMNS}
pick = st.radio("Detector", list(rule_titles), horizontal=True)
col = rule_titles[pick]
examples = flows[flows[col] == 1]
if examples.empty:
    st.info("No replay flow triggers this detector.")
else:
    show = ["flow_id", "src_ip", "dst_ip", "proto", "service", "conn_state", "src_port", "dst_port", "dns_RA", "dns_rcode", "ssl_version", "src_bytes", "dst_bytes", "type", "predicted"]
    table = examples[show].rename(columns={"src_port": "src_port / ICMP type", "dst_port": "dst_port / ICMP code", "type": "ground truth", "predicted": "system decision"})
    table["ground truth"] = table["ground truth"].map(FAMILY_LABELS)
    table["system decision"] = table["system decision"].map(FAMILY_LABELS)
    st.dataframe(table.head(25), hide_index=True)
    share = (examples["type"] == "mitm").mean()
    st.caption(f"{len(examples)} replay flows trigger this detector; {share:.0%} of them are MITM in the lab ground truth. "
               "For ICMP, Zeek stores the ICMP type in the source-port field and the code in the destination-port field.")

with st.expander("What this cannot see (and how we'd add it)"):
    st.markdown(
        "- **ARP spoofing** happens below IP. These flow logs have no MAC addresses or ARP records, so we can't watch an IP "
        "suddenly move to a new MAC. With TON_IoT's raw packet captures or Zeek's ARP logs, a rule for *one IP claimed by two MACs* would add a fourth layer.\n"
        "- The DNS rule depends on knowing how the network's real resolver behaves. It's checked on training traffic in every fold "
        "(0.5–4% of normal DNS replies set the recursion flag), so it would need re-checking on a new network."
    )
