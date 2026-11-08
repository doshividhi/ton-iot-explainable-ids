"""TON_IoT Explainable IDS — demo dashboard.

Run from the project folder:
    python app/build_demo_assets.py      # once, ~3 minutes
    streamlit run app/app.py
"""
import streamlit as st

st.set_page_config(page_title="IoT Intrusion Detection Demo", page_icon=":material/shield:", layout="wide")

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/home:", default=True),
    st.Page("views/upload_detect.py", title="Upload & detect", icon=":material/upload_file:"),
    st.Page("views/live_monitor.py", title="Live monitor", icon=":material/monitor_heart:"),
    st.Page("views/mitm_lab.py", title="MITM spoofing lab", icon=":material/policy:"),
    st.Page("views/model_results.py", title="Model results", icon=":material/insights:"),
    st.Page("views/attack_simulator.py", title="Attack simulator", icon=":material/science:"),
    st.Page("views/drift_monitor.py", title="Drift monitor", icon=":material/monitoring:"),
]
nav = st.navigation(pages)
with st.sidebar:
    st.caption("IS 665 Cybersecurity Analytics · Group 11")
    st.caption("Vidhi Hemal Doshi · Rishikesh Pechetti")
nav.run()
