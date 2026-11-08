"""IoT Attack Detector — demo app.

Run from the project folder:
    python app/build_demo_assets.py      # once, ~3 minutes
    streamlit run app/app.py
"""
import streamlit as st

st.set_page_config(page_title="IoT Attack Detector", page_icon=":material/shield:", layout="wide")

pages = [
    st.Page("views/detect.py", title="Detect attacks", icon=":material/search:", default=True),
    st.Page("views/how_it_works.py", title="How it works", icon=":material/info:"),
    st.Page("views/try_to_fool_it.py", title="Try to fool it", icon=":material/science:"),
]
nav = st.navigation(pages)
with st.sidebar:
    st.caption("IS 665 Cybersecurity Analytics · Group 11")
    st.caption("Vidhi Hemal Doshi · Rishikesh Pechetti")
nav.run()
