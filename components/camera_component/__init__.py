import os
import streamlit as st

_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_dir, "camera_v2.html"), encoding="utf-8") as f:
    _html = f.read()
with open(os.path.join(_dir, "camera_v2.js"), encoding="utf-8") as f:
    _js = f.read()

camera_component = st.components.v2.component(
    "camera_component",
    html=_html,
    js=_js,
)
