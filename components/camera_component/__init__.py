import os
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Declares the component, pointing Streamlit at the directory containing index.html.
# Streamlit serves index.html from this directory over HTTPS (Community Cloud).
camera_component = components.declare_component(
    "camera_component",
    path=_COMPONENT_DIR,
)
