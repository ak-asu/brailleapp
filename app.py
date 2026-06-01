import numpy as np
import cv2
import streamlit as st
import psutil

from components.camera_component import camera_component
from pipeline.preprocessor import preprocess
from pipeline.detector import load_onnx_session, detect_hough, detect_onnx, ensemble
from pipeline.grid import reconstruct_grid, _cluster_1d
from pipeline.corrector import estimate_skew, correct_skew
from pipeline.translator import translate
from pipeline.runtime import (
    should_run_onnx,
    should_hold_for_blur,
    with_guidance,
    is_new_frame,
)
from utils.image_utils import (
    decode_frame,
    encode_frame,
    letterbox,
    draw_dots,
    map_dots_from_letterbox,
)
from utils.quality import blur_score, brightness, dot_density, get_guidance

MODEL_PATH = "models/yolov8n_braille_combined.onnx"
_COUNTER_KEY = "frame_counter"
_PROCESSING_KEY = "processing_frame"
_LAST_TS_KEY = "last_frame_ts"

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Braille Scanner",
    page_icon="⠃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Load shared resources once per process ───────────────────────────────────
@st.cache_resource
def _load_model():
    return load_onnx_session(MODEL_PATH)

onnx_session = _load_model()

# ─── Sidebar settings ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    grade = st.selectbox(
        "Braille Grade",
        options=["grade2", "grade1", "nemeth", "computer"],
        format_func=lambda x: {
            "grade2": "Grade 2 (UEB contracted)",
            "grade1": "Grade 1 (uncontracted)",
            "nemeth": "Nemeth (math)",
            "computer": "Computer (8-dot)",
        }[x],
        index=0,
    )
    auto_speak    = st.checkbox("Auto-speak translation", value=True)
    speak_guide   = st.checkbox("Speak camera guidance", value=False)
    high_contrast = st.checkbox("High contrast", value=False)
    large_text    = st.checkbox("Large text", value=False)

# ─── CSS injection ────────────────────────────────────────────────────────────
_sz = "1.4em" if large_text else "1em"
_hc = """
html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    background-color: #000 !important;
}
header[data-testid="stHeader"] {
    background-color: #000 !important;
    border-bottom: 1px solid #333 !important;
}
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div > div {
    background-color: #0d0d0d !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #e8e8e8 !important;
}
section[data-testid="stMain"] > div,
.block-container {
    background-color: #000 !important;
    color: #fff !important;
}
section[data-testid="stMain"] h1,
section[data-testid="stMain"] h2,
section[data-testid="stMain"] h3,
section[data-testid="stMain"] p,
section[data-testid="stMain"] label,
section[data-testid="stMain"] span {
    color: #e8e8e8 !important;
}
button[data-testid^="stBaseButton"] {
    background-color: #1e1e1e !important;
    color: #fff !important;
    border: 1px solid #444 !important;
}
button[data-testid^="stBaseButton"]:hover {
    background-color: #333 !important;
}
.stTextArea textarea {
    background-color: #111 !important;
    color: #fff !important;
    border-color: #444 !important;
}
[data-testid="stFileUploaderDropzone"] {
    background-color: #111 !important;
    border-color: #555 !important;
}
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span {
    color: #aaa !important;
}
[data-testid="stSelectbox"] > div > div {
    background-color: #1e1e1e !important;
    color: #fff !important;
}
.stProgress > div > div { background-color: #222 !important; }
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] { color: #fff !important; }
""" if high_contrast else ""

st.markdown(f"""<style>
.stTextArea textarea {{ font-size: {_sz}; font-family: monospace; }}
{_hc}
</style>""", unsafe_allow_html=True)

# ─── Title ────────────────────────────────────────────────────────────────────
st.title("⠃ Braille Scanner")
if onnx_session is None:
    st.caption("Hough-only mode (ONNX model not found — place yolov8n_braille_combined.onnx in models/)")

# ─── Session state init ───────────────────────────────────────────────────────
if "last_result" not in st.session_state:
    st.session_state.last_result = {
        "text": "",
        "confidence": 0.0,
        "guidance": {"status": "ok", "message": "Ready — point camera at Braille"},
    }
if _COUNTER_KEY not in st.session_state:
    st.session_state[_COUNTER_KEY] = 0
if _PROCESSING_KEY not in st.session_state:
    st.session_state[_PROCESSING_KEY] = False
if _LAST_TS_KEY not in st.session_state:
    st.session_state[_LAST_TS_KEY] = None


# ─── Pipeline helper ──────────────────────────────────────────────────────────
def run_pipeline(raw_bgr: np.ndarray, counter: int, selected_grade: str) -> dict:
    """Full pipeline on a raw BGR frame. Returns result dict."""
    # Downscale if memory is tight (free-tier safeguard)
    mem = psutil.virtual_memory()
    target_size = 480 if mem.used > 700 * 1024 * 1024 else 640

    frame_s, scale, (pad_x, pad_y) = letterbox(raw_bgr, target_size)

    # Dynamic CLAHE based on brightness
    gray_quick = cv2.cvtColor(frame_s, cv2.COLOR_BGR2GRAY)
    bright = brightness(gray_quick)
    clip = 3.5 if bright < 40 else (1.0 if bright > 210 else 2.0)

    binary, gray = preprocess(frame_s, clip_limit=clip)

    # Stage 2A: Hough every cycle
    hough_raw = detect_hough(binary, gray)

    # Stage 2B: ONNX every 3rd cycle
    if should_run_onnx(onnx_session, counter, target_size):
        onnx_raw = detect_onnx(frame_s, onnx_session)
        confirmed = ensemble(hough_raw, onnx_raw)
    else:
        confirmed = [(x, y, 0.45) for x, y, r in hough_raw]

    # Quality signals
    b_score = blur_score(gray)
    density = dot_density(confirmed, target_size, target_size)
    skew = 0.0

    # Skew correction if enough dots
    if len(confirmed) >= 4:
        ys = np.array([d[1] for d in confirmed])
        row_labels = _cluster_1d(ys, gap=20.0)
        skew = estimate_skew(confirmed, row_labels)
        if abs(skew) > 5.0:
            frame_s = correct_skew(frame_s, skew)
            binary, gray = preprocess(frame_s, clip_limit=clip)
            hough_raw = detect_hough(binary, gray)
            if should_run_onnx(onnx_session, counter, target_size):
                onnx_raw = detect_onnx(frame_s, onnx_session)
                confirmed = ensemble(hough_raw, onnx_raw)
            else:
                confirmed = [(x, y, 0.45) for x, y, r in hough_raw]

    guidance = get_guidance(b_score, bright, density, skew)

    if should_hold_for_blur(b_score):
        return with_guidance(st.session_state.last_result, guidance)

    # Draw overlays on the original camera frame so the JS overlay aspect ratio
    # matches the live video instead of a square letterboxed processing frame.
    draw_list = map_dots_from_letterbox(
        [(x, y, 5.0) for x, y, _ in confirmed],
        scale=scale,
        pad=(pad_x, pad_y),
        original_shape=raw_bgr.shape,
    )
    annotated = draw_dots(raw_bgr, draw_list)
    annotated_b64 = encode_frame(annotated)

    # Grid + translation
    braille_str, conf = reconstruct_grid(confirmed)
    if braille_str:
        text = translate(braille_str, selected_grade)
    else:
        text = st.session_state.last_result.get("text", "")

    return {
        "annotated_frame": annotated_b64,
        "text": text,
        "confidence": float(conf),  # numpy.float64 is not JSON-serialisable
        "guidance": guidance,
    }


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_live, tab_upload = st.tabs(["📷 Live Camera", "📁 Upload Image"])

# ── Live Camera tab ───────────────────────────────────────────────────────────
with tab_live:

    @st.fragment(run_every="500ms")
    def live_fragment():
        cam_result = camera_component(
            data={
                "auto_speak": auto_speak,
                "speak_guidance": speak_guide,
                "result": st.session_state.last_result,
            },
            key="camera",
        )
        frame_data = getattr(cam_result, "frame", None)

        if (
            frame_data
            and isinstance(frame_data, dict)
            and "frame" in frame_data
            and not st.session_state[_PROCESSING_KEY]
            and is_new_frame(frame_data, st.session_state[_LAST_TS_KEY])
        ):
            raw = decode_frame(frame_data["frame"])
            if raw is not None:
                st.session_state[_PROCESSING_KEY] = True
                try:
                    st.session_state[_COUNTER_KEY] += 1
                    st.session_state[_LAST_TS_KEY] = frame_data.get("ts")
                    result = run_pipeline(raw, st.session_state[_COUNTER_KEY], grade)
                    st.session_state.last_result = result
                finally:
                    st.session_state[_PROCESSING_KEY] = False

        result = st.session_state.last_result
        guidance_msg = result["guidance"]["message"]
        st.markdown(f"**{guidance_msg}**")
        conf_pct = int(result.get("confidence", 0) * 100)
        st.progress(conf_pct, text=f"Confidence: {conf_pct}%")

        text_val = result.get("text", "")
        st.text_area("Translation", value=text_val, height=100, key="live_text_area")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Copy text", key="btn_copy_live"):
                st.toast("Use Ctrl+A / Ctrl+C in the text box above to copy")
        with col_b:
            if st.button("▶ Speak", key="btn_speak_live") and text_val:
                escaped = text_val.replace("'", "\\'").replace('"', '\\"')
                st.markdown(
                    f"<script>window.speechSynthesis.cancel();"
                    f"window.speechSynthesis.speak("
                    f"new SpeechSynthesisUtterance('{escaped}'));</script>",
                    unsafe_allow_html=True,
                )

    live_fragment()

# ── Upload tab ────────────────────────────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader(
        "Upload a Braille image (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        raw = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if raw is not None:
            result = run_pipeline(raw, counter=0, selected_grade=grade)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(
                    f"data:image/jpeg;base64,{result['annotated_frame']}",
                    caption="Detected dots (green circles)",
                    use_container_width=True,
                )
            with col2:
                st.metric("Confidence", f"{int(result['confidence'] * 100)}%")
                st.text_area("Translation", value=result["text"], height=150)

                if st.button("▶ Speak", key="btn_speak_upload") and result["text"]:
                    escaped = result["text"].replace("'", "\\'").replace('"', '\\"')
                    st.markdown(
                        f"<script>window.speechSynthesis.cancel();"
                        f"window.speechSynthesis.speak("
                        f"new SpeechSynthesisUtterance('{escaped}'));</script>",
                        unsafe_allow_html=True,
                    )

                if st.button("⬇ Download Audio (MP3)", key="btn_audio") and result["text"]:
                    import asyncio
                    import tempfile
                    import os
                    import edge_tts

                    async def _synth(text: str) -> bytes:
                        communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                            fname = f.name
                        await communicate.save(fname)
                        with open(fname, "rb") as f:
                            data = f.read()
                        os.unlink(fname)
                        return data

                    with st.spinner("Generating audio…"):
                        audio_bytes = asyncio.run(_synth(result["text"]))
                    st.download_button(
                        "Save MP3",
                        data=audio_bytes,
                        file_name="braille_translation.mp3",
                        mime="audio/mpeg",
                    )
        else:
            st.error("Could not decode image. Please upload a valid JPG or PNG.")
