import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os
import base64

# import your modules
from prepare_dataset import DatasetPreparer
from live_detection import LivePotholeDetector
from train_yolov8 import PotholeTrainer
from valid import ModelValidator

# -------------------------
# Config / audio setup
# -------------------------
ALERT_SOUND = "audio-editor-output.wav"  # must be in same folder
AUDIO_BASE64 = None
if os.path.exists(ALERT_SOUND):
    with open(ALERT_SOUND, "rb") as f:
        AUDIO_BASE64 = base64.b64encode(f.read()).decode()

# -------------------------
# Session state
# -------------------------
if "last_detection_time" not in st.session_state:
    st.session_state["last_detection_time"] = 0.0
if "alert_playing" not in st.session_state:
    st.session_state["alert_playing"] = False

audio_slot = st.empty()

# -------------------------
# Audio functions
# -------------------------
def start_alert():
    """Start continuous alert (HTML <audio autoplay loop>)."""
    if AUDIO_BASE64:
        audio_html = f"""
        <audio autoplay loop>
            <source src="data:audio/wav;base64,{AUDIO_BASE64}" type="audio/wav">
        </audio>
        """
        audio_slot.markdown(audio_html, unsafe_allow_html=True)
        st.session_state["alert_playing"] = True

def stop_alert():
    """Stop alert by clearing HTML slot."""
    audio_slot.empty()
    st.session_state["alert_playing"] = False

def reset_alert_state():
    """Reset state so audio can restart after stopping/restarting detection."""
    stop_alert()
    st.session_state["last_detection_time"] = 0.0
    st.session_state["alert_playing"] = False

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Pothole Detection", page_icon="🚧")
st.title("🚧 Pothole Detection Dashboard")

st.sidebar.header("Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)

detector = LivePotholeDetector()

# -------------------------
# 1) Image detection
# -------------------------
st.header("1️⃣ Detect Potholes in Image")
uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_img:
    reset_alert_state()

    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    detections = detector.detect(img, conf_threshold)
    result = detector.draw_detections(img, detections)

    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Detection Result", use_column_width=True)
    st.write(f"✅ Detections: {len(detections)}")

    if len(detections) > 0 and not st.session_state["alert_playing"]:
        start_alert()
    elif len(detections) == 0 and st.session_state["alert_playing"]:
        stop_alert()

# -------------------------
# 2) Video detection
# -------------------------
st.header("2️⃣ Detect Potholes in Video")
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if uploaded_video:
    reset_alert_state()

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    info_box = st.empty()

    frame_count, start_time = 0, time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        detections = detector.detect(frame, conf_threshold)
        result = detector.draw_detections(frame, detections)

        fps = frame_count / max(1e-6, (time.time() - start_time))
        stframe.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), channels="RGB")
        info_box.markdown(
            f"**FPS:** {fps:.1f} | **Detections:** {len(detections)} | **Conf:** {conf_threshold:.2f} | **Device:** {'GPU' if getattr(detector, 'device', 'cpu') == '0' else 'CPU'}"
        )

        now = time.time()
        if len(detections) > 0:
            st.session_state["last_detection_time"] = now
            if not st.session_state["alert_playing"]:
                start_alert()
        else:
            if now - st.session_state["last_detection_time"] >= 3 and st.session_state["alert_playing"]:
                stop_alert()

        time.sleep(0.01)

    cap.release()
    reset_alert_state()
    st.success("✅ Video processing finished")

# -------------------------
# 3) Live camera detection
# -------------------------
st.header("3️⃣ Live Camera Detection (Webcam Stream)")
run_live = st.checkbox("Start Live Detection")
if run_live:
    reset_alert_state()

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while run_live and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame, conf_threshold)
        result = detector.draw_detections(frame, detections)
        stframe.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), channels="RGB")

        now = time.time()
        if len(detections) > 0:
            st.session_state["last_detection_time"] = now
            if not st.session_state["alert_playing"]:
                start_alert()
        else:
            if now - st.session_state["last_detection_time"] >= 3 and st.session_state["alert_playing"]:
                stop_alert()

        time.sleep(0.01)

    cap.release()
    reset_alert_state()
