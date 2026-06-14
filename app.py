import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile, os
from database import init_db, save_detection, fetch_all_detections, get_stats
from detector import detect_defects
 
# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Metal Defect Detector",
    page_icon="🔩",
    layout="wide"
)
 
# ── Initialize DB on startup ──────────────────────────────────
init_db()
 
# ── Sidebar controls ──────────────────────────────────────────
st.sidebar.title("⚙️ Controls")
video_source = st.sidebar.radio(
    "Video Source",
    ["Upload a video file", "Use webcam (camera 0)"]
)
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
save_to_db = st.sidebar.checkbox("Save detections to database", value=True)
skip_frames = st.sidebar.slider(
    "Process every N frames", 1, 10, 2,
    help="Higher = faster but less accurate"
)
 
# ── Main layout ───────────────────────────────────────────────
st.title("🔩 Metal Plate Defect Detector")
st.markdown("Real-time defect detection on conveyor belt video stream")
 
col1, col2 = st.columns([2, 1])
 
with col1:
    st.subheader("📹 Live Video Stream")
    video_placeholder = st.empty()
    status_placeholder = st.empty()
 
with col2:
    st.subheader("📊 Live Stats")
    stat_total = st.empty()           # FIX: use empty() so it can be updated
    stat_total.metric("Total Defects Detected", 0)
    stat_frame = st.empty()
    st.divider()
    st.subheader("🕐 Recent Detections")
    recent_placeholder = st.empty()
 
# ── Video source setup ────────────────────────────────────────
uploaded_file = None
if video_source == "Upload a video file":
    uploaded_file = st.sidebar.file_uploader(
        "Upload video", type=["mp4", "avi", "mov"]
    )
 
# ── Start / Stop buttons ──────────────────────────────────────
col_btn1, col_btn2 = st.columns(2)
start_btn = col_btn1.button("▶️ Start Detection", type="primary", use_container_width=True)
stop_btn  = col_btn2.button("⏹️ Stop",            use_container_width=True)
 
if "running" not in st.session_state:
    st.session_state.running = False
if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False
 
# ── Main detection loop ───────────────────────────────────────
if st.session_state.running:
 
    # Open video source
    cap = None
    if video_source == "Use webcam (camera 0)":
        cap = cv2.VideoCapture(0)
 
    elif uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()
        tfile.close()
        cap = cv2.VideoCapture(tfile.name)
 
    else:
        st.warning("⚠️ Please upload a video file first!")
        st.session_state.running = False
 
    if cap is not None and cap.isOpened():
        frame_count   = 0
        total_defects = 0
        recent_rows   = []
 
        status_placeholder.success("🟢 Detection running...")
 
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                status_placeholder.info("✅ Video ended.")
                st.session_state.running = False
                break
 
            frame_count += 1
 
            # Skip frames for performance
            if frame_count % skip_frames != 0:
                continue
 
            # FIX: initialize defaults so variables always exist
            annotated, detections = frame.copy(), []
 
            try:
                annotated, detections = detect_defects(frame, confidence)
            except Exception as e:
                st.error(f"Detection Error: {str(e)}")
                break   # FIX: break is now correctly INSIDE except
 
            # Save detections to DB
            if save_to_db:
                for d in detections:
                    save_detection(
                        frame_count,
                        d["defect_type"], d["confidence"],
                        d["x1"], d["y1"], d["x2"], d["y2"]
                    )
                    recent_rows.append({
                        "Frame": frame_count,
                        "Type":  d["defect_type"],
                        "Conf":  f"{d['confidence']:.2f}"
                    })
 
            total_defects += len(detections)
 
            # Resize → convert BGR→RGB → display
            annotated = cv2.resize(annotated, (960, 540))
            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(
                rgb_frame,
                channels="RGB",
                use_container_width=True
            )
 
            # FIX: update stat_total properly via .metric() on the empty placeholder
            stat_total.metric("Total Defects Detected", total_defects)
            stat_frame.metric("Frame", frame_count)
 
            if recent_rows:
                recent_placeholder.dataframe(
                    pd.DataFrame(recent_rows[-10:]),
                    use_container_width=True,
                    hide_index=True
                )
 
        cap.release()
 
        # Cleanup temp file if used
        if video_source == "Upload a video file" and uploaded_file is not None:
            try:
                os.unlink(tfile.name)
            except Exception:
                pass
 
    elif cap is not None:
        st.error("❌ Could not open video source.")
 
# ── Database viewer (always visible below) ────────────────────
st.divider()
st.subheader("🗃️ Detection Database")
 
col_db1, col_db2 = st.columns([3, 1])
with col_db1:
    if st.button("🔄 Refresh Database View"):
        pass  # triggers rerun automatically
 
with col_db2:
    total, by_type = get_stats()
    st.metric("Total Records", total)
 
all_data = fetch_all_detections()
if all_data:
    df = pd.DataFrame(all_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
 
    csv = df.to_csv(index=False)
    st.download_button(
        "⬇️ Download CSV",
        csv,
        "defect_detections.csv",
        "text/csv"
    )
else:
    st.info("No detections in database yet. Start the detector above!")