import time
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="AI Lake Enforcement System",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 AI-Based Lake Enforcement System")
st.caption(
    "AI-powered monitoring of lakes for littering, unauthorized boats, "
    "and suspicious activity using YOLOv8 object detection"
)

# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Enforcement Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.95,
    value=0.40,
    step=0.05
)

show_all_objects = st.sidebar.checkbox(
    "Show all detected objects",
    value=False
)

# ---------------- Enforcement rules ----------------
ENFORCEMENT_RULES = {
    "bottle": {
        "alert": "Possible plastic littering detected",
        "severity": "High",
        "action": "Verify waste and initiate cleanup / littering notice"
    },
    "cup": {
        "alert": "Possible disposable waste detected",
        "severity": "Medium",
        "action": "Verify waste and initiate cleanup"
    },
    "boat": {
        "alert": "Boat activity detected in lake area",
        "severity": "Medium",
        "action": "Verify boat permit and activity"
    },
    "person": {
        "alert": "Human activity detected near lake",
        "severity": "Low",
        "action": "Monitor activity for safety or violation"
    },
    "backpack": {
        "alert": "Person carrying baggage near lake",
        "severity": "Low",
        "action": "Monitor for dumping or unauthorized activity"
    }
}

# ---------------- Load YOLOv8 ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------- Upload image ----------------
uploaded_file = st.file_uploader(
    "Upload a lake image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Analyzing lake image using YOLOv8..."):
        results = model.predict(image, conf=confidence, verbose=False)[0]

    annotated_image = results.plot()[:, :, ::-1]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Uploaded Lake Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🤖 AI Detection Result")
        st.image(annotated_image, use_container_width=True)

    # ---------------- Create alerts ----------------
    alerts = []
    all_detections = []

    for box in results.boxes:
        class_name = model.names[int(box.cls[0])]
        conf = round(float(box.conf[0]), 2)

        all_detections.append({
            "Detected Object": class_name,
            "Confidence": conf
        })

        if class_name in ENFORCEMENT_RULES:
            rule = ENFORCEMENT_RULES[class_name]

            alerts.append({
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Detected Object": class_name,
                "Confidence": conf,
                "Enforcement Alert": rule["alert"],
                "Severity": rule["severity"],
                "Recommended Action": rule["action"]
            })

    # ---------------- Show alerts ----------------
    st.subheader("🚨 Lake Enforcement Alerts")

    if alerts:
        alert_df = pd.DataFrame(alerts)

        st.dataframe(alert_df, use_container_width=True)

        high_count = (alert_df["Severity"] == "High").sum()
        medium_count = (alert_df["Severity"] == "Medium").sum()
        low_count = (alert_df["Severity"] == "Low").sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 High Severity", high_count)
        c2.metric("🟠 Medium Severity", medium_count)
        c3.metric("🟡 Low Severity", low_count)

        csv_data = alert_df.to_csv(index=False).encode()

        st.download_button(
            label="⬇️ Download Enforcement Report",
            data=csv_data,
            file_name=f"lake_enforcement_report_{int(time.time())}.csv",
            mime="text/csv"
        )

    else:
        st.success("No enforcement-relevant violation detected in this image.")

    # ---------------- Show all detections ----------------
    if show_all_objects and all_detections:
        st.subheader("🔍 All Objects Detected by YOLOv8")
        st.dataframe(pd.DataFrame(all_detections), use_container_width=True)

else:
    st.info("Upload a lake image to begin AI-based enforcement analysis.")

# ---------------- Footer ----------------
st.markdown("---")
st.caption(
    "Demo system using YOLOv8 object detection. "
    "In a full deployment, this can be connected to drone cameras, CCTV, "
    "GPS, and a municipal lake-monitoring dashboard."
)
