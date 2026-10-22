import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="AI Lake Enforcement System",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 AI-Based Lake Enforcement System")
st.caption(
    "Computer-vision based monitoring of lakes for human activity, "
    "water discoloration, and possible pollution indicators"
)

# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Enforcement Settings")

person_conf = st.sidebar.slider(
    "Human Detection Sensitivity",
    min_value=0.1,
    max_value=0.9,
    value=0.4,
    step=0.05
)

pollution_sensitivity = st.sidebar.slider(
    "Pollution Detection Sensitivity",
    min_value=1,
    max_value=10,
    value=5
)

# ---------------- Load HOG people detector ----------------
@st.cache_resource
def load_person_detector():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog

hog = load_person_detector()

# ---------------- Upload image ----------------
uploaded_file = st.file_uploader(
    "Upload a lake image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Resize large images for faster processing
    max_width = 900
    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        image = cv2.resize(image, (max_width, int(image.shape[0] * scale)))

    original_image = image.copy()

    alerts = []

    # ---------------- 1. Human activity detection ----------------
    resized = cv2.resize(image, (640, 480))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    boxes, weights = hog.detectMultiScale(
        gray,
        winStride=(8, 8),
        padding=(4, 4),
        scale=1.05
    )

    person_count = 0

    for (x, y, w, h), weight in zip(boxes, weights):
        if weight >= person_conf:
            person_count += 1

            # Convert coordinates back to original image size
            scale_x = image.shape[1] / 640
            scale_y = image.shape[0] / 480

            x1 = int(x * scale_x)
            y1 = int(y * scale_y)
            x2 = int((x + w) * scale_x)
            y2 = int((y + h) * scale_y)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                image,
                "Person",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    if person_count > 0:
        alerts.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Violation": "Human activity detected",
            "Severity": "Medium" if person_count > 2 else "Low",
            "Details": f"{person_count} person(s) detected near lake area",
            "Recommended Action": "Monitor activity for safety or unauthorized behavior"
        })

    # ---------------- 2. Water pollution indicator ----------------
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Green algae-like color range
    lower_green = np.array([35, 60, 60])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    green_pixels = cv2.countNonZero(green_mask)
    total_pixels = image.shape[0] * image.shape[1]
    green_percentage = (green_pixels / total_pixels) * 100

    algae_threshold = pollution_sensitivity * 2

    if green_percentage > algae_threshold:
        cv2.putText(
            image,
            "Possible Algae / Pollution",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 0, 255),
            3
        )

        alerts.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Violation": "Possible algae growth / water pollution",
            "Severity": "High",
            "Details": f"Green region coverage: {green_percentage:.2f}%",
            "Recommended Action": "Send inspection team and collect water sample"
        })

    # ---------------- 3. Dark / murky water indicator ----------------
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    darkness = 100 - np.mean(gray_full)

    if darkness > 55:
        alerts.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Violation": "Murky or low-visibility water detected",
            "Severity": "Medium",
            "Details": f"Water clarity indicator: {darkness:.1f}",
            "Recommended Action": "Check turbidity and possible contamination"
        })

    # ---------------- Show results ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Uploaded Lake Image")
        st.image(
            cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

    with col2:
        st.subheader("🤖 AI Analysis Result")
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

    # ---------------- Metrics ----------------
    st.subheader("📊 Lake Monitoring Metrics")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("People Detected", person_count)
    m2.metric("Green Coverage", f"{green_percentage:.2f}%")
    m3.metric("Water Clarity Score", f"{darkness:.1f}")
    m4.metric("Alerts Generated", len(alerts))

    # ---------------- Alerts ----------------
    st.subheader("🚨 Lake Enforcement Alerts")

    if alerts:
        alert_df = pd.DataFrame(alerts)
        st.dataframe(alert_df, use_container_width=True)

        csv_data = alert_df.to_csv(index=False).encode()

        st.download_button(
            label="⬇️ Download Enforcement Report",
            data=csv_data,
            file_name=f"lake_enforcement_report_{int(time.time())}.csv",
            mime="text/csv"
        )
    else:
        st.success("No major enforcement issue detected in this image.")

else:
    st.info("Upload a lake image to begin AI-based enforcement analysis.")

# ---------------- Footer ----------------
st.markdown("---")
st.caption(
    "Demo system using OpenCV computer vision. "
    "In a full deployment, this can be connected to drone cameras, CCTV, "
    "GPS, and a municipal lake-monitoring dashboard."
)
