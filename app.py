import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="Advanced AI Lake Enforcement System",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Advanced AI Lake Enforcement System")
st.caption(
    "Lake segmentation, pollution analysis, encroachment detection, "
    "IoT anomaly detection, and enforcement workflow"
)

if "alerts" not in st.session_state:
    st.session_state.alerts = []

if "cases" not in st.session_state:
    st.session_state.cases = []

# ---------------- Sidebar ----------------
st.sidebar.header("🏞️ Lake Profile")

lake_name = st.sidebar.text_input("Lake Name", value="Sample Lake")
location = st.sidebar.text_input("Location", value="Bengaluru, India")
latitude = st.sidebar.number_input("Latitude", value=12.9716, format="%.4f")
longitude = st.sidebar.number_input("Longitude", value=77.5946, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ AI Configuration")

input_source = st.sidebar.selectbox(
    "Monitoring Source",
    ["Drone Image", "CCTV Camera", "Satellite Image"]
)

algae_threshold = st.sidebar.slider("Algae Threshold (%)", 1, 50, 10)
waste_threshold = st.sidebar.slider("Waste Threshold (%)", 1, 50, 8)
encroachment_threshold = st.sidebar.slider("Encroachment Threshold (%)", 1, 50, 10)

st.sidebar.markdown("---")
st.sidebar.header("📡 IoT Sensor Inputs")

ph_value = st.sidebar.slider("pH Level", 0.0, 14.0, 7.2, 0.1)
turbidity = st.sidebar.slider("Turbidity (NTU)", 0, 100, 25)
dissolved_oxygen = st.sidebar.slider(
    "Dissolved Oxygen (mg/L)", 0.0, 15.0, 6.5, 0.1
)
temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 27.0, 0.5)
water_level = st.sidebar.slider("Water Level (%)", 0, 100, 78)

# ---------------- Functions ----------------
def resize_image(image, max_size=900):
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size))
    return image


def extract_lake_mask(image):
    img = np.array(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower = np.array([35, 30, 30])
    upper = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    lake_mask = np.zeros_like(mask)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(
            lake_mask,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

    return lake_mask


def analyze_lake(image, lake_mask):
    img = np.array(image)
    lake_pixels = np.count_nonzero(lake_mask)

    if lake_pixels == 0:
        return {
            "algae": 0.0,
            "waste": 0.0,
            "murky": 0.0,
            "lake_area": 0
        }

    r = img[:, :, 0].astype(int)
    g = img[:, :, 1].astype(int)
    b = img[:, :, 2].astype(int)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    algae_mask = (g > r + 20) & (g > b + 20) & (lake_mask > 0)
    waste_mask = (gray < 70) & (lake_mask > 0)
    murky_mask = (r > b + 25) & (g > b + 15) & (lake_mask > 0)

    return {
        "algae": (np.sum(algae_mask) / lake_pixels) * 100,
        "waste": (np.sum(waste_mask) / lake_pixels) * 100,
        "murky": (np.sum(murky_mask) / lake_pixels) * 100,
        "lake_area": lake_pixels
    }


def detect_encroachment(reference_image, current_image):
    reference_mask = extract_lake_mask(reference_image)
    current_mask = extract_lake_mask(current_image)

    reference_area = np.count_nonzero(reference_mask)
    current_area = np.count_nonzero(current_mask)

    if reference_area == 0:
        return 0.0, False

    reduction = ((reference_area - current_area) / reference_area) * 100
    reduction = max(reduction, 0.0)

    return reduction, reduction > 0


def calculate_sensor_anomaly(ph, turbidity, dissolved_oxygen, temperature):
    normal_data = np.array([
        [7.2, 20, 7.5, 26],
        [7.0, 25, 7.0, 27],
        [7.4, 30, 6.8, 28],
        [6.8, 35, 6.5, 29],
        [7.1, 28, 7.2, 26]
    ])

    current_data = np.array([
        [ph, turbidity, dissolved_oxygen, temperature]
    ])

    model = IsolationForest(
        contamination=0.20,
        random_state=42
    )

    model.fit(normal_data)
    anomaly_score = model.decision_function(current_data)[0]

    return max(0.0, round((0.5 - anomaly_score) * 100, 1))

# ---------------- Image Upload ----------------
st.header("📤 Upload Lake Monitoring Images")

col_ref, col_current = st.columns(2)

with col_ref:
    reference_file = st.file_uploader(
        "Reference Lake Image",
        type=["jpg", "jpeg", "png"],
        key="reference"
    )

with col_current:
    current_file = st.file_uploader(
        "Current Lake Image",
        type=["jpg", "jpeg", "png"],
        key="current"
    )

if current_file is not None:

    current_image = resize_image(
        Image.open(current_file).convert("RGB")
    )

    current_array = np.array(current_image)
    lake_mask = extract_lake_mask(current_image)
    metrics = analyze_lake(current_image, lake_mask)

    reference_image = None
    encroachment_percent = 0.0
    encroachment_detected = False

    if reference_file is not None:
        reference_image = resize_image(
            Image.open(reference_file).convert("RGB")
        )

        encroachment_percent, encroachment_detected = detect_encroachment(
            reference_image,
            current_image
        )

    sensor_score = calculate_sensor_anomaly(
        ph_value,
        turbidity,
        dissolved_oxygen,
        temperature
    )

    # ---------------- Alerts ----------------
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alerts = []

    if metrics["algae"] > algae_threshold:
        alerts.append({
            "Alert ID": f"LK-{int(time.time())}-A",
            "Time": current_time,
            "Lake": lake_name,
            "Source": input_source,
            "Violation": "Algae Bloom Detected",
            "Severity": "High",
            "Detected Value": f"{metrics['algae']:.2f}% algae coverage",
            "Recommended Action": "Collect water sample and inspect nutrient inflow",
            "Status": "Pending"
        })

    if metrics["waste"] > waste_threshold:
        alerts.append({
            "Alert ID": f"LK-{int(time.time())}-W",
            "Time": current_time,
            "Lake": lake_name,
            "Source": input_source,
            "Violation": "Floating Waste Detected",
            "Severity": "High",
            "Detected Value": f"{metrics['waste']:.2f}% suspected waste coverage",
            "Recommended Action": "Deploy cleanup team and identify waste source",
            "Status": "Pending"
        })

    if metrics["murky"] > 30:
        alerts.append({
            "Alert ID": f"LK-{int(time.time())}-M",
            "Time": current_time,
            "Lake": lake_name,
            "Source": input_source,
            "Violation": "Sediment / Muddy Water Detected",
            "Severity": "Medium",
            "Detected Value": f"{metrics['murky']:.2f}% muddy coverage",
            "Recommended Action": "Inspect nearby construction and soil erosion",
            "Status": "Pending"
        })

    if encroachment_detected and encroachment_percent > encroachment_threshold:
        alerts.append({
            "Alert ID": f"LK-{int(time.time())}-E",
            "Time": current_time,
            "Lake": lake_name,
            "Source": "Change Detection",
            "Violation": "Possible Lake Encroachment",
            "Severity": "High",
            "Detected Value": f"{encroachment_percent:.2f}% water-area reduction",
            "Recommended Action": "Verify land records and conduct field survey",
            "Status": "Pending"
        })

    if sensor_score > 50:
        alerts.append({
            "Alert ID": f"LK-{int(time.time())}-S",
            "Time": current_time,
            "Lake": lake_name,
            "Source": "IoT Anomaly Detection",
            "Violation": "Abnormal Water-Quality Pattern",
            "Severity": "High",
            "Detected Value": f"Anomaly score: {sensor_score}",
            "Recommended Action": "Validate sensors and investigate pollution source",
            "Status": "Pending"
        })

    if ph_value < 6.5 or ph_value > 8.5:
        alerts.append({
            "Alert ID": f"LK-{int(time.time())}-PH",
            "Time": current_time,
            "Lake": lake_name,
            "Source": "IoT Sensor",
            "Violation": "Abnormal pH Level",
            "Severity": "Medium",
            "Detected Value": f"pH {ph_value}",
            "Recommended Action": "Validate sensor and inspect lake",
            "Status": "Pending"
        })

    if turbidity > 50:
        alerts.append({
            "Alert ID": f"LK-{int(time.time())}-T",
            "Time": current_time,
            "Lake": lake_name,
            "Source": "IoT Sensor",
            "Violation": "High Turbidity",
            "Severity": "Medium",
            "Detected Value": f"{turbidity} NTU",
            "Recommended Action": "Check sediment inflow and pollution source",
            "Status": "Pending"
        })

    if dissolved_oxygen < 4:
        alerts.append({
            "Alert ID": f"LK-{int(time.time())}-DO",
            "Time": current_time,
            "Lake": lake_name,
            "Source": "IoT Sensor",
            "Violation": "Low Dissolved Oxygen",
            "Severity": "High",
            "Detected Value": f"{dissolved_oxygen} mg/L",
            "Recommended Action": "Check aquatic health and pollution source",
            "Status": "Pending"
        })

    if not alerts:
        alerts.append({
            "Alert ID": f"LK-{int(time.time())}-N",
            "Time": current_time,
            "Lake": lake_name,
            "Source": input_source,
            "Violation": "No Major Violation Detected",
            "Severity": "Low",
            "Detected Value": "Lake appears normal",
            "Recommended Action": "Continue routine monitoring",
            "Status": "Pending"
        })

    st.session_state.alerts.extend(alerts)

    # ---------------- Risk Score ----------------
    waste_score = min(metrics["waste"] * 2, 100)
    algae_score = min(metrics["algae"] * 2, 100)
    encroachment_score = min(encroachment_percent * 3, 100)

    risk_score = min(
        round(
            (0.30 * waste_score) +
            (0.25 * algae_score) +
            (0.20 * encroachment_score) +
            (0.15 * sensor_score) +
            (0.10 * 0),
            1
        ),
        100
    )

    if risk_score >= 70:
        risk_level = "HIGH RISK"
    elif risk_score >= 40:
        risk_level = "MEDIUM RISK"
    else:
        risk_level = "LOW RISK"

    # ---------------- Dashboard ----------------
    st.header("📊 Advanced Lake Monitoring Dashboard")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("AI Risk Score", f"{risk_score}/100", risk_level)
    m2.metric("Algae Coverage", f"{metrics['algae']:.2f}%")
    m3.metric("Waste Indicator", f"{metrics['waste']:.2f}%")
    m4.metric("Active Alerts", len(alerts))

    m5, m6, m7, m8 = st.columns(4)

    m5.metric("Lake Area Pixels", metrics["lake_area"])
    m6.metric("Encroachment Change", f"{encroachment_percent:.2f}%")
    m7.metric("IoT Anomaly Score", sensor_score)
    m8.metric("Turbidity", f"{turbidity} NTU")

    # ---------------- Tabs ----------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Lake AI Analysis",
        "🏗️ Encroachment Detection",
        "🚨 Enforcement Alerts",
        "📡 IoT Monitoring",
        "📄 Reports & Cases"
    ])

    # ---------------- Tab 1 ----------------
    with tab1:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📸 Current Lake Image")
            st.image(current_image, use_container_width=True)

        with col2:
            st.subheader("💧 Detected Lake Region")
            st.image(lake_mask, use_container_width=True)

        with col3:
            st.subheader("🧠 AI Violation Overlay")

            overlay = current_array.copy()

            r = current_array[:, :, 0].astype(int)
            g = current_array[:, :, 1].astype(int)
            b = current_array[:, :, 2].astype(int)
            gray = cv2.cvtColor(current_array, cv2.COLOR_RGB2GRAY)

            algae_mask = (g > r + 20) & (g > b + 20) & (lake_mask > 0)
            waste_mask = (gray < 70) & (lake_mask > 0)

            overlay[algae_mask] = [255, 0, 0]
            overlay[waste_mask] = [0, 0, 255]

            st.image(overlay, use_container_width=True)
            st.caption("Red = algae | Blue = suspected waste")

        st.subheader("📈 Environmental Analysis")

        chart_data = pd.DataFrame({
            "Indicator": [
                "Algae Coverage",
                "Waste Indicator",
                "Muddy Coverage",
                "Encroachment Change",
                "IoT Anomaly Score"
            ],
            "Value": [
                round(metrics["algae"], 2),
                round(metrics["waste"], 2),
                round(metrics["murky"], 2),
                round(encroachment_percent, 2),
                sensor_score
            ]
        })

        st.bar_chart(chart_data.set_index("Indicator"))

    # ---------------- Tab 2 ----------------
    with tab2:

        st.subheader("🏗️ Encroachment and Change Detection")
