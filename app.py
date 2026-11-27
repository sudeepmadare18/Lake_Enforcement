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

st.markdown("""
<style>
.main-title {font-size:2.4rem;font-weight:800;color:#4fc3f7;}
.sub-title {color:#cccccc;font-size:1.05rem;}
.alert-high,.alert-medium,.alert-low {
color:#111 !important;padding:14px;border-radius:10px;margin-bottom:12px;}
.alert-high {background:#ffe5e5 !important;border-left:6px solid #d9534f !important;}
.alert-medium {background:#fff6e0 !important;border-left:6px solid #f0ad4e !important;}
.alert-low {background:#e8f6ff !important;border-left:6px solid #5bc0de !important;}
.alert-high b,.alert-medium b,.alert-low b {color:#111 !important;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌊 Advanced AI Lake Enforcement System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Lake segmentation, pollution analysis, encroachment '
    'detection, IoT anomaly detection, and enforcement workflow</div>',
    unsafe_allow_html=True
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
dissolved_oxygen = st.sidebar.slider("Dissolved Oxygen (mg/L)", 0.0, 15.0, 6.5, 0.1)
temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 27.0, 0.5)
water_level = st.sidebar.slider("Water Level (%)", 0, 100, 78)

# ---------------- Image upload ----------------
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
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    lake_mask = np.zeros_like(mask)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(lake_mask, [largest], -1, 255, thickness=cv2.FILLED)

    return lake_mask, mask


def analyze_lake(image, lake_mask):
    img = np.array(image)
    total_pixels = np.count_nonzero(lake_mask)

    if total_pixels == 0:
        return {
            "algae": 0,
            "waste": 0,
            "murky": 0,
            "lake_area": 0
        }

    r = img[:, :, 0].astype(int)
    g = img[:, :, 1].astype(int)
    b = img[:, :, 2].astype(int)

    algae_mask = (g > r + 20) & (g > b + 20) & (lake_mask > 0)
    waste_mask = (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) < 70) & (lake_mask > 0)
    murky_mask = (r > b + 25) & (g > b + 15) & (lake_mask > 0)

    return {
        "algae": (np.sum(algae_mask) / total_pixels) * 100,
        "waste": (np.sum(waste_mask) / total_pixels) * 100,
        "murky": (np.sum(murky_mask) / total_pixels) * 100,
        "lake_area": total_pixels
    }


def detect_encroachment(reference_image, current_image):
    ref_mask, _ = extract_lake_mask(reference_image)
    cur_mask, _ = extract_lake_mask(current_image)

    ref_area = np.count_nonzero(ref_mask)
    cur_area = np.count_nonzero(cur_mask)

    if ref_area == 0:
        return 0, False, ref_mask, cur_mask

    reduction = ((ref_area - cur_area) / ref_area) * 100

    return max(reduction, 0), reduction > 0, ref_mask, cur_mask


def sensor_anomaly_score(ph, turbidity, do, temperature):
    data = np.array([
        [7.2, 20, 7.5, 26],
        [7.0, 25, 7.0, 27],
        [7.4, 30, 6.8, 28],
        [6.8, 35, 6.5, 29],
        [7.1, 28, 7.2, 26],
        [ph, turbidity, do, temperature]
    ])

    model = IsolationForest(
        contamination=0.25,
        random_state=42
    )

    model.fit(data)
    score = model.decision_function(data)[-1]

    return max(0, round((0.5 - score) * 100, 1))

# ---------------- Main analysis ----------------
if current_file is not None:

    current_image = resize_image(Image.open(current_file).convert("RGB"))
    current_arr = np.array(current_image)

    lake_mask, raw_mask = extract_lake_mask(current_image)
    metrics = analyze_lake(current_image, lake_mask)

    encroachment_percent = 0
    encroachment_detected = False
    reference_image = None

    if reference_file is not None:
        reference_image = resize_image(Image.open(reference_file).convert("RGB"))
        encroachment_percent, encroachment_detected, ref_mask, cur_mask = detect_encroachment(
            reference_image,
            current_image
        )

    sensor_score = sensor_anomaly_score(
        ph_value,
        turbidity,
        dissolved_oxygen,
        temperature
    )

    # ---------------- Alerts ----------------
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alerts = []

    if metrics["algae"] > algae_threshold:
        alerts.append({
            "Alert ID": f"LK-{int(time.time())}-A",
            "Time": now,
            "Lake": lake_name,
            "Source": input_source,
            "Violation": "Algae Bloom Detected",
            "Severity": "High",
            "Value": f"{metrics['algae']:.2f}% algae coverage",
            "Action": "Collect water sample and inspect nutrient inflow",
            "Status": "Pending"
        })

    if metrics["waste"] > waste_threshold:
        alerts.append({
            "Alert ID": f"LK-{int(time.time())}-W",
            "Time": now,
            "Lake": lake_name,
            "Source": input_source,
            "Violation": "Floating Waste Detected",
            "Severity": "High",
            "Value": f"{metrics['waste']:.2f}% suspected waste coverage",
            "Action": "Deploy cleanup team and identify waste source",
            "Status": "Pending"
        })

    if metrics["murky"] > 30:
        alerts.append({
            "Alert ID": f"LK-{int(time.time())}-M",
            "Time": now,
            "Lake": lake_name,
            "Source": input_source,
            "Violation": "Sediment / Muddy Water Detected",
            "Severity": "Medium",
            "Value": f"{metrics['murky']:.2f}% muddy coverage",
            "Action": "Inspect nearby construction and soil erosion",
            "Status": "Pending"
        })

    if encroachment_detected and encroachment_percent > encroachment_threshold:
        alerts.append({
            "Alert ID": f"LK-{int(time.time())}-E",
            "Time": now,
            "Lake": lake_name,
            "Source": "Change Detection",
            "Violation": "Possible Lake Encroachment",
            "Severity": "High",
            "Value": f"{encroachment_percent:.2f}% water-area reduction",
            "Action": "Verify land records and conduct field survey",
            "Status": "Pending"
        })

    if sensor_score > 50:
        alerts.append({
            "Alert ID": f"LK-{int(time.time())}-S",
            "Time": now,
            "Lake": lake_name,
            "Source": "IoT Anomaly Detection",
            "Violation": "Abnormal Water-Quality Pattern",
            "Severity": "High",
            "Value": f"Anomaly score: {sensor_score}",
            "Action": "Validate sensors and investigate pollution source",
            "Status": "Pending"
        })

    if ph_value < 6.5 or ph_value > 8.5:
        alerts.append({
            "Alert ID": f"LK-{int(time.time())}-PH",
            "Time": now,
            "Lake": lake_name,
            "Source": "IoT Sensor",
            "Violation": "Abnormal pH Level",
            "Severity": "Medium",
            "Value": f"pH {ph_value}",
            "Action": "Validate sensor and inspect lake",
            "Status": "Pending"
        })

    if turbidity > 50:
        alerts.append({
            "Alert ID": f"LK-{int(time.time())}-T",
            "Time": now,
            "Lake": lake_name,
            "Source": "IoT Sensor",
            "Violation": "High Turbidity",
            "Severity": "Medium",
            "Value": f"{turbidity} NTU",
            "Action": "Check sediment inflow and pollution source",
            "Status": "Pending"
        })

    if dissolved_oxygen < 4:
        alerts.append({
            "Alert ID": f"LK-{int(time.time())}-DO",
            "Time": now,
            "Lake": lake_name,
            "Source": "IoT Sensor",
            "Violation": "Low Dissolved Oxygen",
            "Severity": "High",
            "Value": f"{dissolved_oxygen} mg/L",
            "Action": "Check aquatic health and pollution source",
            "Status": "Pending"
        })

    if not alerts:
        alerts.append({
            "Alert ID": f"LK-{int(time.time())}-N",
            "Time": now,
            "Lake": lake_name,
            "Source": input_source,
            "Violation": "No Major Violation Detected",
            "Severity": "Low",
            "Value": "Lake appears normal",
            "Action": "Continue routine monitoring",
            "Status": "Pending"
        })

    st.session_state.alerts.extend(alerts)

    # ---------------- Risk score ----------------
    waste_score = min(metrics["waste"] * 2, 100)
    algae_score = min(metrics["algae"] * 2, 100)
    encroachment_score = min(encroachment_percent * 3, 100)
    human_activity_score = 0

    risk_score = min(
        round(
            (0.30 * waste_score) +
            (0.25 * algae_score) +
            (0.20 * encroachment_score) +
            (0.15 * sensor_score) +
            (0.10 * human_activity_score),
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

    m1.metric("AI
