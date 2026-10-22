import time
from datetime import datetime

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
    "AI-based analysis of lake images for pollution indicators, "
    "water clarity, algae presence, and enforcement alerts"
)

# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Enforcement Settings")

algae_sensitivity = st.sidebar.slider(
    "Algae Detection Sensitivity",
    min_value=1,
    max_value=10,
    value=5
)

murky_sensitivity = st.sidebar.slider(
    "Murky Water Sensitivity",
    min_value=1,
    max_value=10,
    value=5
)

# ---------------- Upload image ----------------
uploaded_file = st.file_uploader(
    "Upload a lake image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    # Resize very large images for faster processing
    max_size = 1000
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size))
        image_array = np.array(image)

    height, width, _ = image_array.shape
    total_pixels = height * width

    # Separate color channels
    red_channel = image_array[:, :, 0].astype(float)
    green_channel = image_array[:, :, 1].astype(float)
    blue_channel = image_array[:, :, 2].astype(float)

    # ---------------- Algae detection ----------------
    # Green pixels: green is higher than red and blue
    algae_mask = (
        (green_channel > red_channel + 20) &
        (green_channel > blue_channel + 20)
    )

    algae_percentage = (np.sum(algae_mask) / total_pixels) * 100

    # ---------------- Water clarity ----------------
    brightness = np.mean(image_array)

    # Lower brightness means darker/murkier water
    clarity_score = 100 - brightness

    # ---------------- Pollution indicator ----------------
    # Brownish/muddy water: red and green higher than blue
    muddy_mask = (
        (red_channel > blue_channel + 25) &
        (green_channel > blue_channel + 15)
    )

    muddy_percentage = (np.sum(muddy_mask) / total_pixels) * 100

    # ---------------- Create alerts ----------------
    alerts = []

    algae_threshold = algae_sensitivity * 3
    murky_threshold = 100 - (murky_sensitivity * 5)

    if algae_percentage > algae_threshold:
        alerts.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Violation Type": "Possible Algae Growth",
            "Severity": "High",
            "Detected Value": f"{algae_percentage:.2f}% green coverage",
            "Recommended Action": "Send inspection team and collect water sample"
        })

    if clarity_score > murky_threshold:
        alerts.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Violation Type": "Murky / Low-Clarity Water",
            "Severity": "Medium",
            "Detected Value": f"Clarity score: {clarity_score:.1f}",
            "Recommended Action": "Check turbidity and possible contamination"
        })

    if muddy_percentage > 30:
        alerts.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Violation Type": "Possible Muddy / Sediment Pollution",
            "Severity": "Medium",
            "Detected Value": f"{muddy_percentage:.2f}% muddy-color coverage",
            "Recommended Action": "Inspect surrounding construction or soil erosion"
        })

    if not alerts:
        alerts.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Violation Type": "No Major Violation Detected",
            "Severity": "Low",
            "Detected Value": "Lake image appears normal",
            "Recommended Action": "Continue routine monitoring"
        })

    # ---------------- Show images ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Uploaded Lake Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🧠 AI Analysis")

        # Create a simple analysis visualization
        analysis_image = image_array.copy()

        # Highlight algae-like areas in red
        analysis_image[algae_mask] = [255, 0, 0]

        st.image(analysis_image, use_container_width=True)
        st.caption("Red areas indicate possible algae-like green coverage")

    # ---------------- Metrics ----------------
    st.subheader("📊 Lake Monitoring Metrics")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Algae Coverage",
        f"{algae_percentage:.2f}%"
    )

    m2.metric(
        "Water Clarity Score",
        f"{clarity_score:.1f}"
    )

    m3.metric(
        "Muddy Color Coverage",
        f"{muddy_percentage:.2f}%"
    )

    m4.metric(
        "Alerts Generated",
        len(alerts)
    )

    # ---------------- Alerts table ----------------
    st.subheader("🚨 Lake Enforcement Alerts")

    alert_df = pd.DataFrame(alerts)

    st.dataframe(alert_df, use_container_width=True)

    # ---------------- Download report ----------------
    csv_data = alert_df.to_csv(index=False).encode()

    st.download_button(
        label="⬇️ Download Enforcement Report",
        data=csv_data,
        file_name=f"lake_enforcement_report_{int(time.time())}.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a lake image to begin AI-based enforcement analysis.")

# ---------------- Footer ----------------
st.markdown("---")
st.caption(
    "Demo system using image-processing-based lake analysis. "
    "In a full deployment, this can be connected to drone cameras, CCTV, "
    "GPS, water-quality sensors, and a municipal monitoring dashboard."
)
