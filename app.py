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
st.caption("AI-based lake image analysis for pollution indicators and enforcement alerts")

st.sidebar.title("⚙️ Settings")
algae_sensitivity = st.sidebar.slider("Algae Sensitivity", 1, 10, 5)
murky_sensitivity = st.sidebar.slider("Murky Water Sensitivity", 1, 10, 5)

uploaded_file = st.file_uploader("Upload a lake image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    if max(image.size) > 1000:
        image.thumbnail((1000, 1000))

    arr = np.array(image)
    total_pixels = arr.shape[0] * arr.shape[1]

    r = arr[:, :, 0].astype(int)
    g = arr[:, :, 1].astype(int)
    b = arr[:, :, 2].astype(int)

    algae_mask = (g > r + 20) & (g > b + 20)
    algae_percent = (np.sum(algae_mask) / total_pixels) * 100

    brightness = np.mean(arr)
    clarity_score = 100 - brightness

    muddy_mask = (r > b + 25) & (g > b + 15)
    muddy_percent = (np.sum(muddy_mask) / total_pixels) * 100

    alerts = []

    if algae_percent > algae_sensitivity * 3:
        alerts.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Violation": "Possible Algae Growth",
            "Severity": "High",
            "Value": f"{algae_percent:.2f}% green coverage",
            "Action": "Send inspection team and collect water sample"
        })

    if clarity_score > 100 - murky_sensitivity * 5:
        alerts.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Violation": "Murky / Low-Clarity Water",
            "Severity": "Medium",
            "Value": f"Clarity score: {clarity_score:.1f}",
            "Action": "Check turbidity and possible contamination"
        })

    if muddy_percent > 30:
        alerts.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Violation": "Possible Muddy / Sediment Pollution",
            "Severity": "Medium",
            "Value": f"{muddy_percent:.2f}% muddy coverage",
            "Action": "Inspect nearby construction or soil erosion"
        })

    if not alerts:
        alerts.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Violation": "No Major Violation Detected",
            "Severity": "Low",
            "Value": "Lake image appears normal",
            "Action": "Continue routine monitoring"
        })

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Uploaded Lake Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🧠 AI Analysis")
        analysis = arr.copy()
        analysis[algae_mask] = [255, 0, 0]
        st.image(analysis, use_container_width=True)
        st.caption("Red areas indicate possible algae-like green coverage")

    st.subheader("📊 Lake Monitoring Metrics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algae Coverage", f"{algae_percent:.2f}%")
    c2.metric("Clarity Score", f"{clarity_score:.1f}")
    c3.metric("Muddy Coverage", f"{muddy_percent:.2f}%")
    c4.metric("Alerts", len(alerts))

    st.subheader("🚨 Enforcement Alerts")
    alert_df = pd.DataFrame(alerts)
    st.dataframe(alert_df, use_container_width=True)

    st.download_button(
        "⬇️ Download Enforcement Report",
        data=alert_df.to_csv(index=False).encode(),
        file_name=f"lake_report_{int(time.time())}.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a lake image to begin analysis.")

st.markdown("---")
st.caption(
    "Demo system using AI-based image processing. "
    "In full deployment, this can connect to drones, CCTV, GPS, and municipal dashboards."
)
