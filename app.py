import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ---------------- Page setup ----------------
st.set_page_config(
    page_title="AI Lake Enforcement System",
    page_icon="🌊",
    layout="wide"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f4c81;
    }
    .sub-title {
        color: #555;
        font-size: 1rem;
    }
    .alert-high {
        background-color: #ffe5e5;
        border-left: 6px solid #d9534f;
        padding: 12px;
        border-radius: 8px;
    }
    .alert-medium {
        background-color: #fff6e0;
        border-left: 6px solid #f0ad4e;
        padding: 12px;
        border-radius: 8px;
    }
    .alert-low {
        background-color: #e8f6ff;
        border-left: 6px solid #5bc0de;
        padding: 12px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌊 AI-Based Lake Enforcement System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Intelligent monitoring of lakes for algae growth, '
    'water pollution indicators, and environmental risk assessment</div>',
    unsafe_allow_html=True
)

# ---------------- Initialize alert history ----------------
if "alert_history" not in st.session_state:
    st.session_state.alert_history = []

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Enforcement Settings")

lake_name = st.sidebar.text_input(
    "Lake Name",
    value="Sample Lake"
)

location = st.sidebar.text_input(
    "Location / Coordinates",
    value="12.9716° N, 77.5946° E"
)

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

st.sidebar.markdown("---")
st.sidebar.info(
    "Upload a lake image to analyze pollution indicators "
    "and generate enforcement alerts."
)

# ---------------- Upload image ----------------
uploaded_file = st.file_uploader(
    "Upload Lake Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    if max(image.size) > 1000:
        image.thumbnail((1000, 1000))

    arr = np.array(image)
    total_pixels = arr.shape[0] * arr.shape[1]

    r = arr[:, :, 0].astype(int)
    g = arr[:, :, 1].astype(int)
    b = arr[:, :, 2].astype(int)

    # ---------------- AI analysis ----------------
    algae_mask = (g > r + 20) & (g > b + 20)
    algae_percent = (np.sum(algae_mask) / total_pixels) * 100

    brightness = np.mean(arr)
    clarity_score = 100 - brightness

    muddy_mask = (r > b + 25) & (g > b + 15)
    muddy_percent = (np.sum(muddy_mask) / total_pixels) * 100

    # ---------------- Risk score ----------------
    algae_risk = min(algae_percent / 3, 40)
    clarity_risk = min(clarity_score / 2, 35)
    muddy_risk = min(muddy_percent / 4, 25)

    risk_score = round(algae_risk + clarity_risk + muddy_risk, 1)

    if risk_score >= 70:
        risk_level = "HIGH RISK"
        risk_color = "🔴"
    elif risk_score >= 40:
        risk_level = "MEDIUM RISK"
        risk_color = "🟠"
    else:
        risk_level = "LOW RISK"
        risk_color = "🟢"

    # ---------------- Generate alerts ----------------
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alerts = []

    if algae_percent > algae_sensitivity * 3:
        alerts.append({
            "Time": current_time,
            "Lake": lake_name,
            "Violation": "Possible Algae Growth",
            "Severity": "High",
            "Risk Score": risk_score,
            "Detected Value": f"{algae_percent:.2f}% green coverage",
            "Recommended Action": "Send inspection team and collect water sample"
        })

    if clarity_score > 100 - murky_sensitivity * 5:
        alerts.append({
            "Time": current_time,
            "Lake": lake_name,
            "Violation": "Murky / Low-Clarity Water",
            "Severity": "Medium",
            "Risk Score": risk_score,
            "Detected Value": f"Clarity score: {clarity_score:.1f}",
            "Recommended Action": "Check turbidity and possible contamination"
        })

    if muddy_percent > 30:
        alerts.append({
            "Time": current_time,
            "Lake": lake_name,
            "Violation": "Possible Muddy / Sediment Pollution",
            "Severity": "Medium",
            "Risk Score": risk_score,
            "Detected Value": f"{muddy_percent:.2f}% muddy coverage",
            "Recommended Action": "Inspect nearby construction or soil erosion"
        })

    if not alerts:
        alerts.append({
            "Time": current_time,
            "Lake": lake_name,
            "Violation": "No Major Violation Detected",
            "Severity": "Low",
            "Risk Score": risk_score,
            "Detected Value": "Lake image appears normal",
            "Recommended Action": "Continue routine monitoring"
        })

    # Save alerts in session history
    st.session_state.alert_history.extend(alerts)

    # ---------------- Top metrics ----------------
    st.markdown("### 📊 Lake Monitoring Dashboard")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "AI Risk Score",
        f"{risk_score}/100",
        risk_level
    )

    m2.metric(
        "Algae Coverage",
        f"{algae_percent:.2f}%",
        "Green region analysis"
    )

    m3.metric(
        "Water Clarity Score",
        f"{clarity_score:.1f}",
        "Lower means murkier"
    )

    m4.metric(
        "Active Alerts",
        len(alerts),
        "Generated from image analysis"
    )

    # ---------------- Tabs ----------------
    tab1, tab2, tab3 = st.tabs(
        ["🔍 Image Analysis", "🚨 Enforcement Alerts", "📄 Report"]
    )

    # ---------------- Tab 1: Image analysis ----------------
    with tab1:

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Uploaded Lake Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("🧠 AI Detection Overlay")
            analysis_image = arr.copy()
            analysis_image[algae_mask] = [255, 0, 0]
            st.image(analysis_image, use_container_width=True)
            st.caption(
                "Red regions indicate possible algae-like green coverage."
            )

        st.markdown("### 📈 Environmental Indicators")

        chart_data = pd.DataFrame(
            {
                "Indicator": [
                    "Algae Coverage",
                    "Muddy Coverage",
                    "Water Clarity Risk"
                ],
                "Value": [
                    round(algae_percent, 2),
                    round(muddy_percent, 2),
                    round(clarity_score, 2)
                ]
            }
        )

        st.bar_chart(chart_data.set_index("Indicator"))

    # ---------------- Tab 2: Alerts ----------------
    with tab2:

        st.subheader("🚨 Current Enforcement Alerts")

        alert_df = pd.DataFrame(alerts)

        for _, row in alert_df.iterrows():
            if row["Severity"] == "High":
                st.markdown(
                    f"""
                    <div class="alert-high">
                        <b>🔴 HIGH:</b> {row['Violation']}<br>
                        <b>Detected:</b> {row['Detected Value']}<br>
                        <b>Action:</b> {row['Recommended Action']}
                    </div>
                    <br>
                    """,
                    unsafe_allow_html=True
                )

            elif row["Severity"] == "Medium":
                st.markdown(
                    f"""
                    <div class="alert-medium">
                        <b>🟠 MEDIUM:</b> {row['Violation']}<br>
                        <b>Detected:</b> {row['Detected Value']}<br>
                        <b>Action:</b> {row['Recommended Action']}
                    </div>
                    <br>
                    """,
                    unsafe_allow_html=True
                )

            else:
                st.markdown(
                    f"""
                    <div class="alert-low">
                        <b>🟢 LOW:</b> {row['Violation']}<br>
                        <b>Detected:</b> {row['Detected Value']}<br>
                        <b>Action:</b> {row['Recommended Action']}
                    </div>
                    <br>
                    """,
                    unsafe_allow_html=True
                )

        st.subheader("🗂️ Alert History")

        if st.session_state.alert_history:
            history_df = pd.DataFrame(st.session_state.alert_history)
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No previous alerts in this session.")

    # ---------------- Tab 3: Report ----------------
    with tab3:

        st.subheader("📄 Lake Enforcement Report")

        st.markdown(f"""
        **Lake Name:** {lake_name}  
        **Location:** {location}  
        **Analysis Time:** {current_time}  
        **AI Risk Score:** {risk_score}/100  
        **Risk Level:** {risk_level}  
        """)

        st.markdown("### Environmental Measurements")

        report_metrics = pd.DataFrame(
            {
                "Parameter": [
                    "Algae Coverage",
                    "Muddy Coverage",
                    "Water Clarity Score",
                    "AI Risk Score"
                ],
                "Value": [
                    f"{algae_percent:.2f}%",
                    f"{muddy_percent:.2f}%",
                    f"{clarity_score:.1f}",
                    f"{risk_score}/100"
                ]
            }
        )

        st.dataframe(report_metrics, use_container_width=True)

        st.markdown("### Download Report")

        csv_data = pd.DataFrame(st.session_state.alert_history).to_csv(
            index=False
        ).encode()

        st.download_button(
            label="⬇️ Download Full Enforcement Report",
            data=csv_data,
            file_name=f"lake_enforcement_report_{int(time.time())}.csv",
            mime="text/csv"
        )

else:
    st.info("Upload a lake image to begin AI-based enforcement analysis.")

# ---------------- Footer ----------------
st.markdown("---")
st.caption(
    "AI Lake Enforcement System | Image-processing-based environmental monitoring demo | "
    "Can be extended with drone feeds, CCTV, GPS, IoT sensors, and municipal dashboards."
)
