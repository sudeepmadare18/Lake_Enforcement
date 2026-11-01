import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageFilter

st.set_page_config(
    page_title="AI Lake Enforcement System",
    page_icon="🌊",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #4fc3f7;
    }
    .sub-title {
        color: #cccccc;
        font-size: 1.05rem;
    }
    .alert-high,
    .alert-medium,
    .alert-low {
        color: #111111 !important;
        padding: 14px;
        border-radius: 10px;
        margin-bottom: 12px;
    }
    .alert-high {
        background-color: #ffe5e5 !important;
        border-left: 6px solid #d9534f !important;
    }
    .alert-medium {
        background-color: #fff6e0 !important;
        border-left: 6px solid #f0ad4e !important;
    }
    .alert-low {
        background-color: #e8f6ff !important;
        border-left: 6px solid #5bc0de !important;
    }
    .alert-high b,
    .alert-medium b,
    .alert-low b {
        color: #111111 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌊 AI-Based Lake Enforcement System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI image analysis for algae, water pollution, '
    'floating waste, and lake enforcement alerts</div>',
    unsafe_allow_html=True
)

if "alert_history" not in st.session_state:
    st.session_state.alert_history = []

# ---------------- Sidebar ----------------
st.sidebar.header("🏞️ Lake Profile")

lake_name = st.sidebar.text_input("Lake Name", value="Sample Lake")
location = st.sidebar.text_input("Location", value="Bengaluru, India")
latitude = st.sidebar.number_input("Latitude", value=12.9716, format="%.4f")
longitude = st.sidebar.number_input("Longitude", value=77.5946, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ AI Settings")

input_source = st.sidebar.selectbox(
    "Monitoring Source",
    ["Drone Image", "CCTV Camera", "Satellite Image"]
)

algae_sensitivity = st.sidebar.slider("Algae Sensitivity", 1, 10, 5)
waste_sensitivity = st.sidebar.slider("Waste Sensitivity", 1, 10, 5)
murky_sensitivity = st.sidebar.slider("Murky Water Sensitivity", 1, 10, 5)

st.sidebar.markdown("---")
st.sidebar.header("📡 IoT Sensor Simulation")

ph_value = st.sidebar.slider("pH Level", 0.0, 14.0, 7.2, 0.1)
turbidity = st.sidebar.slider("Turbidity (NTU)", 0, 100, 25)
dissolved_oxygen = st.sidebar.slider("Dissolved Oxygen (mg/L)", 0.0, 15.0, 6.5, 0.1)
water_level = st.sidebar.slider("Water Level (%)", 0, 100, 78)

# ---------------- Upload ----------------
st.header("📤 Upload Lake Image")

uploaded_file = st.file_uploader(
    "Upload a lake image",
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

    gray = np.array(image.convert("L"))
    brightness = np.mean(gray)
    clarity_score = 100 - brightness

    waste_mask = gray < 70
    waste_percent = (np.sum(waste_mask) / total_pixels) * 100

    muddy_mask = (r > b + 25) & (g > b + 15)
    muddy_percent = (np.sum(muddy_mask) / total_pixels) * 100

    algae_threshold = algae_sensitivity * 3
    waste_threshold = 15 - waste_sensitivity
    murky_threshold = 100 - murky_sensitivity * 5

    # ---------------- Alerts ----------------
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alerts = []

    if algae_percent > algae_threshold:
        alerts.append({
            "Time": current_time,
            "Lake": lake_name,
            "Source": input_source,
            "Violation": "Possible Algae Growth",
            "Severity": "High",
            "Detected Value": f"{algae_percent:.2f}% green coverage",
            "Recommended Action": "Send inspection team and collect water sample",
            "Status": "Pending"
        })

    if waste_percent > waste_threshold:
        alerts.append({
            "Time": current_time,
            "Lake": lake_name,
            "Source": input_source,
            "Violation": "Possible Floating Waste Detected",
            "Severity": "High",
            "Detected Value": f"{waste_percent:.2f}% dark floating regions",
            "Recommended Action": "Deploy cleanup team and verify waste type",
            "Status": "Pending"
        })

    if clarity_score > murky_threshold:
        alerts.append({
            "Time": current_time,
            "Lake": lake_name,
            "Source": input_source,
            "Violation": "Murky / Low-Clarity Water",
            "Severity": "Medium",
            "Detected Value": f"Clarity score: {clarity_score:.1f}",
            "Recommended Action": "Check turbidity and possible contamination",
            "Status": "Pending"
        })

    if muddy_percent > 30:
        alerts.append({
            "Time": current_time,
            "Lake": lake_name,
            "Source": input_source,
            "Violation": "Possible Muddy / Sediment Pollution",
            "Severity": "Medium",
            "Detected Value": f"{muddy_percent:.2f}% muddy coverage",
            "Recommended Action": "Inspect nearby construction or soil erosion",
            "Status": "Pending"
        })

    if ph_value < 6.5 or ph_value > 8.5:
        alerts.append({
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
            "Time": current_time,
            "Lake": lake_name,
            "Source": input_source,
            "Violation": "No Major Violation Detected",
            "Severity": "Low",
            "Detected Value": "Lake appears normal",
            "Recommended Action": "Continue routine monitoring",
            "Status": "Pending"
        })

    st.session_state.alert_history.extend(alerts)

    # ---------------- Risk score ----------------
    risk_score = min(
        round(
            (algae_percent / 3) +
            (waste_percent / 2) +
            (clarity_score / 2.5) +
            (len([a for a in alerts if a["Source"] == "IoT Sensor"]) * 5),
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
    st.header("📊 Lake Monitoring Dashboard")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("AI Risk Score", f"{risk_score}/100", risk_level)
    m2.metric("Algae Coverage", f"{algae_percent:.2f}%")
    m3.metric("Waste Indicator", f"{waste_percent:.2f}%")
    m4.metric("Active Alerts", len(alerts))

    m5, m6, m7, m8 = st.columns(4)

    m5.metric("Water Clarity Score", f"{clarity_score:.1f}")
    m6.metric("Muddy Coverage", f"{muddy_percent:.2f}%")
    m7.metric("Turbidity", f"{turbidity} NTU")
    m8.metric("Dissolved Oxygen", f"{dissolved_oxygen} mg/L")

    # ---------------- Tabs ----------------
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔍 AI Image Analysis",
            "🚨 Enforcement Alerts",
            "📡 Sensor Monitoring",
            "📄 Report & Map"
        ]
    )

    # ---------------- Tab 1 ----------------
    with tab1:

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Uploaded Lake Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("🧠 AI Detection Overlay")

            analysis = arr.copy()
            analysis[algae_mask] = [255, 0, 0]
            analysis[waste_mask] = [0, 0, 255]

            st.image(analysis, use_container_width=True)
            st.caption(
                "Red = possible algae | Blue = possible floating waste / dark objects"
            )

        st.subheader("📈 Environmental Indicators")

        chart_data = pd.DataFrame(
            {
                "Indicator": [
                    "Algae Coverage",
                    "Waste Indicator",
                    "Muddy Coverage",
                    "Water Clarity Risk"
                ],
                "Value": [
                    round(algae_percent, 2),
                    round(waste_percent, 2),
                    round(muddy_percent, 2),
                    round(clarity_score, 2)
                ]
            }
        )

        st.bar_chart(chart_data.set_index("Indicator"))

    # ---------------- Tab 2 ----------------
    with tab2:

        st.subheader("🚨 Current Enforcement Alerts")

        alert_df = pd.DataFrame(alerts)

        for _, row in alert_df.iterrows():

            if row["Severity"] == "High":
                st.markdown(
                    f"""
                    <div class="alert-high">
                        <b>🔴 HIGH:</b> {row['Violation']}<br>
                        <b>Source:</b> {row['Source']}<br>
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
                        <b>Source:</b> {row['Source']}<br>
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
                        <b>Source:</b> {row['Source']}<br>
                        <b>Detected:</b> {row['Detected Value']}<br>
                        <b>Action:</b> {row['Recommended Action']}
                    </div>
                    <br>
                    """,
                    unsafe_allow_html=True
                )

        st.subheader("🗂️ Alert History")

        history_df = pd.DataFrame(st.session_state.alert_history)
        st.dataframe(history_df, use_container_width=True)

    # ---------------- Tab 3 ----------------
    with tab3:

        st.subheader("📡 IoT Water-Quality Monitoring")

        s1, s2, s3, s4 = st.columns(4)

        s1.metric("pH Level", f"{ph_value}", "Normal: 6.5 - 8.5")
        s2.metric("Turbidity", f"{turbidity} NTU", "Normal: below 50")
        s3.metric("Dissolved Oxygen", f"{dissolved_oxygen} mg/L", "Normal: above 4")
        s4.metric("Water Level", f"{water_level}%")

        st.subheader("💧 Water Level")
        st.progress(water_level / 100)

        st.subheader("🧪 Sensor Alerts")

        sensor_alerts = [
            a for a in alerts
            if a["Source"] == "IoT Sensor"
        ]

        if sensor_alerts:
            for a in sensor_alerts:
                st.warning(f"⚠️ {a['Violation']}")
        else:
            st.success("All sensor readings are normal.")

        st.subheader("📊 Sensor Values")

        sensor_chart = pd.DataFrame(
            {
                "Parameter": [
                    "pH",
                    "Turbidity",
                    "Dissolved Oxygen",
                    "Water Level"
                ],
                "Value": [
                    ph_value,
                    turbidity,
                    dissolved_oxygen,
                    water_level
                ]
            }
        )

        st.bar_chart(sensor_chart.set_index("Parameter"))

    # ---------------- Tab 4 ----------------
    with tab4:

        st.subheader("📄 Lake Enforcement Report")

        st.markdown(f"""
        **Lake Name:** {lake_name}  
        **Location:** {location}  
        **Coordinates:** {latitude}, {longitude}  
        **Input Source:** {input_source}  
        **Analysis Time:** {current_time}  
        **AI Risk Score:** {risk_score}/100  
        **Risk Level:** {risk_level}  
        """)

        st.subheader("📍 Lake Location Map")

        map_data = pd.DataFrame(
            {
                "lat": [latitude],
                "lon": [longitude]
            }
        )

        st.map(map_data, zoom=12)

        st.subheader("⬇️ Download Report")

        csv_data = pd.DataFrame(st.session_state.alert_history).to_csv(
            index=False
        ).encode()

        st.download_button(
            label="Download Full Enforcement Report",
            data=csv_data,
            file_name=f"lake_enforcement_report_{int(time.time())}.csv",
            mime="text/csv"
        )

else:
    st.info("Upload a lake image to begin AI-based enforcement analysis.")

# ---------------- Footer ----------------
st.markdown("---")
st.caption(
    "AI Lake Enforcement System | Image-processing-based environmental monitoring | "
    "Can be extended with drones, CCTV, satellite imagery, and government dashboards."
)
