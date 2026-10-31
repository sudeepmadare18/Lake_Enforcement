import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ---------------- Page setup ----------------
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
    '<div class="sub-title">YOLOv8-based detection of floating waste, boats, '
    'and human activity for lake enforcement</div>',
    unsafe_allow_html=True
)

# ---------------- Session state ----------------
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
    "Monitoring Input Source",
    ["Drone Image", "CCTV Camera", "Satellite Image"]
)

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.95,
    value=0.35,
    step=0.05
)

# ---------------- Enforcement rules ----------------
ENFORCEMENT_RULES = {
    "bottle": {
        "violation": "Plastic Bottle / Littering Detected",
        "severity": "High",
        "action": "Deploy cleanup team and issue littering notice if verified"
    },
    "cup": {
        "violation": "Disposable Waste Detected",
        "severity": "Medium",
        "action": "Verify waste and initiate cleanup"
    },
    "boat": {
        "violation": "Boat Activity Detected",
        "severity": "Medium",
        "action": "Verify boat permit and activity"
    },
    "person": {
        "violation": "Human Activity Detected",
        "severity": "Low",
        "action": "Monitor activity for safety or violation"
    },
    "backpack": {
        "violation": "Person Carrying Baggage Detected",
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
st.header("📤 Lake Monitoring Input")

uploaded_file = st.file_uploader(
    "Upload Lake / Drone Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Running YOLOv8 lake enforcement analysis..."):
        results = model.predict(image, conf=confidence, verbose=False)[0]

    annotated_image = results.plot()[:, :, ::-1]

    # ---------------- Analyze detections ----------------
    alerts = []
    detections = []

    for box in results.boxes:
        class_name = model.names[int(box.cls[0])]
        conf = round(float(box.conf[0]), 2)

        detections.append({
            "Detected Object": class_name,
            "Confidence": conf
        })

        if class_name in ENFORCEMENT_RULES:
            rule = ENFORCEMENT_RULES[class_name]

            alerts.append({
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Lake": lake_name,
                "Source": input_source,
                "Violation": rule["violation"],
                "Detected Object": class_name,
                "Confidence": conf,
                "Severity": rule["severity"],
                "Recommended Action": rule["action"],
                "Status": "Pending"
            })

    if not alerts:
        alerts.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Lake": lake_name,
            "Source": input_source,
            "Violation": "No Major Violation Detected",
            "Detected Object": "None",
            "Confidence": 0.0,
            "Severity": "Low",
            "Recommended Action": "Continue routine monitoring",
            "Status": "Pending"
        })

    st.session_state.alert_history.extend(alerts)

    # ---------------- Dashboard ----------------
    st.header("📊 Lake Monitoring Dashboard")

    total_detections = len(detections)
    high_alerts = sum(1 for a in alerts if a["Severity"] == "High")
    medium_alerts = sum(1 for a in alerts if a["Severity"] == "Medium")
    low_alerts = sum(1 for a in alerts if a["Severity"] == "Low")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Objects Detected", total_detections)
    m2.metric("High Alerts", high_alerts)
    m3.metric("Medium Alerts", medium_alerts)
    m4.metric("Low Alerts", low_alerts)

    # ---------------- Tabs ----------------
    tab1, tab2, tab3 = st.tabs(
        ["🔍 AI Detection", "🚨 Enforcement Alerts", "📄 Report"]
    )

    # ---------------- Tab 1 ----------------
    with tab1:

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Uploaded Lake Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("🤖 YOLOv8 Detection")
            st.image(annotated_image, use_container_width=True)

        st.subheader("🔍 Detected Objects")

        if detections:
            detection_df = pd.DataFrame(detections)
            st.dataframe(detection_df, use_container_width=True)
        else:
            st.info("No objects detected above the selected confidence level.")

    # ---------------- Tab 2 ----------------
    with tab2:

        st.subheader("🚨 Current Enforcement Alerts")

        alert_df = pd.DataFrame(alerts)

        for index, row in alert_df.iterrows():

            if row["Severity"] == "High":
                st.markdown(
                    f"""
                    <div class="alert-high">
                        <b>🔴 HIGH:</b> {row['Violation']}<br>
                        <b>Detected Object:</b> {row['Detected Object']}<br>
                        <b>Confidence:</b> {row['Confidence']}<br>
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
                        <b>Detected Object:</b> {row['Detected Object']}<br>
                        <b>Confidence:</b> {row['Confidence']}<br>
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
                        <b>Detected Object:</b> {row['Detected Object']}<br>
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

        st.subheader("📄 Lake Enforcement Report")

        st.markdown(f"""
        **Lake Name:** {lake_name}  
        **Location:** {location}  
        **Coordinates:** {latitude}, {longitude}  
        **Input Source:** {input_source}  
        **Analysis Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
        **Total Detections:** {total_detections}  
        """)

        st.markdown("### Detected Objects")

        if detections:
            st.dataframe(pd.DataFrame(detections), use_container_width=True)

        st.markdown("### Download Report")

        csv_data = pd.DataFrame(st.session_state.alert_history).to_csv(
            index=False
        ).encode()

        st.download_button(
            label="⬇️ Download Enforcement Report",
            data=csv_data,
            file_name=f"lake_enforcement_report_{int(time.time())}.csv",
            mime="text/csv"
        )

else:
    st.info("Upload a lake or drone image to begin AI-based enforcement analysis.")

# ---------------- Footer ----------------
st.markdown("---")
st.caption(
    "AI Lake Enforcement System | YOLOv8 object detection | "
    "Can be extended with custom waste model, drone feeds, GPS, and municipal dashboard."
)
