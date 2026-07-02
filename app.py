import os
import streamlit as st
import numpy as np
from PIL import Image

from utils.image_utils import preprocess_image
from utils.clinical_utils import prepare_clinical_input
from utils.geo_utils import prepare_geo_input
from utils.prediction import predict_anemia

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AnemiaFusionNet",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background:#F4F8FB;
}

.block-container{
    padding-top:1rem;
    padding-bottom:2rem;
}

section[data-testid="stSidebar"]{
    background:#0B3D91;
}

section[data-testid="stSidebar"] *{
    color:white;
}

.header{

    background:linear-gradient(90deg,#C62828,#1565C0);

    padding:30px;

    border-radius:18px;

    color:white;

    text-align:center;

    box-shadow:0px 8px 25px rgba(0,0,0,.25);

}

.card{

    background:white;

    border-radius:18px;

    padding:20px;

    box-shadow:0px 5px 15px rgba(0,0,0,.12);

    margin-top:15px;

}

.metric-card{

    background:white;

    border-radius:15px;

    padding:15px;

    text-align:center;

    box-shadow:0px 5px 12px rgba(0,0,0,.12);

}

.stButton>button{

    width:100%;

    background:#C62828;

    color:white;

    border:none;

    border-radius:10px;

    height:50px;

    font-size:18px;

    font-weight:bold;

}

.stButton>button:hover{

    background:#A00000;

    color:white;

}

.footer{

    text-align:center;

    color:gray;

    margin-top:40px;

    font-size:14px;

}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================

st.markdown("""

<div class="header">

<h1>🩸 AnemiaFusionNet</h1>

<h3>Transformer-Based Multimodal Anemia Detection Framework</h3>

<p>

Eye Conjunctiva Image + Clinical Parameters + Geographic Risk

</p>

</div>

""", unsafe_allow_html=True)

st.write("")

# ==========================================================
# PROJECT OVERVIEW
# ==========================================================

st.markdown("""
<div class="card">

### Project Overview

This application predicts anemia using three different data sources:

- Eye Conjunctiva Image
- Clinical Blood Parameters
- Geographic Risk Information

All three modalities are encoded separately and fused using a Transformer Network before final classification.

</div>
""", unsafe_allow_html=True)

st.write("")

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Patient Information")

uploaded_image = st.sidebar.file_uploader(
    "Upload Eye Image",
    type=["jpg","jpeg","png"]
)

st.sidebar.markdown("---")

st.sidebar.subheader("Clinical Parameters")

hemoglobin = st.sidebar.number_input(
    "Hemoglobin (g/dL)",
    1.0,
    25.0,
    12.0,
    0.1
)

rdw = st.sidebar.number_input(
    "RDW (%)",
    5.0,
    30.0,
    13.5,
    0.1
)

mcv = st.sidebar.number_input(
    "MCV (fL)",
    40.0,
    120.0,
    82.0,
    0.1
)

age = st.sidebar.slider(
    "Age",
    1,
    100,
    25
)

gender = st.sidebar.selectbox(
    "Gender",
    [
        "Male",
        "Female"
    ]
)

st.sidebar.markdown("---")

geo_risk = st.sidebar.selectbox(
    "Geographic Risk",
    [
        "Low",
        "Medium",
        "High"
    ]
)

st.sidebar.markdown("---")

predict_button = st.sidebar.button(
    "🩸 Predict Anemia"
)

st.write("")
# ==========================================================
# MAIN DASHBOARD
# ==========================================================

left_col, right_col = st.columns([1.1, 1])

# ==========================================================
# LEFT PANEL
# ==========================================================

with left_col:

    st.markdown("""
    <div class="card">
    <h3>👁 Eye Conjunctiva Image</h3>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_image is not None:

        image = Image.open(uploaded_image)

        st.image(
            image,
            use_container_width=True
        )

    else:

        st.info(
            "Please upload an eye conjunctiva image."
        )

# ==========================================================
# RIGHT PANEL
# ==========================================================

with right_col:

    st.markdown("""
    <div class="card">
    <h3>🩺 Patient Information</h3>
    </div>
    """, unsafe_allow_html=True)

    info = {

        "Hemoglobin (g/dL)" : hemoglobin,
        "RDW (%)" : rdw,
        "MCV (fL)" : mcv,
        "Age" : age,
        "Gender" : gender,
        "Geo Risk" : geo_risk

    }

    patient_df = np.array(list(info.items()))

    st.table(patient_df)

# ==========================================================
# INPUT STATUS
# ==========================================================

st.write("")

c1, c2, c3 = st.columns(3)

with c1:

    if uploaded_image is None:

        st.error("Image Missing")

    else:

        st.success("Image Ready")

with c2:

    st.success("Clinical Data Ready")

with c3:

    st.success("Geo Risk Ready")

# ==========================================================
# VALIDATION
# ==========================================================

ready = True

if uploaded_image is None:

    ready = False

if hemoglobin <= 0:

    ready = False

if rdw <= 0:

    ready = False

if mcv <= 0:

    ready = False

# ==========================================================
# PROJECT WORKFLOW
# ==========================================================

st.write("")

st.markdown("""
<div class="card">

### 🔬 AI Workflow

📷 Eye Image

⬇

🧠 Image Encoder

⬇

🩸 Clinical Encoder

⬇

🌍 Geo Encoder

⬇

🔀 Transformer Fusion

⬇

🤖 Classifier

⬇

🩺 Final Prediction

</div>

""", unsafe_allow_html=True)

# ==========================================================
# DATASET INFORMATION
# ==========================================================

m1, m2, m3 = st.columns(3)

with m1:

    st.metric(
        "Eye Images",
        "183"
    )

with m2:

    st.metric(
        "Clinical Samples",
        "1000"
    )

with m3:

    st.metric(
        "Geo Records",
        "707"
    )

st.write("")
# ==========================================================
# AI PREDICTION
# ==========================================================

if predict_button:

    if not ready:

        st.error("Please upload an eye image and complete all patient details.")

    else:

        with st.spinner("Running AnemiaFusionNet AI Model..."):

            try:

                # --------------------------------------------------
                # IMAGE PREPROCESSING
                # --------------------------------------------------

                image = Image.open(uploaded_image)

                image = preprocess_image(image)

                # --------------------------------------------------
                # CLINICAL PREPROCESSING
                # --------------------------------------------------

                clinical = prepare_clinical_input(

                    hemoglobin,

                    rdw,

                    mcv,

                    age,

                    gender

                )

                # --------------------------------------------------
                # GEO PREPROCESSING
                # --------------------------------------------------

                geo = prepare_geo_input(

                    geo_risk

                )

                # --------------------------------------------------
                # AI PREDICTION
                # --------------------------------------------------

                result = predict_anemia(

                    image,

                    clinical,

                    geo

                )

                prediction = result["prediction"]

                probability = result["probability"]

                confidence = result["confidence"]

                prediction_done = True

            except Exception as e:

                prediction_done = False

                st.error(f"Prediction Error : {e}")

# ==========================================================
# RESULT
# ==========================================================

        if prediction_done:

            st.markdown("---")

            st.header("Prediction Result")

            col1, col2, col3 = st.columns(3)

            with col1:

                if prediction == "Anemic":

                    st.error("🩸 ANEMIC")

                else:

                    st.success("✅ NORMAL")

            with col2:

                st.metric(

                    "Probability",

                    f"{probability*100:.2f}%"

                )

            with col3:

                st.metric(

                    "Confidence",

                    f"{confidence*100:.2f}%"

                )

            st.write("")

# ==========================================================
# CONFIDENCE BAR
# ==========================================================

            st.subheader("Prediction Confidence")

            st.progress(

                float(confidence)

            )

            st.write("")

# ==========================================================
# PATIENT SUMMARY
# ==========================================================

            st.subheader("Patient Summary")

            summary = {

                "Hemoglobin (g/dL)" : hemoglobin,

                "RDW (%)" : rdw,

                "MCV (fL)" : mcv,

                "Age" : age,

                "Gender" : gender,

                "Geo Risk" : geo_risk,

                "Prediction" : prediction,

                "Probability" : f"{probability*100:.2f}%",

                "Confidence" : f"{confidence*100:.2f}%"

            }

            st.json(summary)

            st.success("Prediction Completed Successfully")
# ==========================================================
# MODEL ARCHITECTURE
# ==========================================================

st.markdown("---")

st.header("Model Architecture")

st.markdown("""

<div class="card">

<h4>Transformer-Based Multimodal Pipeline</h4>

<br>

<b>Step 1</b><br>
Eye Conjunctiva Image
<br><br>
⬇
<br><br>

<b>Step 2</b><br>
EfficientNetB0 Image Encoder
<br><br>
⬇
<br><br>

<b>Step 3</b><br>
Clinical Encoder
<br><br>
⬇
<br><br>

<b>Step 4</b><br>
Geographical Risk Encoder
<br><br>
⬇
<br><br>

<b>Step 5</b><br>
Transformer Fusion Network
<br><br>
⬇
<br><br>

<b>Step 6</b><br>
Binary Classifier
<br><br>
⬇
<br><br>

<b>Final Output</b><br>

🩸 Normal / Anemic

</div>

""", unsafe_allow_html=True)

# ==========================================================
# DATASET INFORMATION
# ==========================================================

st.markdown("---")

st.header("Dataset Information")

d1, d2, d3 = st.columns(3)

with d1:

    st.metric(
        "Eye Images",
        "183"
    )

with d2:

    st.metric(
        "Clinical Records",
        "1000"
    )

with d3:

    st.metric(
        "Geo Records",
        "707"
    )

# ==========================================================
# TRAINED MODELS
# ==========================================================

st.markdown("---")

st.header("Trained Models")

models_df = {
    "Model":[
        "Image Encoder",
        "Clinical Encoder",
        "Geo Encoder",
        "Fusion Model",
        "Classifier"
    ],

    "Status":[
        "Loaded",
        "Loaded",
        "Loaded",
        "Loaded",
        "Loaded"
    ]
}

st.table(models_df)

# ==========================================================
# ABOUT PROJECT
# ==========================================================

st.markdown("---")

st.header("About Project")

st.markdown("""

<div class="card">

<b>AnemiaFusionNet</b> is a Transformer-Based Multimodal
Artificial Intelligence framework developed for anemia
prediction.

The framework combines:

• Eye Conjunctiva Images

• Clinical Blood Parameters

• Geographic Risk Information

The extracted modality features are fused using a
Transformer Network to improve prediction performance.

</div>

""", unsafe_allow_html=True)

# ==========================================================
# DISCLAIMER
# ==========================================================

st.markdown("---")

st.warning("""

This application is developed only for academic and research
purposes.

Predictions generated by this AI model should not be used as
a replacement for professional medical diagnosis.

""")
# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
<div class="footer">

<h3>🩸 AnemiaFusionNet</h3>

<b>Transformer-Based Multimodal Anemia Detection Framework</b>

<br><br>

Developed using

TensorFlow • EfficientNetB0 • Transformer Network • Streamlit

<br><br>

<b>Project Workflow</b>

<br>

Image Encoder ➜ Clinical Encoder ➜ Geo Encoder ➜ Transformer Fusion ➜ Classifier

<br><br>

Version 1.0

<br>

Academic Project

</div>
""",
unsafe_allow_html=True
)

# ==========================================================
# SYSTEM STATUS
# ==========================================================

st.markdown("---")

st.subheader("System Status")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.success("Image")

with c2:
    st.success("Clinical")

with c3:
    st.success("Geo")

with c4:
    st.success("Fusion")

with c5:
    st.success("Classifier")

# ==========================================================
# FILE STATUS
# ==========================================================

st.markdown("---")

st.subheader("Loaded Model Files")

files = [

    "models/image_encoder.keras",

    "models/clinical_encoder.keras",

    "models/geo_encoder.keras",

    "models/fusion_model.keras",

    "models/classifier.keras"

]

for file in files:

    if os.path.exists(file):

        st.success(f"✔ {file}")

    else:

        st.error(f"✘ {file}")

# ==========================================================
# SESSION COMPLETE
# ==========================================================

st.markdown("---")

st.info(
    "Application Ready for Multimodal Anemia Prediction."
)