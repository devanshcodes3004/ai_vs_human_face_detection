import os
import time
import tempfile

import streamlit as st

from src.pipeline.prediction_pipeline import PredictionPipeline


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI vs Human Face Detection",
    page_icon="🤖",
    layout="centered"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

MODEL_PATH = "artifacts/models/mobilenet_v2.keras"

with st.spinner("Loading AI Model..."):

    predictor = PredictionPipeline(MODEL_PATH)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🤖 AI vs Human Face Detection")

st.write(
    """
Upload a face image and the model will determine whether it is **AI Generated**
or a **Real Human Face**.
"""
)

st.divider()

# --------------------------------------------------
# Upload Image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Preview
# --------------------------------------------------

if uploaded_file is not None:

    st.success("Image uploaded successfully!")

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------
    # Predict Button
    # --------------------------------------------------

    if st.button(
        "🔍 Predict",
        use_container_width=True
    ):

        # Save uploaded image temporarily

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as tmp:

            tmp.write(uploaded_file.read())

            image_path = tmp.name

        progress_text = st.empty()

        progress_bar = st.progress(0)

        # ----------------------------------------------

        with st.spinner("Analyzing Image..."):

            for percent in range(101):

                progress_bar.progress(percent)

                progress_text.text(
                    f"Analyzing Image... {percent}%"
                )

                time.sleep(0.01)

            result = predictor.predict(image_path)

        os.remove(image_path)

        progress_bar.empty()

        progress_text.empty()

        st.divider()

        st.header("Prediction Result")

        if result["prediction"] == "AI Generated":

            st.error("🚨 AI Generated Face")

        else:

            st.success("✅ Real Human Face")

        confidence = result["confidence"] * 100

        st.subheader("Confidence")

        st.progress(int(confidence))

        st.write(
            f"### {confidence:.2f}%"
        )

        st.divider()

        st.info(
            "The prediction is based on the uploaded image using a trained "
            "deep learning model."
        )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption("Developed using TensorFlow, MobileNetV2 and Streamlit")