from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

from utils.preprocessing import load_mapping, preprocess_image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "handwritten_character_cnn.keras"
MAPPING_PATH = BASE_DIR / "emnist-balanced-mapping.txt"

st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #17211b; --muted: #627067; --paper: #f7f8f3; --line: #dfe6db; --leaf: #2d6a4f; --coral: #e76f51; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    .stApp { background: radial-gradient(circle at 10% 0%, rgba(216,243,220,.75), transparent 27rem), radial-gradient(circle at 96% 16%, rgba(255,218,193,.5), transparent 23rem), var(--paper); }
    [data-testid="stSidebar"] { background: #173b2b; border-right: 0; }
    [data-testid="stSidebar"] * { color: #f2f7ef; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #c7d8ca; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.16); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; color: var(--ink); }
    h1 { font-size: clamp(2.2rem, 5vw, 4.5rem) !important; line-height: 1.02 !important; }
    .block-container { padding-top: 3.2rem; padding-bottom: 4rem; max-width: 1280px; }
    .hero-kicker { color: var(--coral); font-size: .76rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; margin-bottom: .8rem; }
    .hero-copy { color: var(--muted); font-size: 1.12rem; line-height: 1.65; max-width: 42rem; }
    .quote-strip { border-left: 4px solid var(--coral); color: var(--leaf); font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; padding: .7rem 1rem; margin: 1.6rem 0 2rem; background: rgba(255,255,255,.46); }
    .feature-panel, .workflow-panel { background: rgba(255,255,252,.86); border: 1px solid var(--line); border-radius: 14px; box-shadow: 0 18px 45px rgba(35,61,43,.08); padding: 1.35rem 1.5rem; }
    .workflow-panel { background: linear-gradient(145deg, #eff8ed, #fffaf3); }
    .workflow-panel ul { color: var(--muted); line-height: 2; padding-left: 1.15rem; }
    .workflow-panel li::marker { color: var(--coral); }
    [data-testid="stFileUploader"] { background: rgba(255,255,255,.62); border: 1px dashed #a9bca9; border-radius: 12px; padding: .4rem; }
    .stButton > button[kind="primary"] { background: var(--coral); border: 0; border-radius: 8px; box-shadow: 0 8px 18px rgba(231,111,81,.24); font-weight: 700; }
    .stButton > button[kind="primary"]:hover { background: #d85e42; }
    [data-testid="stMetric"] { background: white; border: 1px solid var(--line); border-radius: 12px; padding: 1rem; }
    [data-testid="stProgressBar"] > div > div { background: var(--leaf); }
    .stAlert { border-radius: 10px; }
    .dashboard-header {
        background: linear-gradient(135deg, #2d6a4f 0%, #173b2b 100%);
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 12px 28px rgba(45, 106, 79, 0.2);
    }
    .dashboard-title {
        color: #f7f8f3;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -1px;
    }
    .dashboard-subtitle {
        color: #e76f51;
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin: 0.5rem 0 1.5rem 0;
    }
    .dashboard-description {
        color: #c7d8ca;
        font-size: 1.1rem;
        line-height: 1.6;
        margin: 0;
    }
    .stats-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        margin-top: 2rem;
    }
    .stat-card {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .stat-value {
        color: #e76f51;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .stat-label {
        color: #c7d8ca;
        font-size: 0.9rem;
    }
    @media (prefers-reduced-motion: no-preference) { .block-container { animation: rise-in .55s ease-out both; } @keyframes rise-in { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_and_mapping() -> Tuple[tf.keras.Model, Dict[int, str]]:
    """Load the trained CNN model and EMNIST mapping file."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(f"Mapping file not found: {MAPPING_PATH}")

    try:
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    except (TypeError, ValueError) as exc:
        if "quantization_config" not in str(exc):
            raise

        # Older Keras exports may contain this optional field even when unset.
        compatible_file = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
        compatible_path = Path(compatible_file.name)
        compatible_file.close()
        try:
            with zipfile.ZipFile(MODEL_PATH, "r") as source:
                with zipfile.ZipFile(compatible_path, "w") as target:
                    for entry in source.infolist():
                        contents = source.read(entry.filename)
                        if entry.filename == "config.json":
                            config = json.loads(contents.decode("utf-8"))
                            contents = json.dumps(
                                _remove_quantization_config(config)
                            ).encode("utf-8")
                        target.writestr(entry, contents)
            model = tf.keras.models.load_model(compatible_path, compile=False)
        finally:
            compatible_path.unlink(missing_ok=True)
    mapping = load_mapping(MAPPING_PATH)
    return model, mapping


def _remove_quantization_config(value: Any) -> Any:
    """Remove an unset export field that older Keras cannot deserialize."""
    if isinstance(value, dict):
        return {
            key: _remove_quantization_config(item)
            for key, item in value.items()
            if key != "quantization_config"
        }
    if isinstance(value, list):
        return [_remove_quantization_config(item) for item in value]
    return value


def render_intro_page() -> None:
    """Display the project introduction page with professional dashboard."""
    st.markdown(
        """
        <div class="dashboard-header">
            <p class="dashboard-subtitle">✍️ AI-Powered Recognition System</p>
            <h1 class="dashboard-title">Handwritten Character Recognition</h1>
            <p class="dashboard-description">Leverage advanced deep learning to convert handwritten characters into digital predictions with high accuracy.</p>
            <div class="stats-container">
                <div class="stat-card">
                    <div class="stat-value">47</div>
                    <div class="stat-label">Character Classes</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">CNN</div>
                    <div class="stat-label">Model Architecture</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">28×28</div>
                    <div class="stat-label">Input Resolution</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div class="feature-panel"><h3>🎯 How it works</h3><p class="hero-copy">Upload a handwritten character or draw directly on the canvas. The CNN model processes the image through normalization and feature extraction to predict the most likely character with confidence scoring.</p></div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="feature-panel"><h3>⚡ Key Features</h3><ul><li>Real-time character recognition</li><li>Confidence scoring</li><li>Top-3 predictions</li><li>Instant visual feedback</li></ul></div>',
            unsafe_allow_html=True,
        )

    if st.button("🚀 Start Recognition", use_container_width=True, type="primary"):
        st.session_state["current_page"] = "recognition"
        st.rerun()


def prepare_canvas_image(canvas_result: Any) -> Optional[Image.Image]:
    """Convert the drawable canvas output into a usable PIL image."""
    if canvas_result is None or canvas_result.image_data is None:
        return None

    image_data = canvas_result.image_data
    if image_data.size == 0:
        return None

    if image_data.shape[-1] == 4:
        rgb = image_data[:, :, :3]
    else:
        rgb = image_data

    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    return ImageOps.grayscale(image)


def show_prediction(image: Image.Image) -> None:
    """Run prediction for the provided image and render the prediction UI."""
    try:
        model, mapping = load_model_and_mapping()
        processed = preprocess_image(image)
        probabilities = model.predict(processed, verbose=0)[0]
        top_indices = np.argsort(probabilities)[::-1][:3]

        top_predictions: List[Dict[str, Any]] = []
        for idx in top_indices:
            class_id = int(idx)
            char = mapping.get(class_id, str(class_id))
            top_predictions.append(
                {
                    "class_id": class_id,
                    "character": char,
                    "confidence": round(float(probabilities[class_id] * 100), 2),
                }
            )

        best_prediction = top_predictions[0]
        st.success(f"Predicted Character: {best_prediction['character']}")

        confidence = best_prediction["confidence"]
        st.subheader("Confidence")
        st.metric(label="Confidence", value=f"{confidence:.2f}%")
        st.progress(value=confidence / 100)

        st.subheader("Top 3 Predictions")
        prediction_df = pd.DataFrame(top_predictions)
        st.dataframe(prediction_df, hide_index=True, use_container_width=True)

        st.subheader("Processed 28x28 Image Preview")
        processed_image = processed[0, :, :, 0]
        st.image(processed_image, clamp=True, caption="Processed 28x28 grayscale input")

    except FileNotFoundError as exc:
        st.error(str(exc))
    except ValueError as exc:
        st.error(f"Invalid image: {exc}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


def render_recognition_page() -> None:
    """Display the main character recognition page with upload and canvas modes."""
    st.title("Character Recognition")

    left_col, right_col = st.columns([1.1, 1.0])

    with left_col:
        uploaded_file = st.file_uploader(
            "Upload handwritten character image",
            type=["png", "jpg", "jpeg"],
            help="Upload a clear grayscale or RGB image of a single handwritten character.",
        )
        uploaded_image = None
        if uploaded_file is not None:
            try:
                uploaded_image = Image.open(uploaded_file).convert("RGB")
                st.image(uploaded_image, caption="Uploaded Character", use_container_width=True)
            except Exception:
                st.error("Invalid image uploaded. Please upload a valid PNG or JPG file.")

        st.markdown("---")
        st.subheader("Or draw manually")
        canvas = st_canvas(
            fill_color="rgba(255,255,255,1)",
            stroke_width=18,
            stroke_color="#000000",
            background_color="#FFFFFF",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="draw_canvas",
            display_toolbar=True,
            update_streamlit=True,
        )

        drawing_image = prepare_canvas_image(canvas)
        if drawing_image is not None:
            st.image(drawing_image, caption="Canvas drawing preview", use_container_width=True)

        prediction_image = uploaded_image or drawing_image
        if st.button("Predict", type="primary", use_container_width=True):
            if prediction_image is None:
                st.warning("Please upload an image or draw a character before predicting.")
                return
            show_prediction(prediction_image)

    with right_col:
        st.markdown(
            """
            <div style='padding: 1rem; background: linear-gradient(135deg, #ecf3ff, #f8f9fb); border-radius: 12px;'>
                <h3 style='margin-top: 0;'>Recognition Workflow</h3>
                <ul>
                    <li>Convert the input to grayscale</li>
                    <li>Resize to 28x28</li>
                    <li>Normalize pixel values</li>
                    <li>Predict with the trained CNN</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Model Information")
        st.info("Model: handwritten_character_cnn.keras\nMapping: emnist-balanced-mapping.txt")


def main() -> None:
    """Main application entry point."""
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "intro"

    st.sidebar.header("📊 Dashboard")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Project Details")
    st.sidebar.write("**Project:** Handwritten Character Recognition")
    st.sidebar.write("**Architecture:** Convolutional Neural Network")
    st.sidebar.write("**Framework:** TensorFlow + Streamlit")
    st.sidebar.write("**Dataset:** EMNIST Balanced (47 classes)")

    page = st.sidebar.radio(
        "Navigate",
        ["Project Introduction", "Character Recognition"],
        index=0 if st.session_state["current_page"] == "intro" else 1,
    )

    if page == "Project Introduction":
        st.session_state["current_page"] = "intro"
        render_intro_page()
    else:
        st.session_state["current_page"] = "recognition"
        render_recognition_page()


if __name__ == "__main__":
    main()
