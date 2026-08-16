import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input
import cv2
from tensorflow.keras.preprocessing import image
import os
import sys
from dotenv import load_dotenv

# Path setup for modular packages
sys.path.append("src")

# Import custom RAG & Agritech modules
from rag_engine import retrieve_disease_context, generate_rag_care_advice, chat_with_agronomist_rag
from severity_estimator import estimate_disease_severity
from pdf_generator import create_pdf_report
from voice_utils import render_voice_input, speak_response
from evaluation.batch_processor import process_batch_images
from models.efficientnet_benchmark import benchmark_single_image, get_model_specs

# Page setup
st.set_page_config(
    page_title="AgriVision AI - Plant Health & Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Modern CSS Design System
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0b1d16 0%, #11281f 40%, #0d1b15 100%);
        color: #e2e8f0;
    }
    
    .hero-container {
        background: rgba(16, 44, 34, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(52, 211, 153, 0.2);
        border-radius: 20px;
        padding: 28px 36px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6ee7b7 0%, #34d399 50%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        max-width: 800px;
        line-height: 1.5;
    }
    
    .glass-card {
        background: rgba(15, 33, 26, 0.55);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(52, 211, 153, 0.15);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    .badge-healthy {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    
    .badge-diseased {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.3);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .metric-label {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    header[data-testid="stHeader"] { display: none; }
    footer { display: none; }
    </style>
""", unsafe_allow_html=True)

# Sidebar - Settings & Features
with st.sidebar:
    st.markdown("### 🌿 AgriVision AI")
    st.caption("RAG-Powered Agritech Diagnostic System")
    st.divider()
    
    st.markdown("##### 🌍 Select Preferred Language")
    language = st.selectbox("Language", ["English", "Hindi", "Spanish", "French"], index=0)
    
    st.divider()
    st.markdown("##### 🔑 Hugging Face Token")
    user_hf_key = st.text_input("HF API Token", type="password", help="Required for RAG AI Advice & Agri-Chatbot")
    if user_hf_key:
        os.environ["HF_TOKEN"] = user_hf_key
        st.session_state["HF_TOKEN"] = user_hf_key
        
    st.divider()
    st.markdown("""
    **System Features:**
    - 🧠 **ResNet50 Classifier** (38 Classes)
    - 📚 **RAG Agronomist Engine** (Zero-Hallucination)
    - 📊 **Grad-CAM Infection Severity Estimator**
    - 💬 **Interactive Crop Health Chatbot**
    - 📄 **Downloadable PDF Diagnostic Report**
    """)

# Hero Header
st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Automated Plant Disease Detection & RAG Agronomist</div>
        <div class="hero-subtitle">
            Upload a plant leaf image for instant CNN classification, Grad-CAM infection severity estimation, RAG-certified treatment protocols, and an interactive AI agronomist chatbot.
        </div>
    </div>
""", unsafe_allow_html=True)

classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy',
           'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy' , 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
           'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
           'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
           'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
           'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
           'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
           'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
           'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

class_descriptions = {
    'Apple___Apple_scab': "Apple plant affected by Apple scab",
    'Apple___Black_rot': "Apple plant affected by Black rot",
    'Apple___Cedar_apple_rust': "Apple plant affected by Cedar apple rust",
    'Apple___healthy': "Apple plant — healthy and disease-free",
    'Blueberry___healthy': "Blueberry plant — healthy and disease-free",
    'Cherry_(including_sour)___Powdery_mildew': "Cherry plant affected by Powdery mildew",
    'Cherry_(including_sour)___healthy': "Cherry plant — healthy and disease-free",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Corn plant affected by Gray leaf spot",
    'Corn_(maize)___Common_rust_': "Corn plant affected by Common rust",
    'Corn_(maize)___Northern_Leaf_Blight': "Corn plant affected by Northern Leaf Blight",
    'Corn_(maize)___healthy': "Corn plant — healthy and disease-free",
    'Grape___Black_rot': "Grape plant affected by Black rot",
    'Grape___Esca_(Black_Measles)': "Grape plant affected by Esca (Black Measles)",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Grape plant affected by Leaf blight (Isariopsis Leaf Spot)",
    'Grape___healthy': "Grape plant — healthy and disease-free",
    'Orange___Haunglongbing_(Citrus_greening)': "Orange plant affected by Huanglongbing (Citrus greening)",
    'Peach___Bacterial_spot': "Peach plant affected by Bacterial spot",
    'Peach___healthy': "Peach plant — healthy and disease-free",
    'Pepper,_bell___Bacterial_spot': "Bell pepper plant affected by Bacterial spot",
    'Pepper,_bell___healthy': "Bell pepper plant — healthy and disease-free",
    'Potato___Early_blight': "Potato plant affected by Early blight",
    'Potato___Late_blight': "Potato plant affected by Late blight",
    'Potato___healthy': "Potato plant — healthy and disease-free",
    'Raspberry___healthy': "Raspberry plant — healthy and disease-free",
    'Soybean___healthy': "Soybean plant — healthy and disease-free",
    'Squash___Powdery_mildew': "Squash plant affected by Powdery mildew",
    'Strawberry___Leaf_scorch': "Strawberry plant affected by Leaf scorch",
    'Strawberry___healthy': "Strawberry plant — healthy and disease-free",
    'Tomato___Bacterial_spot': "Tomato plant affected by Bacterial spot",
    'Tomato___Early_blight': "Tomato plant affected by Early blight",
    'Tomato___Late_blight': "Tomato plant affected by Late blight",
    'Tomato___Leaf_Mold': "Tomato plant affected by Leaf Mold",
    'Tomato___Septoria_leaf_spot': "Tomato plant affected by Septoria leaf spot",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Tomato plant affected by Two-spotted spider mite",
    'Tomato___Target_Spot': "Tomato plant affected by Target Spot",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Tomato plant affected by Tomato Yellow Leaf Curl Virus",
    'Tomato___Tomato_mosaic_virus': "Tomato plant affected by Tomato mosaic virus",
    'Tomato___healthy': "Tomato plant — healthy and disease-free"
}

@st.cache_resource
def load_model():
    inputs = Input(shape=(224, 224, 3))
    resnet_base = ResNet50(weights='imagenet', include_top=False, input_tensor=(inputs))
    resnet_base.trainable = False
    
    x = resnet_base.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation='relu', name="head_dense")(x)
    x = layers.Dropout(0.4, name="head_dropout")(x)
    outputs = layers.Dense(len(classes), activation='softmax', name="predictions")(x)

    model = models.Model(inputs, outputs)
    weights = np.load("resnet_weights.npz", allow_pickle=True)
    model.set_weights([weights[key] for key in weights])
    return model

model = load_model()

# Image Upload Card
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
upload_files = st.file_uploader(
    "📷 Select or drop plant leaf image(s) for instant single or batch field-level diagnosis",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True
)
st.markdown('</div>', unsafe_allow_html=True)

# Helper function for Grad-CAM Heatmap calculation
def make_gradcam_heatmap(img_array, model, last_conv_layer_name='conv5_block3_out'):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap

if upload_files and len(upload_files) > 1:
    # ──────────────────────────────────────────────────────────────────────────
    # BATCH / MULTI-IMAGE FIELD DIAGNOSTIC DASHBOARD
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("### 🌾 Batch Field Diagnostic Dashboard")
    st.caption("Field-level crop health analytics generated across multiple leaf inspection samples.")

    with st.spinner("Processing batch field diagnostic samples..."):
        summary_stats, results_list, df_export = process_batch_images(
            upload_files, model, classes, class_descriptions, estimate_disease_severity, make_gradcam_heatmap
        )

    # Metric summary row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'''
            <div class="glass-card">
                <div class="metric-label">Total Leaves Scanned</div>
                <div class="metric-value">{summary_stats["total_scanned"]}</div>
            </div>
        ''', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''
            <div class="glass-card">
                <div class="metric-label">Healthy Leaves</div>
                <div class="metric-value" style="color:#34d399;">{summary_stats["healthy_count"]}</div>
            </div>
        ''', unsafe_allow_html=True)
    with c3:
        st.markdown(f'''
            <div class="glass-card">
                <div class="metric-label">Infected Leaves</div>
                <div class="metric-value" style="color:#f87171;">{summary_stats["infected_count"]}</div>
            </div>
        ''', unsafe_allow_html=True)
    with c4:
        st.markdown(f'''
            <div class="glass-card">
                <div class="metric-label">Field Health Score</div>
                <div class="metric-value" style="color:#6ee7b7;">{summary_stats["field_health_score"]}%</div>
            </div>
        ''', unsafe_allow_html=True)

    st.markdown("#### 📋 Field Inspection Log")
    st.dataframe(df_export, use_container_width=True)

    # CSV Download Button
    csv_bytes = df_export.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Field Inspection Log (CSV)",
        data=csv_bytes,
        file_name="Field_Inspection_Log.csv",
        mime="text/csv"
    )

elif upload_files and len(upload_files) == 1:
    # ──────────────────────────────────────────────────────────────────────────
    # SINGLE LEAF DIAGNOSTIC & BENCHMARKING DASHBOARD
    # ──────────────────────────────────────────────────────────────────────────
    upload_file = upload_files[0]
    image = Image.open(upload_file).convert("RGB")
    
    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized)
    image_batch = np.expand_dims(image_array, axis=0)
    image_batch_preprocessed = preprocess_input(image_batch.copy())
    
    predictions = model.predict(image_batch_preprocessed)
    confidence = float(np.max(predictions))
    predicted_class = classes[np.argmax(predictions)]
    readable_prediction = class_descriptions.get(predicted_class, predicted_class)
    is_healthy = "healthy" in readable_prediction.lower()

    heatmap = make_gradcam_heatmap(image_batch_preprocessed, model)
    severity_metrics = estimate_disease_severity(heatmap, confidence=confidence)
    
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # Low Confidence / Uncertainty Safety Guardrail
    if confidence < 0.60:
        st.warning("⚠️ **Low Prediction Confidence**: Model confidence is below 60%. Please upload a clearer, well-lit leaf image or consult a professional agronomist before taking chemical action.")

    # Diagnostic Header Metrics
    badge_class = "badge-healthy" if is_healthy else "badge-diseased"
    status_icon = "✅" if is_healthy else "⚠️"
    status_text = "HEALTHY CROP" if is_healthy else "DISEASE IDENTIFICATION SUPPORT"

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f'''
            <div class="glass-card">
                <span class="status-badge {badge_class}">{status_icon} {status_text}</span>
                <h2 style="margin:0; color:#ffffff; font-size:1.5rem;">{readable_prediction}</h2>
            </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''
            <div class="glass-card">
                <div class="metric-label">Model Confidence Score</div>
                <div class="metric-value">{confidence * 100:.1f}%</div>
            </div>
        ''', unsafe_allow_html=True)
    with col3:
        st.markdown(f'''
            <div class="glass-card">
                <div class="metric-label">Attention Coverage Heuristic</div>
                <div class="metric-value" style="color:{severity_metrics['badge_color']};">{severity_metrics['affected_percentage']}%</div>
                <div style="font-size:0.78rem; color:#94a3b8;">{severity_metrics['severity_level']}</div>
            </div>
        ''', unsafe_allow_html=True)

    st.caption("ℹ️ *Disclaimer: Attention Coverage is an experimental Grad-CAM feature area heuristic and not a ground-truth measurement of biological leaf infection.*")

    # Knowledge Context Retrieval
    rag_data = retrieve_disease_context(predicted_class)

    # Multi-Tab Application Blueprint
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Visual & Grad-CAM Analysis",
        "📚 Knowledge-Grounded Protocol",
        "💬 Knowledge-Grounded Agri-Assistant",
        "📄 PDF Decision-Support Report",
        "📊 Model Comparison (Viva Defense)"
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.image(image, caption="Original Input Leaf Image", use_container_width=True)
        with c2:
            st.image(superimposed_img_rgb, caption="Grad-CAM Attention Heatmap (Model Focus Region)", use_container_width=True)

    with tab2:
        st.markdown("### 📚 Knowledge-Grounded Disease Management Guidelines")
        col_rag1, col_rag2 = st.columns(2)
        with col_rag1:
            st.markdown(f"**🔬 Symptoms:**\n{rag_data['symptoms']}")
            st.markdown(f"**🧪 Chemical Management:**\n{rag_data['chemical_treatment']}")
        with col_rag2:
            st.markdown(f"**🌿 Organic & Cultural Management:**\n{rag_data['organic_remedy']}")
            st.markdown(f"**🛡️ Prevention & Sanitation:**\n{rag_data['prevention']}")
            
        load_dotenv()
        HF_TOKEN = st.session_state.get("HF_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        
        if HF_TOKEN:
            st.divider()
            if st.button("🤖 Generate Knowledge-Grounded Care Protocol (LLM)", type="primary"):
                with st.spinner(f"Generating decision-support protocol in {language}..."):
                    try:
                        advice = generate_rag_care_advice(readable_prediction, predicted_class, HF_TOKEN, language)
                        st.markdown(f'''
                            <div class="glass-card">
                                <h3 style="color:#34d399; margin-top:0;">📋 Evidence-Grounded Recommendations</h3>
                                <div>{advice}</div>
                            </div>
                        ''', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Hugging Face API Authentication Error: {e}. Please double check your token permissions.")
        else:
            st.info("💡 Enter your Hugging Face API token in the sidebar to generate custom LLM decision support.")

    with tab3:
        st.markdown("### 💬 Knowledge-Grounded Agri-Assistant")
        st.caption("Ask follow-up questions about disease management, chemical safety, or organic alternatives — by **typing** or **speaking**.")

        # Voice output toggle
        enable_voice_out = st.toggle("🔊 Read AI responses aloud", value=False, help="Automatically plays TTS audio for each AI reply.")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "voice_pending" not in st.session_state:
            st.session_state.voice_pending = None

        # ── Chat history display ──────────────────────────────────────────────
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # ── Voice input widget ────────────────────────────────────────────────
        voice_transcript = render_voice_input(language=language)

        # ── Text input ────────────────────────────────────────────────────────
        user_input = st.chat_input(f"Or type your question about {readable_prediction}...")

        # Prefer voice transcript if it arrived this rerun
        if voice_transcript and isinstance(voice_transcript, str) and voice_transcript.strip():
            user_input = voice_transcript.strip()

        # ── Process question ─────────────────────────────────────────────────
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            load_dotenv()
            HF_TOKEN = st.session_state.get("HF_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
            if HF_TOKEN:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            bot_response = chat_with_agronomist_rag(
                                user_input, readable_prediction, predicted_class, HF_TOKEN, st.session_state.chat_history, language
                            )
                            st.write(bot_response)
                            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

                            # ── Voice output ──────────────────────────────
                            if enable_voice_out:
                                speak_response(bot_response, language=language)
                        except Exception as e:
                            st.error(f"API Error: {e}")
            else:
                st.error("Please provide a Hugging Face API Token in the sidebar to use the Agri-Assistant.")

    with tab4:
        st.markdown("### 📄 Export Decision-Support PDF Report")
        st.write("Generate and download an official decision-support card containing model confidence, attention metrics, and management guidelines.")
        
        if st.button("📥 Generate & Download PDF Report"):
            pdf_path = create_pdf_report(
                disease_name=readable_prediction,
                confidence_score=confidence,
                severity_info=severity_metrics,
                rag_context=rag_data,
                language=language
            )
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="💾 Save Decision-Support PDF Report",
                    data=f,
                    file_name=f"Plant_Health_Report_{predicted_class}.pdf",
                    mime="application/pdf"
                )

    with tab5:
        st.markdown("### 📊 Model Architecture Comparison & Benchmarking")
        st.caption("Academic research benchmarking comparing ResNet50 baseline against EfficientNetV2-B0.")
        
        specs = get_model_specs()
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f'''
                <div class="glass-card">
                    <h4 style="color:#6ee7b7; margin-top:0;">ResNet50 Baseline</h4>
                    <p><b>Parameters:</b> {specs["ResNet50"]["params_formatted"]}</p>
                    <p><b>Model Size:</b> {specs["ResNet50"]["model_size_mb"]} MB</p>
                    <p><b>Scaling:</b> {specs["ResNet50"]["scaling_type"]}</p>
                </div>
            ''', unsafe_allow_html=True)
        with col_m2:
            st.markdown(f'''
                <div class="glass-card">
                    <h4 style="color:#34d399; margin-top:0;">EfficientNetV2-B0 Benchmark</h4>
                    <p><b>Parameters:</b> {specs["EfficientNetV2-B0"]["params_formatted"]} <i>(~77% parameter reduction)</i></p>
                    <p><b>Model Size:</b> {specs["EfficientNetV2-B0"]["model_size_mb"]} MB</p>
                    <p><b>Scaling:</b> {specs["EfficientNetV2-B0"]["scaling_type"]}</p>
                </div>
            ''', unsafe_allow_html=True)

        st.divider()
        if st.button("⚡ Run Live Side-by-Side Inference Benchmarking"):
            with st.spinner("Running side-by-side ResNet50 vs EfficientNetV2 inference..."):
                bench_res = benchmark_single_image(image_array, model, len(classes))
                
                b1, b2 = st.columns(2)
                with b1:
                    st.markdown(f"**ResNet50 Latency:** `{bench_res['ResNet50']['latency_ms']} ms`")
                    st.markdown(f"**ResNet50 Top-1 Confidence:** `{bench_res['ResNet50']['confidence']*100:.1f}%`")
                with b2:
                    st.markdown(f"**EfficientNetV2-B0 Latency:** `{bench_res['EfficientNetV2-B0']['latency_ms']} ms`")
                    st.markdown(f"**EfficientNetV2-B0 Top-1 Confidence:** `{bench_res['EfficientNetV2-B0']['confidence']*100:.1f}%`")

                st.markdown("#### 🏆 Top-3 Probability Breakdown")
                t3_res = bench_res['ResNet50']['top3_indices']
                t3_probs_res = bench_res['ResNet50']['top3_probs']
                
                chart_data = {
                    classes[idx].replace("___", " - "): round(prob * 100, 1)
                    for idx, prob in zip(t3_res, t3_probs_res)
                }
                st.bar_chart(chart_data)

        st.info("💡 **Viva Defense Insight:** EfficientNetV2 utilizes compound scaling (jointly balancing depth, width, and resolution) along with Fused-MBConv layers, achieving higher parameter efficiency and lower memory footprint compared to traditional fixed-depth ResNet architectures.")