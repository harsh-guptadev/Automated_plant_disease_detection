# Reliable & Explainable Plant Disease Diagnosis & Management System (AgriVision AI)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://automated-plant-disease-detection.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An explainable, uncertainty-aware, and Retrieval-Augmented Generation (RAG) decision-support system for automated plant disease identification and evidence-grounded crop care recommendations.

---

## 🔬 Research Overview & Problem Statement

Plant diseases threaten global agricultural yields, food security, and smallholder farmer livelihoods. While Deep Convolutional Neural Networks (CNNs) achieve impressive accuracy on benchmark leaf image datasets, traditional black-box classifiers exhibit critical limitations in real-world agricultural deployment:

1. **Lack of Interpretability**: Visual predictions lack visual justification for agronomists.
2. **Uncalibrated Overconfidence**: Standard Softmax networks produce high confidence scores even on corrupted, noisy, or out-of-distribution (OOD) images.
3. **LLM Hallucinations**: Generative AI models often fabricate unverified chemical dosages or pesticide names when asked for plant care advice.

### Primary Research Framing
> *"An Explainable and Retrieval-Augmented AI Decision-Support System for Plant Disease Diagnosis and Management."*

*Note: This system provides clinical decision support and verified agronomic protocols—it does not claim guaranteed cures or medical/agricultural certainty.*

---

## 🏗️ System Architecture & Workflow

```text
               ┌──────────────────────────────┐
               │    Input Leaf Image (224x224)│
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │ Image Preprocessing & Norm   │
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │  ResNet50 Backbone (GAP Head)│
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │ 38-Class Softmax Prediction  │
               └──────────────┬───────────────┘
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
┌──────────────────────────┐      ┌──────────────────────────┐
│ Grad-CAM Feature Map     │      │ Confidence & Reliability │
│ (Layer: conv5_block3_out)│      │ Threshold Check (>0.60)  │
└────────────┬─────────────┘      └────────────┬─────────────┘
             │                                 │
             ▼                                 ▼
┌──────────────────────────┐      ┌──────────────────────────┐
│ Attention Severity Area  │      │ RAG Knowledge Retrieval  │
│ (Heuristic Indicator)    │      │ (38-Class Agronomic DB)  │
└────────────┬─────────────┘      └────────────┬─────────────┘
             │                                 │
             └────────────────┬────────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │ Context-Grounded LLM Engine  │
               │ (Qwen2.5 / Mistral-7B)       │
               └──────────────┬───────────────┘
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
┌──────────────────────────┐      ┌──────────────────────────┐
│ Streamlit Decision UI    │      │ Printable PDF Report     │
│ (Chatbot & Multi-Lang)   │      │ (Executive Diagnostic Card)│
└──────────────────────────┘      └──────────────────────────┘
```

---

## ✨ Key Features & Capabilities

- 🌿 **38-Category Plant Diagnosis**: Classifies healthy and diseased leaves across Apple, Corn, Grape, Potato, Tomato, Strawberry, Cherry, Peach, Pepper, and Blueberry crops.
- 🔍 **Grad-CAM Visual Explainability**: Renders spatial heatmap overlays targeting layer `conv5_block3_out` to highlight leaf regions driving model classification.
- 📊 **Attention-Based Severity Heuristic**: Computes spatial feature activation area percentages with built-in uncertainty guardrails.
- 📚 **RAG Agronomist Engine**: Retrieves verified symptoms, chemical dosages, organic remedies, and prevention strategies from `rag_knowledge_base.json` to ground LLM care plans.
- 💬 **Interactive Agri-Chatbot**: Farmers can ask follow-up questions regarding chemical safety, organic alternatives, or application timing.
- 📄 **Downloadable PDF Health Cards**: Exports an official diagnostic card containing model confidence, severity metrics, and care protocols.
- 🌍 **Multi-Language Support**: Fully compatible with English, Hindi, Spanish, and French.

---

## 📁 Repository Structure

```text
.
├── AUDIT.md                        # Scientific audit of code, methodology, & claims
├── RESEARCH_PLAN.md                # Research question, hypotheses, & experiment design
├── VIVA_PREPARATION.md             # Oral examination & defense prep guide
├── App.py                          # Streamlit web application
├── pdf_generator.py                # FPDF2 PDF report generator
├── rag_engine.py                   # RAG context retriever & multi-model LLM engine
├── rag_knowledge_base.json         # 38-class verified agronomic knowledge base
├── severity_estimator.py           # Attention-based severity heuristic module
├── requirements.txt                # Python package dependencies
├── runtime.txt                     # Python 3.11 environment configuration
├── src/                            # Modular Python research source packages
│   ├── calibration/
│   ├── evaluation/
│   ├── explainability/
│   ├── models/
│   ├── preprocessing/
│   └── rag/
├── configs/                        # YAML experiment configurations
├── data_splits/                    # Structured train/val/test CSV splits
├── experiments/                    # Model run logs and artifacts
└── results/                        # Evaluation metrics and publication plots
```

---

## 💻 Local Setup & Execution

### 1. Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Key (Optional for LLM Advice)
Create a `.env` file in the root folder:
```env
HF_TOKEN=your_huggingface_api_token_here
```

### 3. Run Application Locally
```powershell
streamlit run App.py
```

---

## 📜 Research Limitations & Disclaimer

1. **Decision Support**: This application is an AI-powered decision-support tool and should be paired with expert field agronomist verification.
2. **Dataset Shift**: Images are trained on the PlantVillage dataset; field performance on heavily shadowed, blurry, or multi-leaf images may vary.
3. **Attention Heuristic**: Grad-CAM heatmap activation indicates model spatial focus—it is not a substitute for ground-truth semantic lesion segmentation.
