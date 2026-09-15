"""
generate_models_deep_dive_guide.py
===================================
Generates a dedicated, clear, friendly PDF guide titled
"AgriVision_Deep_Dive_Models_Guide.pdf" that explains ResNet50, EfficientNetV2,
Grad-CAM, Grad-CAM++, Temperature Scaling, OOD Guardrails, and RAG in detail:
  - What each feature is
  - Why we chose it & What problem it solves
  - How we used it in AgriVision AI (code files & parameters)
  - What profit/advantage it gives us
"""

import os
from fpdf import FPDF
from datetime import datetime

def sanitize_pdf_text(text: str) -> str:
    """Replaces Unicode characters with Latin-1 safe equivalents for FPDF."""
    if not isinstance(text, str):
        return str(text)
    replacements = {
        "—": "-", "–": "-", "“": '"', "”": '"', "‘": "'", "’": "'",
        "…": "...", "°": " deg ", "•": "*", "→": "->", "≥": ">=", "≤": "<=",
        "✅": "[OK]", "⚠️": "[Warning]", "🧪": "[Chem]", "🌿": "[Organic]",
        "🛡️": "[Prevent]", "🔍": "[Inspect]", "💡": "[Tip]", "🧠": "[AI]",
        "📊": "[Stats]", "💬": "[Chat]", "📄": "[PDF]", "🌾": "[Crop]"
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text.encode("latin-1", "ignore").decode("latin-1")

class ModelsGuidePDF(FPDF):
    def header(self):
        self.set_fill_color(16, 44, 34)  # Deep Agritech Emerald
        self.rect(0, 0, 210, 22, 'F')
        
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 4)
        self.cell(190, 7, sanitize_pdf_text("AGRIVISION AI: DEEP DIVE INTO MODELS & TECHNICAL FEATURES"), align="C")
        
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(167, 243, 208)
        self.set_xy(10, 12)
        self.cell(190, 5, sanitize_pdf_text("Clear Explanation of ResNet50, EfficientNetV2, Grad-CAM++, Calibration & RAG"), align="C")
        self.ln(8)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, sanitize_pdf_text(f"AgriVision AI Deep Dive Models Guide | Page {self.page_no()} of {{nb}} | Group 113"), align="C")

def build_models_guide_pdf(output_filename="AgriVision_Deep_Dive_Models_Guide.pdf"):
    pdf = ModelsGuidePDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_margins(12, 26, 12)
    pdf.set_auto_page_break(auto=True, margin=16)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 1: RESNET50 & EFFICIENTNETV2-B0 EXPLAINED
    # ──────────────────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(16, 44, 34)
    pdf.cell(0, 8, sanitize_pdf_text("1. ResNet50 & EfficientNetV2-B0 (Our AI Brains)"), ln=True)
    pdf.set_draw_color(52, 211, 153)
    pdf.line(12, 34, 198, 34)
    pdf.ln(4)

    # ResNet50 Box
    pdf.set_fill_color(240, 253, 244)
    pdf.set_draw_color(187, 247, 208)
    pdf.rect(12, pdf.get_y(), 186, 54, 'DF')
    pdf.set_xy(15, pdf.get_y() + 2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(20, 83, 45)
    pdf.cell(0, 5, sanitize_pdf_text("A. ResNet50 (Residual Network - 50 Layers)"), ln=True)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(30, 41, 59)
    resnet_text = (
        "* What it is: A famous 50-layer deep neural network invented by Microsoft Research.\n"
        "* The Problem it Solves: When neural networks get very deep (30+ layers), training signals disappear (Vanishing Gradient Problem).\n"
        "* The Solution (Residual Skip Connections): ResNet uses shortcut connections that pass data directly past layers [F(x) + x]. "
        "This allows signals to flow freely without fading away during training!\n"
        "* How We Used It in AgriVision AI:\n"
        "  - Loaded ImageNet pre-trained weights.\n"
        "  - Added Global Average Pooling (GAP) + Dense(256, ReLU) + Dropout(0.4) + Dense(38, Softmax).\n"
        "  - Fine-tuned on 37,998 PlantVillage training images (saved in resnet_weights.npz).\n"
        "* PROFIT / ADVANTAGE: Heavyweight accuracy champion achieving 94.87% Top-1 Accuracy (Macro F1: 0.9335)."
    )
    pdf.set_x(15)
    pdf.multi_cell(180, 3.8, sanitize_pdf_text(resnet_text))
    pdf.ln(6)

    # EfficientNetV2-B0 Box
    pdf.set_fill_color(240, 253, 244)
    pdf.set_draw_color(187, 247, 208)
    pdf.rect(12, pdf.get_y(), 186, 54, 'DF')
    pdf.set_xy(15, pdf.get_y() + 2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(20, 83, 45)
    pdf.cell(0, 5, sanitize_pdf_text("B. EfficientNetV2-B0 (Lightweight NAS Network)"), ln=True)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(30, 41, 59)
    effnet_text = (
        "* What it is: A next-generation lightweight neural network designed by Google using Neural Architecture Search (NAS).\n"
        "* The Problem it Solves: ResNet50 has 25.6 Million parameters (large 96MB file size), which can be slow on low-cost smartphones.\n"
        "* The Solution (Compound Scaling): EfficientNet automatically balances network depth, width, and image resolution using search algorithms.\n"
        "* How We Used It in AgriVision AI:\n"
        "  - Built & trained via src/models/train_efficientnet.py on PlantVillage split.\n"
        "  - Saved fine-tuned weights to results/week2/efficientnetv2_weights.weights.h5 (28 MB).\n"
        "* PROFIT / ADVANTAGE:\n"
        "  - Achieves 92.20% Top-1 Accuracy with only 5.9 Million parameters (77% smaller than ResNet50!).\n"
        "  - Runs at 15.22 ms/img latency (2x faster inference for mobile field deployment)."
    )
    pdf.set_x(15)
    pdf.multi_cell(180, 3.8, sanitize_pdf_text(effnet_text))

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 2: GRAD-CAM & GRAD-CAM++ EXPLAINABILITY
    # ──────────────────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(16, 44, 34)
    pdf.cell(0, 8, sanitize_pdf_text("2. Grad-CAM & Grad-CAM++ (Visual Focus Heatmaps)"), ln=True)
    pdf.set_draw_color(52, 211, 153)
    pdf.line(12, 34, 198, 34)
    pdf.ln(4)

    # Grad-CAM Box
    pdf.set_fill_color(240, 253, 244)
    pdf.set_draw_color(187, 247, 208)
    pdf.rect(12, pdf.get_y(), 186, 52, 'DF')
    pdf.set_xy(15, pdf.get_y() + 2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(20, 83, 45)
    pdf.cell(0, 5, sanitize_pdf_text("A. Standard Grad-CAM (Gradient-Weighted Class Activation Mapping)"), ln=True)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(30, 41, 59)
    gcam_text = (
        "* What it is: A visual explainability algorithm that generates glowing color heatmaps over an image.\n"
        "* The Problem it Solves: Neural networks are black boxes. Without Grad-CAM, you don't know if the AI looked at disease spots or background grass.\n"
        "* How It Works in AgriVision AI:\n"
        "  - Extracts feature maps A^k from the final convolutional layer (conv5_block3_out in ResNet50).\n"
        "  - Calculates 1st-order gradients of predicted class score y^c w.r.t feature maps.\n"
        "  - Computes channel weights alpha_k and overlays a red/yellow heatmap (where red = highest AI visual focus).\n"
        "* PROFIT / ADVANTAGE: Provides visual proof to agronomists and farmers that the diagnosis is based on real leaf lesions."
    )
    pdf.set_x(15)
    pdf.multi_cell(180, 3.8, sanitize_pdf_text(gcam_text))
    pdf.ln(6)

    # Grad-CAM++ Box
    pdf.set_fill_color(240, 253, 244)
    pdf.set_draw_color(187, 247, 208)
    pdf.rect(12, pdf.get_y(), 186, 54, 'DF')
    pdf.set_xy(15, pdf.get_y() + 2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(20, 83, 45)
    pdf.cell(0, 5, sanitize_pdf_text("B. Grad-CAM++ (Generalized Grad-CAM for Multi-Lesion Focus)"), ln=True)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(30, 41, 59)
    gcampp_text = (
        "* What it is: An advanced extension of Grad-CAM designed for multi-spot visual attribution.\n"
        "* The Problem it Solves: Standard Grad-CAM often highlights only ONE big central spot. If a leaf has 10 small scattered spots, standard Grad-CAM misses them!\n"
        "* How It Works in AgriVision AI:\n"
        "  - Implemented in src/explainability/gradcam_pp.py using nested gradient tapes.\n"
        "  - Calculates 2nd and 3rd order partial gradients w.r.t feature maps.\n"
        "  - Weights positive first-order gradients using alpha_ij coefficients.\n"
        "* PROFIT / ADVANTAGE: Accurately highlights multiple non-contiguous lesion spots across the leaf blade with sharp boundary resolution (results/week3/xai_comparison.png)."
    )
    pdf.set_x(15)
    pdf.multi_cell(180, 3.8, sanitize_pdf_text(gcampp_text))

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 3: CALIBRATION & OOD GUARDRAILS EXPLAINED
    # ──────────────────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(16, 44, 34)
    pdf.cell(0, 8, sanitize_pdf_text("3. Temperature Scaling & OOD Safety Guardrails"), ln=True)
    pdf.set_draw_color(52, 211, 153)
    pdf.line(12, 34, 198, 34)
    pdf.ln(4)

    # Temperature Scaling Box
    pdf.set_fill_color(240, 253, 244)
    pdf.set_draw_color(187, 247, 208)
    pdf.rect(12, pdf.get_y(), 186, 54, 'DF')
    pdf.set_xy(15, pdf.get_y() + 2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(20, 83, 45)
    pdf.cell(0, 5, sanitize_pdf_text("A. Temperature Scaling Calibration (T = 1.1959)"), ln=True)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(30, 41, 59)
    calib_text = (
        "* What it is: A post-hoc mathematical calibration algorithm (src/calibration/temperature_scaling.py).\n"
        "* The Problem it Solves: Raw neural networks are overconfident. They output 99.8% probability even when wrong!\n"
        "* How It Works in AgriVision AI:\n"
        "  - Divides un-normalized output logits z_i by a scalar temperature T > 0 before Softmax: p_hat_i = Softmax( z_i / T ).\n"
        "  - Fitted optimal T = 1.1959 on 8,161 validation samples by minimizing log-likelihood loss.\n"
        "* PROFIT / ADVANTAGE:\n"
        "  - Drops Expected Calibration Error (ECE) from 1.09% down to 0.42% (61.61% relative calibration error drop!).\n"
        "  - Model confidence scores now strictly reflect real-world diagnostic accuracy."
    )
    pdf.set_x(15)
    pdf.multi_cell(180, 3.8, sanitize_pdf_text(calib_text))
    pdf.ln(6)

    # OOD Guardrail Box
    pdf.set_fill_color(240, 253, 244)
    pdf.set_draw_color(187, 247, 208)
    pdf.rect(12, pdf.get_y(), 186, 54, 'DF')
    pdf.set_xy(15, pdf.get_y() + 2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(20, 83, 45)
    pdf.cell(0, 5, sanitize_pdf_text("B. Uncertainty-Aware OOD Safety Guardrail (tau = 0.60)"), ln=True)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(30, 41, 59)
    ood_text = (
        "* What it is: A safety rejection filter that evaluates calibrated confidence scores.\n"
        "* The Problem it Solves: If a user uploads a photo of a hand, tractor, soil, or super blurry leaf, a raw model picks one of 38 plant classes.\n"
        "* How It Works in AgriVision AI:\n"
        "  - Rejection Rule: If max calibrated confidence < 0.60, the prediction is REJECTED as UNCERTAIN.\n"
        "  - Tested on 30 non-leaf images (Tier 2) and 30 blurred leaves (Tier 3) in results/week1/ood_rejection_results.json.\n"
        "* PROFIT / ADVANTAGE: Displays a prominent warning banner and instructs the AI agronomist to issue an expert verification disclaimer."
    )
    pdf.set_x(15)
    pdf.multi_cell(180, 3.8, sanitize_pdf_text(ood_text))

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 4: KNOWLEDGE-GROUNDED RAG & APP INTERFACE EXPLAINED
    # ──────────────────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(16, 44, 34)
    pdf.cell(0, 8, sanitize_pdf_text("4. Knowledge-Grounded RAG Engine & Full Stack App"), ln=True)
    pdf.set_draw_color(52, 211, 153)
    pdf.line(12, 34, 198, 34)
    pdf.ln(4)

    # RAG Box
    pdf.set_fill_color(240, 253, 244)
    pdf.set_draw_color(187, 247, 208)
    pdf.rect(12, pdf.get_y(), 186, 56, 'DF')
    pdf.set_xy(15, pdf.get_y() + 2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(20, 83, 45)
    pdf.cell(0, 5, sanitize_pdf_text("A. Knowledge-Grounded RAG (Retrieval-Augmented Generation)"), ln=True)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(30, 41, 59)
    rag_text = (
        "* What it is: A framework that combines structured database retrieval with LLM generation (rag_engine.py).\n"
        "* The Problem it Solves: Direct LLMs (like ungrounded ChatGPT) hallucinate unverified chemical dosages, creating agricultural hazards.\n"
        "* How It Works in AgriVision AI:\n"
        "  - Built a 38-class certified Agronomy Database (rag_knowledge_base.json) containing symptoms, chemical active ingredients, organic remedies, and prevention.\n"
        "  - When a disease is diagnosed, retrieve_disease_context() fetches exact reference data and injects it into LLMs (Qwen2.5 / Mistral-7B).\n"
        "* PROFIT / ADVANTAGE:\n"
        "  - Grounded RAG achieved 5.00/5.0 Likert score with 0.0% chemical dosage hallucinations!\n"
        "  - Direct LLMs exhibited a 38.9% chemical dosage hallucination rate (src/evaluation/human_rating_pass.py)."
    )
    pdf.set_x(15)
    pdf.multi_cell(180, 3.8, sanitize_pdf_text(rag_text))
    pdf.ln(6)

    # Full Stack Box
    pdf.set_fill_color(240, 253, 244)
    pdf.set_draw_color(187, 247, 208)
    pdf.rect(12, pdf.get_y(), 186, 52, 'DF')
    pdf.set_xy(15, pdf.get_y() + 2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(20, 83, 45)
    pdf.cell(0, 5, sanitize_pdf_text("B. Streamlit Web App, Voice Utilities & PDF Generator"), ln=True)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(30, 41, 59)
    app_text = (
        "* Streamlit Dashboard (App.py):\n"
        "  - Custom Glassmorphism CSS design (#0b1d16 gradient theme).\n"
        "  - Sidebar radio buttons for dual model selection (ResNet50 vs EfficientNetV2) and dual XAI selection (Grad-CAM vs Grad-CAM++).\n"
        "  - Batch processing gallery calculating field-level crop health scores.\n"
        "* Voice Utilities (voice_utils.py): Uses Web Speech API, OpenAI Whisper, and gTTS for Hindi & English voice interaction.\n"
        "* PDF Extension Generator (pdf_generator.py): Generates printable diagnostic health cards via FPDF2.\n"
        "* Verification Gate (verify_claims.py): Audits all 85 numeric claims in paper against JSON files (100% verified zero-fabrication science)."
    )
    pdf.set_x(15)
    pdf.multi_cell(180, 3.8, sanitize_pdf_text(app_text))

    output_path = os.path.join(os.getcwd(), output_filename)
    pdf.output(output_path)
    print(f"Models Deep Dive Guide PDF generated successfully: {output_path}")
    return output_path

if __name__ == "__main__":
    build_models_guide_pdf()
