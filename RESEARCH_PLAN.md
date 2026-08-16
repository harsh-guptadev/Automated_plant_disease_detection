# Research Plan: Reliable and Explainable Plant Disease Diagnosis and Management

**Project Title**: Reliable and Explainable Plant Disease Diagnosis and Management Using Deep Learning and Retrieval-Augmented Generation  
**Academic Focus**: B.Tech CSE / AI-ML Major Capstone & Research Publication  

---

## 1. Research Question & Objective

### Main Research Question
> *"How reliable is a deep-learning plant disease diagnosis system under dataset shift, and can explainability and retrieval-augmented generation improve the trustworthiness and usefulness of its disease-management recommendations?"*

### Hypotheses
1. **Hypothesis H1 (Model Comparison)**: Transfer learning backbones with modern compound scaling (e.g., EfficientNetV2) achieve comparable or superior Macro F1 scores to ResNet50 while reducing parameter footprint and inference latency.
2. **Hypothesis H2 (Calibration & Uncertainty)**: Softmax confidence in deep CNNs is uncalibrated; Temperature Scaling significantly reduces Expected Calibration Error (ECE), enabling reliable low-confidence / OOD rejection.
3. **Hypothesis H3 (RAG Grounding)**: Grounding LLM recommendations with structured, peer-reviewed agronomic knowledge eliminates hallucinatory chemical dosage guidance compared to ungrounded LLMs.

---

## 2. Experimental Methodology

### A. Model Baseline & Comparison
- **Baseline**: ResNet50 (Transfer Learning, ImageNet pre-trained).
- **Benchmark Model**: EfficientNetV2-B0 / DenseNet121.
- **Evaluation Metrics**: Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1, Per-Class F1, Confusion Matrix, Parameter Count, and GPU/CPU Inference Latency.

### B. Explainability & Attention Analysis
- **Methods**: Grad-CAM (Targeting layer `conv5_block3_out` for ResNet50) and Grad-CAM++.
- **Qualitative Protocol**: Comparative analysis on correctly vs. incorrectly classified samples to verify if spatial attention aligns with biological lesion patterns.
- **Severity Heuristic**: Re-framed as an *"Attention-Based Feature Activation Area"* rather than ground-truth lesion segmentation to maintain academic validity.

### C. Confidence Calibration & OOD Safety
- **Calibration Method**: Post-hoc Temperature Scaling on validation logit outputs.
- **Metrics**: Expected Calibration Error (ECE) and Reliability Diagrams.
- **Uncertainty Guardrail**: Automated rejection state (`confidence < 0.60`) requesting clearer imagery or expert consultation.

### D. RAG Grounding & Evaluation
- **Knowledge Base**: Curated database (`rag_knowledge_base.json`) covering all 38 PlantVillage classes with symptoms, certified chemical treatments, organic remedies, and prevention strategies.
- **Comparative Pipeline**: Evaluates LLM output with vs. without RAG context for factual correctness, groundedness, and hallucination rates across standard agricultural queries.

---

## 3. Implementation Directory Architecture

```text
c:\Users\HARSH\OneDrive\Desktop\plant project\
├── AUDIT.md                        # Full technical & scientific code audit
├── RESEARCH_PLAN.md                # Research methodology & experimental design
├── VIVA_PREPARATION.md             # Academic viva defense questions & answers
├── README.md                       # Research-grade project documentation
├── App.py                          # Streamlit decision-support application
├── pdf_generator.py                # PDF health card report generator
├── rag_engine.py                   # RAG context retriever & multi-model LLM engine
├── rag_knowledge_base.json         # 38-class verified agronomic knowledge base
├── severity_estimator.py           # Attention-based severity heuristic module
├── configs/                        # YAML experiment configurations
├── data_splits/                    # Reproducible CSV train/val/test splits
├── experiments/                    # Execution logs and confusion matrix outputs
├── results/                        # CSV tables and comparison plots
└── src/                            # Modular Python research source packages
    ├── calibration/
    ├── evaluation/
    ├── explainability/
    ├── models/
    ├── preprocessing/
    └── rag/
```
