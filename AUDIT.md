# Full Scientific & Technical Repository Audit (`AUDIT.md`)

**Project Title**: Reliable and Explainable Plant Disease Diagnosis and Management Using Deep Learning and Retrieval-Augmented Generation  
**Audit Date**: August 16, 2026  
**Auditor**: Senior AI/ML & Explainable AI Research Advisor  

---

## 1. Executive Summary

This repository contains an end-to-end plant disease diagnosis and management decision-support system built with **ResNet50 (Transfer Learning)**, **Grad-CAM visual explainability**, a **Retrieval-Augmented Generation (RAG) agronomy engine**, an interactive **Agri-Chatbot**, and a **PDF report generator** deployed via **Streamlit**.

While the application is functional and well-structured for deployment, a scientific audit reveals several critical methodological gaps, data integrity concerns, and conceptual oversights that must be addressed to elevate this into a **research-grade final year project and paper**.

---

## 2. Component-by-Component Audit Findings

| Component | Current Implementation | Working Status | Scientific / Methodological Critique | Severity |
| :--- | :--- | :--- | :--- | :--- |
| **Model Architecture** | ResNet50 (Keras ImageNet base + dense head 256 + dropout 0.4 + softmax 38) loaded via `resnet_weights.npz` | 🟢 Functional | Baseline model works well for in-distribution PlantVillage images, but there is no comparative evaluation against modern lightweight backbones (e.g., EfficientNetV2). | **Medium** |
| **Data Split & Leakage** | Notebook preprocessing (`main-file.ipynb` / `resnet_workingPage.ipynb`) | 🟡 Unverified | Split details are not saved in reproducible CSV files (`data_splits/`). Data augmentation must be audited to ensure augmented samples do not leak into validation/test sets. | **HIGH** |
| **Grad-CAM Explainability** | Targets `conv5_block3_out` in ResNet50 | 🟢 Functional | Correct layer target, but currently lacks quantitative or comparative evaluation against **Grad-CAM++**. Framed only qualitatively. | **Medium** |
| **Severity Estimation** | Activation area thresholding (`severity_estimator.py`) | 🔴 Conceptually Weak | **Critical Flaw**: Claims exact infected leaf percentage based on Grad-CAM activation pixel counts. Grad-CAM measures *model attention*, not ground-truth disease segmentation. Must be re-framed as an *"Attention-Based Severity Heuristic"*. | **CRITICAL** |
| **RAG Knowledge Engine** | `rag_knowledge_base.json` + `rag_engine.py` (JSON lookup + Multi-provider LLM failover) | 🟢 Functional | Grounding works well for the 38 classes. However, RAG vs. Non-RAG hallucination rates have not been systematically evaluated. | **HIGH** |
| **Uncertainty & OOD Detection** | Raw Softmax Confidence (`np.max(predictions)`) | 🔴 Flawed | Deep CNNs are known to produce overconfident predictions (>95% confidence) on out-of-distribution (OOD) or corrupt images. Lacks confidence calibration (Temperature Scaling) and explicit low-confidence safety thresholds. | **HIGH** |
| **PDF & UI Interface** | `pdf_generator.py` (FPDF2) & `App.py` (Streamlit glassmorphism) | 🟢 Functional | Full-featured UI. Needs clear disclaimers stating "Decision-Support System, Not Medical/Agricultural Certainty". | **Low** |

---

## 3. Detailed Audit & Scientific Weaknesses

### 3.1 Data Leakage & Evaluation Rigour
* **Issue**: In `main-file.ipynb`, images were processed using ImageDataGenerator. If data augmentation was applied before train/validation splitting, synthetic variations of the same leaf could exist in both train and test sets, artificially inflating accuracy.
* **Fix Required**: Create a dedicated, deterministic split generator script (`src/dataset_split.py`) that exports fixed filepaths (`data_splits/train.csv`, `validation.csv`, `test.csv`) with seed control.

### 3.2 Severity Estimator Re-Framing
* **Issue**: In `severity_estimator.py`, calculating `affected_percentage = (active_pixels / total_pixels) * 100` from Grad-CAM heatmaps and presenting it as "Infection Severity" is scientifically unsupportable without semantic segmentation labels (e.g., IoU / Dice loss ground truth).
* **Fix Required**: Re-label this output in the UI, PDF report, and paper as an **"Attention-Based Severity Heuristic"** and document its limitations explicitly.

### 3.3 Confidence Calibration & OOD Handling
* **Issue**: The model outputs raw Softmax probabilities. Softmax is uncalibrated and prone to overconfidence on noise or unsupported leaf species.
* **Fix Required**: Implement **Temperature Scaling** to calculate Expected Calibration Error (ECE) and add an **Uncertainty Guardrail**: if `confidence < 0.60`, flag the prediction as "Uncertain - Expert Inspection Recommended".

### 3.4 Scientific Terminology & Claims
* **Issue**: The project README and initial text claimed "Automated Disease Detection" and "Care Instructions".
* **Fix Required**: Standardize academic terminology to **"Decision-Support System"**, **"Disease Identification Support"**, and **"Evidence-Grounded Management Recommendations"**.

---

## 4. Remediation & Upgrade Roadmap

1. **Fix Baseline & Split Reproducibility**: Export structured CSV splits and verify non-overlapping test data.
2. **Implement Model Comparison**: Benchmark **ResNet50 vs. EfficientNetV2** on identical test splits and compute Macro F1, Weighted F1, and Inference Latency.
3. **Refactor Severity Estimator**: Re-frame pixel intensity as an attention heuristic.
4. **Implement Confidence Calibration**: Calculate Expected Calibration Error (ECE) and reliability plots.
5. **Systematize RAG Evaluation**: Compare LLM output accuracy with and without RAG context grounding.
6. **Generate Academic Deliverables**: Produce `RESEARCH_PLAN.md`, `VIVA_PREPARATION.md`, publication-ready charts, and update `README.md`.
