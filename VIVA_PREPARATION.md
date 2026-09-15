# Capstone Defense & Viva Preparation Guide
## Group 113: Harsh Gupta, Manthan, Sumit Kumar | Supervisor: Ms. Pooja Deswal
**Project Title**: Automated Plant Disease Detection and Knowledge-Grounded Management System (AgriVision AI)

---

## 🎯 Executive Defense Strategy

When presenting AgriVision AI to university viva examiners, emphasize that this project is framed as an **Explainable, Uncertainty-Aware Decision-Support System**.

### Core Academic Defense Pitch
> "Rather than claiming an end-to-end black-box classifier or unverified generative AI, AgriVision AI bridges deep learning computer vision with certified agricultural extension protocols through four defense layers:
> 1. **Calibrated Classification**: Fine-tuned ResNet50 (94.87% Top-1 Accuracy) and EfficientNetV2-B0 (92.20% Top-1 Accuracy, 2x faster latency).
> 2. **Post-Hoc Confidence Calibration**: Temperature scaling ($T=1.1959$) reducing Expected Calibration Error (ECE) from 1.09% to 0.42% (61.61% relative reduction).
> 3. **Uncertainty-Aware OOD Guardrail**: Rejection threshold ($\tau=0.60$) flagging low-confidence and non-leaf imagery.
> 4. **Visual & Generative Transparency**: Grad-CAM/Grad-CAM++ spatial heatmaps and RAG-grounded care protocols (5.0/5.0 Likert score vs. 2.66/5.0 for direct LLM)."

---

## 📊 Summary of Empirical Results (Audit Verified)

| Component | Metric / Value | Verification Source |
|---|---|---|
| **ResNet50 Top-1 Accuracy** | **94.87%** (8,146 test samples) | `results/week1/resnet50_test_metrics.json` |
| **ResNet50 Macro F1** | **0.9335** | `results/week1/resnet50_test_metrics.json` |
| **EfficientNetV2-B0 Accuracy** | **92.20%** (5.9M params, 15.22ms latency) | `results/week2/efficientnetv2_test_metrics.json` |
| **Uncalibrated ECE** | **1.09%** (0.0109) | `results/week1/temperature_scale.json` |
| **Calibrated ECE** | **0.42%** (0.0042) | `results/week1/temperature_scale.json` |
| **OOD In-Distribution Accept** | **96.0%** ($\tau=0.60$) | `results/week1/ood_rejection_results.json` |
| **Robustness Degradation** | Clean 98.0% vs. Severe Blur (r=6) 28.0% | `results/week2/robustness_stress_test.json` |
| **Grounding Ablation (RAG vs LLM)** | Strategy A 5.0/5.0 (0% hallucination) vs Strategy B 2.66/5.0 (38.9% hallucination) | `results/week2/grounding_ablation_results.json` |

---

## ❓ Frequently Asked Viva Questions & Standard Answers

### Q1: Why did you choose ResNet50 and EfficientNetV2-B0 instead of training a model from scratch?
**Answer**:
"PlantVillage contains 54,305 images across 38 classes. Transfer learning leveraging ImageNet pre-trained weights allows low-level feature reuse (edges, textures, leaf contours), accelerating convergence. ResNet50 provides a deep residual baseline (25.6M parameters), while EfficientNetV2-B0 provides NAS compound scaling (5.9M parameters, ~77% fewer parameters), reducing inference latency to 15.22 ms/img for edge mobile deployment."

### Q2: What is Expected Calibration Error (ECE) and why does Temperature Scaling matter?
**Answer**:
"Standard Softmax outputs are uncalibrated and overconfident—a network might output 99% probability on a corrupted or out-of-distribution image. Temperature scaling ($T=1.1959$) rescales logits before Softmax without altering classification rank. In our evaluation across 8,146 validation samples, Temperature Scaling reduced ECE from 1.09% to 0.42% (a 61.61% relative calibration error reduction)."

### Q3: How does Grad-CAM++ differ from standard Grad-CAM?
**Answer**:
"Standard Grad-CAM weights feature maps using first-order average gradients, which often highlights a single diffuse region. Grad-CAM++ incorporates second- and third-order partial gradients ($\alpha^{kc}_{ij}$) to weight positive gradients, enabling fine-grained visual attribution across multiple non-contiguous lesion spots on diseased leaves."

### Q4: How do you prevent LLM hallucinations when giving chemical treatment advice?
**Answer**:
"Direct LLM prompting exhibits a 38.9% chemical dosage hallucination rate on disease queries. We implement a Retrieval-Augmented Generation (RAG) engine that fetches structured agronomic protocols (symptoms, chemical active ingredients, organic remedies, prevention) from `rag_knowledge_base.json` and injects them as strict context in the LLM prompt. In our human grounding ablation study across 18 benchmark queries, Strategy A (Grounded RAG) achieved 100.0% factual alignment (5.00/5.0 Likert score)."

---

## 🔒 Claim Verification Audit Gate

All numbers in `IEEE_Paper_Draft.md` are audited via `verify_claims.py`.
To re-run the verification gate before submission:
```powershell
.\.venv\Scripts\python.exe -X utf8 verify_claims.py
```
**Expected Output**: `[SUCCESS] 0 UNVERIFIED claims, 85 CONFIRMED`.
