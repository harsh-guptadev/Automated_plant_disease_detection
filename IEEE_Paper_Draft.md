# Reliable, Explainable, and Uncertainty-Aware Plant Disease Diagnosis with Knowledge-Grounded Agronomic Decision Support

**Authors:** Harsh Gupta, Manthan, Sumit Kumar  
**Affiliation:** Department of Computer Science & Engineering, JSS Academy of Technical Education, Noida, India  
**Project Supervisor:** Ms. Pooja Deswal  

---

## Abstract
Deep Convolutional Neural Networks (CNNs) have achieved impressive top-1 accuracy in automated plant disease identification. However, real-world field deployment remains constrained by three fundamental vulnerabilities: (1) uncalibrated model overconfidence on out-of-distribution (OOD) or non-leaf imagery, (2) opaque black-box predictions without visual spatial justification, and (3) LLM hallucinations when generating chemical treatment advice. This paper presents an integrated agronomic decision-support framework that combines ResNet50 classification, post-hoc temperature scaling confidence calibration, an uncertainty-aware OOD rejection mechanism, Grad-CAM visual explainability, and a Retrieval-Augmented Generation (RAG) agronomy engine grounded in a curated disease knowledge base. Evaluated on a 70/15/15 stratified split of 54,305 PlantVillage images (fixed seed=123), our ResNet50 baseline achieves 94.87% top-1 accuracy (0.9335 macro F1), while temperature scaling reduces Expected Calibration Error (ECE) from 1.09% to 0.42% (a 61.61% relative reduction). An empirical evaluation of calibrated confidence thresholding (tau=0.60) reveals a 96.0% acceptance rate on in-distribution images, but highlights an important limitation with a 40.0% rejection rate on non-leaf images and 30.0% on severely blurred inputs.

**Index Terms:** Plant Pathology, Convolutional Neural Networks, Temperature Scaling, Out-of-Distribution Rejection, Explainable AI, Retrieval-Augmented Generation.

---

## I. Introduction
Plant diseases threaten global agricultural yields, crop productivity, and food security. Early and reliable disease diagnosis enables timely interventions, reducing crop loss and preventing unnecessary chemical pesticide overuse. 

While recent deep learning models excel at supervised classification, standard Softmax outputs do not correspond to true posterior probabilities. Uncalibrated networks produce high confidence scores even when presented with corrupted or non-target images (e.g., human hands, machinery, soil). Additionally, when generative AI models are prompted directly for crop treatment advice, they frequently hallucinate chemical dosages or unverified remedies, creating severe agricultural hazards.

This work addresses these challenges by introducing a reliable, calibrated, and explainable decision-support system paired with knowledge-grounded RAG agronomic recommendations. A grounding ablation study (currently in preparation) will quantitatively compare knowledge-base-grounded responses against direct LLM generation to characterize factual alignment.

---

## II. Related Work & Research Gap

### A. Literature Review
Recent research has actively explored deep learning ensembles, visual explainability, and natural language interfaces for agricultural diagnosis:

1. **Saha et al. (2025)** (*LeafCureX-LLM*, ScienceDirect, Oct 2025) proposed combining a stacking CNN ensemble with explainable AI (XAI) and LLM recommendations for real-time plant leaf diagnosis and treatment advice.
2. **Islam et al. (2025)** (*PlantCareNet*, *Plant Methods*, 21, 52) introduced a dual-mode recommendation system comparing reference-based agronomic guidelines against direct LLM generation for plant protection.
3. **Zhang (2026)** (*Chat Demeter*, *Frontiers in Plant Science*) designed a multi-agent CNN-Transformer system featuring a natural language interface for multi-crop disease diagnosis and management.

### B. Honest Research Gap & Contributions
Acknowledging prior work (Saha et al. 2025; Islam et al. 2025; Zhang 2026), the combination of classification, explainability, and LLM recommendation is **not** an unclaimed novel concept. Rather, this project focuses on two critical, less consistently addressed dimensions:

1. **Measured Model Reliability & Calibration**: Explicit evaluation of temperature scaling calibration (ECE reduction) and empirical robustness under image corruptions.
2. **Uncertainty-Aware OOD Rejection**: An explicit guardrail that detects non-leaf imagery and low-confidence inputs, suppressing forced disease predictions and instructing the LLM to output cautious agronomist disclaimers.

---

## III. Proposed Methodology

```text
┌──────────────────────────────┐
│  Input Image (224x224 RGB)   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ ResNet50 / EfficientNetV2-B0 │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Temperature Scaling (T=1.196)│
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Calibrated Max Confidence    │
└──────┬────────────────┬──────┘
       │                │
(Conf >= 0.60)   (Conf < 0.60)
       │                │
       ▼                ▼
┌──────────────┐ ┌──────────────────────────────┐
│ Diagnosis    │ │ OOD Rejection Guardrail      │
│ & Grad-CAM   │ │ "Uncertain - Consult Expert" │
└──────┬───────┘ └──────────────┬───────────────┘
       │                        │
       └───────────┬────────────┘
                   │
                   ▼
┌──────────────────────────────┐
│ RAG Agronomy Engine          │
│ (Grounded Care Advice)       │
└──────────────────────────────┘
```

### A. Classification Backbones
- **ResNet50 Baseline**: Residual skip connections, 25.6M parameters, Global Average Pooling head.
- **EfficientNetV2-B0 Benchmark**: Compound scaling backbone (5.9M parameters). Training script implemented (`src/models/train_efficientnet.py`) but execution is **pending** due to CPU-only environment constraints (no GPU acceleration available in the current runtime).

### B. Post-Hoc Temperature Scaling
Uncalibrated logits $z$ are scaled by an optimal temperature parameter $T > 0$ fitted on validation negative log-likelihood:
$$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

### C. Uncertainty-Aware OOD Rejection
If $\max_i \hat{p}_i < \tau$ (where $\tau = 0.60$), the prediction is rejected as uncertain. The UI displays an OOD warning banner, and the RAG prompt is automatically injected with an expert verification disclaimer.

---

## IV. Experimental Setup & Results

### A. ResNet50 Baseline Performance (15% Stratified Test Set, Seed=123)
Evaluated across 8,146 test images:
- **Total Test Samples**: 8,146
- **Top-1 Accuracy**: **94.87%**
- **Top-5 Accuracy**: **99.88%**
- **Macro Precision**: 0.9514 | **Macro Recall**: 0.9255 | **Macro F1**: **0.9335**
- **Weighted Precision**: 0.9559 | **Weighted Recall**: 0.9487 | **Weighted F1**: **0.9488**
- **Avg Inference Latency**: **31.53 ms/img**

### B. EfficientNetV2-B0 Comparison
**[PENDING — Training Not Yet Executed]** The EfficientNetV2-B0 transfer learning training script is implemented in `src/models/train_efficientnet.py`. Training execution is pending due to the absence of GPU acceleration in the current environment (TensorFlow ≥2.11 on native Windows does not support CUDA). Once trained, the model will be compared side-by-side on the same 15% TEST split. Expected parameter footprint: 5.9M (~77% fewer than ResNet50).

### C. Confidence Calibration (Validation Set - 8,146 Samples)
- **Optimal Temperature ($T$)**: **1.1959**
- **Uncalibrated ECE ($T=1.00$)**: **1.09%** (0.0109)
- **Calibrated ECE ($T=1.1959$)**: **0.42%** (0.0042)
- **Relative ECE Reduction**: **61.61%**

### D. Out-of-Distribution (OOD) Rejection Evaluation ($\tau = 0.60, T=1.1959$)
- **Tier 1 (PlantVillage Test Set)**: **96.0% Acceptance Rate** (96/100 accepted)
- **Tier 2 (Unrelated Non-Leaf Images)**: **40.0% Rejection Rate** (12/30 rejected, 18/30 accepted)
- **Tier 3 (Severely Blurred Leaves)**: **30.0% Rejection Rate** (9/30 rejected, 21/30 accepted)

#### Honest Analysis of OOD Detection Limitation
The maximum softmax confidence thresholding mechanism ($\tau = 0.60$) achieved a 96.0% acceptance rate on in-distribution PlantVillage test images. However, on Tier 2 (completely non-leaf images including hands, machinery, and surfaces), the rejection rate was only 40.0% (12/30 rejected, 18/30 accepted with a mean calibrated confidence of 0.6755). Similarly, on Tier 3 (severely blurred leaf images), the rejection rate was 30.0%. This reveals a critical finding: post-hoc softmax confidence thresholding, even when temperature calibrated, remains susceptible to overconfidence on feature representations far outside the training domain. Future iterations must incorporate feature-space density estimation (e.g. Mahalanobis distance or energy-based out-of-distribution scoring) to achieve robust non-leaf rejection.

### E. Robustness Stress-Testing Under Corruptions (ResNet50, n=100)

Evaluated across 100 stratified TEST samples (seed=123) under three corruption families. Full results in `results/week2/robustness_stress_test.json`.

| Corruption | Severity | ResNet50 Acc (%) | Degradation vs. Clean |
|---|---|---|---|
| Clean Baseline | — | **98.0%** | 0.0 pp |
| Gaussian Blur | Mild (r=2) | 65.0% | −33.0 pp |
| Gaussian Blur | Moderate (r=4) | 43.0% | −55.0 pp |
| Gaussian Blur | Severe (r=6) | 28.0% | **−70.0 pp** |
| Brightness (+30%) | Level 1 | 97.0% | −1.0 pp |
| Brightness (−30%) | Level 2 | 94.0% | −4.0 pp |
| Low Lighting (−60%) | Level 3 | 88.0% | −10.0 pp |
| JPEG Compression Q=50 | Level 1 | 96.0% | −2.0 pp |
| JPEG Compression Q=30 | Level 2 | 98.0% | 0.0 pp |
| JPEG Compression Q=10 | Level 3 | 84.0% | −14.0 pp |

**Key Finding**: Gaussian blur is the dominant failure mode, reducing accuracy by up to **70 percentage points** at severe blur (r=6). This is significant for field deployment where camera motion blur or rain interference can occur. Brightness and JPEG compression exhibit far greater robustness (at most 14 pp degradation). Future work: Gaussian blur augmentation during training or a blur-detection pre-filter.

### F. Grounding Ablation Study
**[PENDING — Human Evaluation Not Yet Executed]** The grounding ablation study script (`src/evaluation/grounding_ablation.py`) is implemented and defines 18 benchmark disease/question pairs. However, the study requires a genuine human rater pass by a team member — with real independent responses from both a grounded RAG call and a direct LLM call, rated for factual alignment and hallucination presence. The current script generates only mechanically-assigned ratings (every third query flagged regardless of content) and cannot substitute for this evaluation. Results will be reported in `results/week2/grounding_ablation_results.json` once the genuine human evaluation is complete.

### G. Grad-CAM++ Explainability Comparison

We evaluated Grad-CAM++ (`src/explainability/gradcam_pp.py`), which incorporates second- and third-order partial gradients to weight feature map activations ($\alpha^{kc}_{ij}$). A qualitative comparison across test leaf samples was generated and saved to `results/week3/xai_comparison.png`. While standard Grad-CAM tends to produce a single diffuse heat energy region centered on the primary visual feature, Grad-CAM++ captures multiple fine-grained lesion focal points on the leaf blade with improved spatial boundary resolution. In cases where higher-order gradients vanish (e.g. homogenous region activations), the implementation gracefully falls back to first-order Grad-CAM weighting to maintain robust visualization outputs.

---

## V. References
1. Saha, D. K., Ahmed, M. R., Nath, T. D., Boby, R. I., Hossen, M. J., & Mridha, M. F. (2025). "Fusing explainable deep learning ensembles and LLM recommendations for real-time plant leaf disease diagnosis." *ScienceDirect*, Oct 24, 2025.
2. Islam, M., Azad, A. K. M., Arman, S. E., Alyami, S. A., & Hasan, M. M. (2025). "PlantCareNet: An advanced system to recognize plant diseases with dual-mode recommendations for prevention." *Plant Methods*, 21, 52. https://doi.org/10.1186/s13007-025-01366-9
3. Zhang, S. (2026). "Chat Demeter: a multi-agent system for plant disease diagnosis integrating CNN-transformer models." *Frontiers in Plant Science*. https://doi.org/10.3389/fpls.2025.1695227
