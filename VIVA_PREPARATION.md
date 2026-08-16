# Academic Viva Defense & Oral Examination Guide (`VIVA_PREPARATION.md`)

**Project Title**: Reliable and Explainable Plant Disease Diagnosis and Management Using Deep Learning and Retrieval-Augmented Generation  
**Target Level**: B.Tech CSE / AI-ML Major Capstone & Research Presentation  

---

## 1. Project Overview & Pitch

### Question 1: What is the core objective of your major project?
**Answer**: Our project presents an explainable, uncertainty-aware, and retrieval-augmented decision-support system for plant disease diagnosis and crop management. Rather than operating as a simple black-box classifier, it integrates:
1. **ResNet50 Transfer Learning** for multi-class leaf disease identification across 38 crop categories.
2. **Grad-CAM Visual Explainability** to highlight spatial regions influencing model attention.
3. **An Attention-Based Severity Heuristic** to indicate feature activation area without making scientifically unsupported segmentation claims.
4. **A RAG (Retrieval-Augmented Generation) Agronomy Engine** that grounds LLM treatment recommendations in certified pathology data to eliminate AI hallucinations.

---

## 2. Deep Learning & Computer Vision Architecture

### Question 2: Why did you choose ResNet50 and Transfer Learning?
**Answer**: Training deep CNNs from scratch on domain-specific datasets requires millions of images to learn low-level visual primitives (edges, textures, colors). ResNet50 pre-trained on ImageNet provides rich feature representations. Its residual skip connections (\(y = f(x) + x\)) mitigate the vanishing gradient problem, allowing deep feature extraction. Fine-tuning the classification head (Global Average Pooling + Dense 256 + Softmax) allows rapid convergence on the 38-class PlantVillage dataset.

### Question 3: How does Grad-CAM work in your implementation?
**Answer**: Gradient-weighted Class Activation Mapping (Grad-CAM) calculates the gradients of the target class score \(y^c\) with respect to the feature map outputs \(A^k\) of the final convolutional layer (`conv5_block3_out`). We compute neuron importance weights \(\alpha_k^c\) via global average pooling of gradients:
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
A weighted combination of feature maps is passed through a ReLU activation to extract features that positively contribute to the target class decision.

---

## 3. RAG Engine & LLM Grounding

### Question 4: Why is RAG necessary for plant disease management?
**Answer**: Standalone Large Language Models (LLMs) are prone to hallucinations, generating plausible-sounding but inaccurate chemical dosages or non-existent pesticide names. In agriculture, incorrect fungicide advice can ruin crops or cause environmental damage. By retrieving exact, peer-reviewed treatment data from `rag_knowledge_base.json` and injecting it into the LLM system prompt, we constrain the LLM to summarize and translate certified agronomic protocols safely.

---

## 4. Methodological Validity & Limitations

### Question 5: Why do you call your severity calculation an "Attention-Based Heuristic" rather than exact disease severity?
**Answer**: Grad-CAM outputs feature attention heatmaps based on classification gradients—it is not trained on ground-truth semantic segmentation masks (e.g., IoU / Dice loss annotations). Claiming exact percentage infection from Grad-CAM activation area would be scientifically inaccurate. Therefore, we responsibly define it as an *"Attention-Based Feature Activation Area"* to guide agronomists without overselling model capabilities.

### Question 6: What are the main limitations of your system?
**Answer**:
1. **Dataset Shift**: PlantVillage images are captured against clean, controlled backgrounds. Real-world field images with complex foliage, background soil, or lighting glare may reduce classification confidence.
2. **Co-occurring Infections**: The model currently assumes a single primary label per leaf image rather than multi-label compound infections.
3. **Uncertainty Guardrails**: In low-confidence scenarios (\(<60\%\)), the system flags uncertainty rather than making overconfident diagnosis claims.

---

## 5. Summary Table for Quick Viva Revision

| Topic | Technical Detail |
| :--- | :--- |
| **Dataset** | PlantVillage (38 Classes, Healthy & Diseased crops) |
| **Base Model** | ResNet50 (Transfer Learning, GAP + Dense 256 + Softmax) |
| **Explainability** | Grad-CAM on layer `conv5_block3_out` |
| **Severity Estimator** | Attention-Based Feature Intensity Thresholding |
| **RAG Retrieval** | JSON Structured Lookup + Context-Augmented Prompting |
| **LLM Inference** | Multi-model failover (Qwen2.5-Coder / Mistral-7B / Llama-3.2) |
| **PDF Engine** | FPDF2 with Latin-1 text sanitization |
