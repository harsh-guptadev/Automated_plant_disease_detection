"""
src/calibration/temperature_scaling.py
======================================
Post-hoc confidence calibration using Temperature Scaling on the Validation set.
Fits temperature T > 0, computes Expected Calibration Error (ECE) before/after,
and saves reliability diagram plots and fitted scaling parameters.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from src.evaluation.evaluate_resnet import load_resnet_model, load_image

def compute_ece(probs, labels, n_bins=15):
    """
    Computes Expected Calibration Error (ECE).
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
            bin_counts.append(int(np.sum(in_bin)))
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_counts.append(0)

    return float(ece), bin_accs, bin_confs, bin_boundaries

def softmax(logits, temp=1.0):
    logits_scaled = logits / temp
    exps = np.exp(logits_scaled - np.max(logits_scaled, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)

def fit_temperature_scaling(val_split_json="data_splits/val_split.json", output_dir="results/week1"):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(val_split_json, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    print(f"[Temperature Scaling] Loading validation set ({len(val_data)} samples)...")
    model = load_resnet_model()

    # Get layer before softmax (head_dense output before softmax)
    feat_extractor = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer("predictions").input)
    # The output layer weights
    out_dense = model.get_layer("predictions")
    weights, biases = out_dense.get_weights()

    val_logits = []
    val_labels = []

    batch_size = 64
    for i in range(0, len(val_data), batch_size):
        batch_items = val_data[i:i+batch_size]
        batch_imgs = []
        for item in batch_items:
            try:
                img_arr = load_image(item["path"])
                batch_imgs.append(img_arr)
                val_labels.append(item["label"])
            except Exception:
                continue

        if not batch_imgs:
            continue

        batch_imgs = np.array(batch_imgs)
        batch_preprocessed = preprocess_input(batch_imgs.copy())
        
        # Calculate raw logits: features * W + b
        feats = feat_extractor.predict(batch_preprocessed, verbose=0)
        logits = np.dot(feats, weights) + biases
        val_logits.append(logits)

    val_logits = np.concatenate(val_logits, axis=0)
    val_labels = np.array(val_labels)

    # Initial probabilities (T = 1.0)
    uncalibrated_probs = softmax(val_logits, temp=1.0)
    ece_before, accs_before, confs_before, bin_bounds = compute_ece(uncalibrated_probs, val_labels)

    # Loss function for temperature optimization (Negative Log-Likelihood)
    def nll_loss(t):
        t_val = t[0]
        probs = softmax(val_logits, temp=t_val)
        # Avoid log(0)
        probs = np.clip(probs, 1e-15, 1.0 - 1e-15)
        nll = -np.mean(np.log(probs[np.arange(len(val_labels)), val_labels]))
        return nll

    res = minimize(nll_loss, [1.0], bounds=[(0.01, 10.0)], method='L-BFGS-B')
    optimal_temp = float(res.x[0])

    calibrated_probs = softmax(val_logits, temp=optimal_temp)
    ece_after, accs_after, confs_after, _ = compute_ece(calibrated_probs, val_labels)

    print(f"\n================ TEMPERATURE CALIBRATION SUMMARY ================")
    print(f"Optimal Temperature (T):  {optimal_temp:.4f}")
    print(f"ECE Before Calibration:    {ece_before * 100:.2f}%")
    print(f"ECE After Calibration:     {ece_after * 100:.2f}%")
    print(f"ECE Reduction:            {(ece_before - ece_after) * 100:.2f}% drop")
    print(f"==================================================================")

    # Save Reliability Diagram Plot
    plt.figure(figsize=(12, 5), dpi=150)

    # Pre-calibration plot
    plt.subplot(1, 2, 1)
    bin_centers = 0.5 * (bin_bounds[:-1] + bin_bounds[1:])
    plt.bar(bin_centers, accs_before, width=1.0/15, alpha=0.6, color='crimson', label='Outputs (Acc)', edgecolor='black')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.title(f"Uncalibrated (T=1.00)\nECE: {ece_before*100:.2f}%", fontsize=12)
    plt.xlabel("Confidence", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Post-calibration plot
    plt.subplot(1, 2, 2)
    plt.bar(bin_centers, accs_after, width=1.0/15, alpha=0.6, color='forestgreen', label='Outputs (Acc)', edgecolor='black')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.title(f"Calibrated (T={optimal_temp:.2f})\nECE: {ece_after*100:.2f}%", fontsize=12)
    plt.xlabel("Confidence", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    diagram_path = os.path.join(output_dir, "calibration_reliability_diagram.png")
    plt.savefig(diagram_path)
    plt.close()

    # Save calibration metadata JSON
    calibration_meta = {
        "model_name": "ResNet50 Baseline",
        "optimal_temperature": round(optimal_temp, 4),
        "ece_uncalibrated": round(ece_before, 4),
        "ece_calibrated": round(ece_after, 4),
        "ece_improvement_percent": round((ece_before - ece_after) * 100.0, 2),
        "val_sample_count": len(val_labels),
        "n_bins": 15
    }

    json_path = os.path.join(output_dir, "temperature_scale.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(calibration_meta, f, indent=2)
    print(f"[Temperature Scaling] Saved parameters and metadata to {json_path}")

    return calibration_meta

if __name__ == "__main__":
    fit_temperature_scaling()
