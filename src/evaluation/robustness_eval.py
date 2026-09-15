"""
src/evaluation/robustness_eval.py
=================================
Robustness stress-testing module evaluating accuracy degradation under 3 corruption types:
1. Gaussian Blur (kernel radii: 2, 4, 6)
2. Brightness/Contrast Shift (+30%, -30%, low contrast)
3. JPEG Compression Artifacts (Quality levels: 50, 30, 10)
Applied ONLY to the 15% TEST split (no re-training).

NOTE: Only ResNet50 is evaluated here. EfficientNetV2-B0 columns are omitted
until training and weights are available (src/models/train_efficientnet.py).
"""

import os
import sys
import json
import io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from src.evaluation.evaluate_resnet import load_resnet_model


def apply_corruption(img_pil, corruption_type, level):
    """Applies real-world image corruption to a PIL RGB image."""
    if corruption_type == "gaussian_blur":
        radius = [2, 4, 6][level - 1]
        return img_pil.filter(ImageFilter.GaussianBlur(radius=radius))

    elif corruption_type == "brightness_shift":
        factor = [1.3, 0.7, 0.4][level - 1]
        enhancer = ImageEnhance.Brightness(img_pil)
        return enhancer.enhance(factor)

    elif corruption_type == "jpeg_compression":
        quality = [50, 30, 10][level - 1]
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    return img_pil


def run_robustness_evaluation(
    test_split_json="data_splits/test_split.json",
    weights_path="resnet_weights.npz",
    output_dir="results/week2",
    sample_limit=300
):
    os.makedirs(output_dir, exist_ok=True)

    with open(test_split_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    eval_data = test_data[:sample_limit]
    print(f"[Robustness Eval] Running corruption benchmark on {len(eval_data)} test images...")
    print("[Robustness Eval] Loading ResNet50 model...")

    resnet_model = load_resnet_model(weights_path=weights_path)

    corruptions = [
        ("clean", 0, "Clean Baseline"),
        ("gaussian_blur", 1, "Gaussian Blur (Mild r=2)"),
        ("gaussian_blur", 2, "Gaussian Blur (Moderate r=4)"),
        ("gaussian_blur", 3, "Gaussian Blur (Severe r=6)"),
        ("brightness_shift", 1, "Brightness (+30%)"),
        ("brightness_shift", 2, "Brightness (-30%)"),
        ("brightness_shift", 3, "Low Lighting (-60%)"),
        ("jpeg_compression", 1, "JPEG Compression (Q=50)"),
        ("jpeg_compression", 2, "JPEG Compression (Q=30)"),
        ("jpeg_compression", 3, "JPEG Compression (Q=10)")
    ]

    results_table = []
    clean_baseline_acc = None

    for c_type, c_level, label in corruptions:
        y_true = []
        res_preds = []
        skipped = 0

        for item in eval_data:
            try:
                img_pil = Image.open(item["path"]).convert("RGB").resize((224, 224))
                if c_type != "clean":
                    img_corrupted = apply_corruption(img_pil, c_type, c_level)
                else:
                    img_corrupted = img_pil

                img_arr = np.array(img_corrupted, dtype=np.float32)
                y_true.append(item["label"])

                # ResNet50 prediction
                res_in = resnet_preprocess(np.expand_dims(img_arr.copy(), axis=0))
                p_res = resnet_model.predict(res_in, verbose=0)[0]
                res_preds.append(int(np.argmax(p_res)))

            except Exception as e:
                skipped += 1
                continue

        if len(y_true) == 0:
            print(f"[Robustness] {label:<40} | No valid samples!")
            continue

        y_true_arr = np.array(y_true)
        res_acc = float(np.mean(y_true_arr == np.array(res_preds)))

        if c_type == "clean" and c_level == 0:
            clean_baseline_acc = res_acc

        degradation = round((clean_baseline_acc - res_acc) * 100.0, 2) if clean_baseline_acc is not None and c_type != "clean" else 0.0

        row = {
            "corruption_type": c_type,
            "severity_level": c_level,
            "description": label,
            "n_samples_evaluated": len(y_true),
            "n_samples_skipped": skipped,
            "resnet50_accuracy_pct": round(res_acc * 100.0, 2),
            "resnet50_degradation_vs_clean_pct": degradation,
        }
        results_table.append(row)
        print(f"[Robustness] {label:<40} | ResNet50 Acc: {res_acc*100:.1f}% | Degradation: {degradation:.1f}pp | Skipped: {skipped}")

    out_json = os.path.join(output_dir, "robustness_stress_test.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "note": "ResNet50-only robustness evaluation. EfficientNetV2-B0 omitted (weights not yet available).",
            "model": "ResNet50 Baseline (resnet_weights.npz)",
            "test_split": "15% stratified (seed=123)",
            "n_samples_per_corruption": sample_limit,
            "results": results_table
        }, f, indent=2)

    print(f"\n[Robustness Eval] Saved stress-test benchmark to {out_json}")
    return results_table


if __name__ == "__main__":
    run_robustness_evaluation()
