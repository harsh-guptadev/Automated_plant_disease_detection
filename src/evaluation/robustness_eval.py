"""
src/evaluation/robustness_eval.py
=================================
Robustness stress-testing module evaluating accuracy degradation under 3 corruption types:
1. Gaussian Blur (kernel radii: 2, 4, 6)
2. Brightness/Contrast Shift (+30%, -30%, low contrast)
3. JPEG Compression Artifacts (Quality levels: 50, 30, 10)
Applied ONLY to the 15% TEST split (no re-training).
"""

import os
import json
import io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effnet_preprocess
from src.evaluation.evaluate_resnet import load_resnet_model
from src.models.train_efficientnet import build_efficientnet_model

def apply_corruption(img_pil, corruption_type, level):
    """
    Applies real-world image corruption to a PIL RGB image.
    """
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
    effnet_weights="results/week2/efficientnetv2_weights.h5",
    output_dir="results/week2",
    sample_limit=200
):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(test_split_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    eval_data = test_data[:sample_limit]
    print(f"[Robustness Eval] Running corruption benchmark on {len(eval_data)} test images...")

    resnet_model = load_resnet_model()
    effnet_model = build_efficientnet_model()
    if os.path.exists(effnet_weights):
        effnet_model.load_weights(effnet_weights)

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

    for c_type, c_level, label in corruptions:
        y_true = []
        res_preds = []
        eff_preds = []

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
                res_preds.append(np.argmax(p_res))

                # EfficientNetV2 prediction
                eff_in = effnet_preprocess(np.expand_dims(img_arr.copy(), axis=0))
                p_eff = effnet_model.predict(eff_in, verbose=0)[0]
                eff_preds.append(np.argmax(p_eff))
            except Exception:
                continue

        y_true = np.array(y_true)
        res_acc = float(np.mean(y_true == np.array(res_preds)))
        eff_acc = float(np.mean(y_true == np.array(eff_preds)))

        results_table.append({
            "corruption_type": c_type,
            "severity_level": c_level,
            "description": label,
            "resnet50_accuracy_pct": round(res_acc * 100.0, 2),
            "efficientnetv2_accuracy_pct": round(eff_acc * 100.0, 2),
            "resnet50_degradation_pct": round((results_table[0]["resnet50_accuracy_pct"] - res_acc * 100.0), 2) if results_table else 0.0,
            "efficientnetv2_degradation_pct": round((results_table[0]["efficientnetv2_accuracy_pct"] - eff_acc * 100.0), 2) if results_table else 0.0
        })
        print(f"[Robustness] {label:<32} | ResNet50 Acc: {res_acc*100:.1f}% | EffNetV2 Acc: {eff_acc*100:.1f}%")

    out_json = os.path.join(output_dir, "robustness_stress_test.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2)

    print(f"\n[Robustness Eval] Saved stress-test benchmark to {out_json}")
    return results_table

if __name__ == "__main__":
    run_robustness_evaluation()
