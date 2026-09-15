"""
src/calibration/ood_detector.py
================================
Uncertainty-Aware Out-of-Distribution (OOD) Rejection Engine.
Uses calibrated max confidence to distinguish in-distribution plant leaf patterns
from non-target (OOD) images (hands, cars, random noise) and heavily corrupted samples.
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from src.evaluation.evaluate_resnet import load_resnet_model
from src.calibration.temperature_scaling import softmax

DEFAULT_TEMP = 1.62  # Fitted during calibration step

def get_calibrated_confidence(model, img_array, temperature=DEFAULT_TEMP):
    """
    Computes temperature-scaled confidence for an input image array (224, 224, 3).
    """
    img_batch = np.expand_dims(img_array, axis=0)
    preprocessed = preprocess_input(img_batch.copy())

    feat_extractor = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer("predictions").input)
    out_dense = model.get_layer("predictions")
    weights, biases = out_dense.get_weights()

    feats = feat_extractor.predict(preprocessed, verbose=0)
    logits = np.dot(feats, weights) + biases
    calibrated_probs = softmax(logits, temp=temperature)[0]

    top_idx = int(np.argmax(calibrated_probs))
    calibrated_conf = float(np.max(calibrated_probs))

    return top_idx, calibrated_conf, calibrated_probs

def generate_synthetic_ood_images(count=30, output_dir="scratch/ood_test_images"):
    """
    Generates synthetic benchmark OOD images (solid colors, random noise, geometric shapes, non-leaf textures)
    for reproducible OOD threshold testing when local image files are not pre-packaged.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    
    # 1. Random noise images
    for i in range(10):
        noise = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(noise)
        p = os.path.join(output_dir, f"ood_noise_{i}.jpg")
        img.save(p)
        paths.append((p, "Random Noise (OOD)"))

    # 2. Geometric / Non-plant objects (e.g. brick wall / fabric pattern / car outline)
    for i in range(10):
        img = Image.new("RGB", (224, 224), color=(120 + i*10, 80 - i*3, 200 - i*15))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 200, 200], outline=(255, 255, 0), width=5)
        draw.ellipse([50, 50, 170, 170], fill=(255, 0, 100))
        p = os.path.join(output_dir, f"ood_synthetic_object_{i}.jpg")
        img.save(p)
        paths.append((p, "Non-Leaf Synthetic Object (OOD)"))

    # 3. Solid blank wall / skin tone / indoor surface images
    for i in range(10):
        color = (210 + i*2, 160 + i, 140) if i % 2 == 0 else (50 + i*5, 50 + i*5, 50 + i*5)
        img = Image.new("RGB", (224, 224), color=color)
        p = os.path.join(output_dir, f"ood_surface_{i}.jpg")
        img.save(p)
        paths.append((p, "Solid Surface/Skin Tone (OOD)"))

    return paths

def evaluate_ood_rejection(
    test_split_json="data_splits/test_split.json",
    calibration_json="results/week1/temperature_scale.json",
    threshold=0.60,
    output_dir="results/week1"
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Read fitted temperature
    temp = DEFAULT_TEMP
    if os.path.exists(calibration_json):
        with open(calibration_json, "r", encoding="utf-8") as f:
            cdata = json.load(f)
            temp = cdata.get("optimal_temperature", DEFAULT_TEMP)

    print(f"[OOD Rejection] Running evaluation with Calibrated Temp T={temp:.2f} and Threshold tau={threshold:.2f}")
    model = load_resnet_model()

    # Tier 1: Real In-Distribution PlantVillage Test Images (Sample of 100)
    with open(test_split_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Use first 100 test images for clean evaluation benchmark
    id_samples = test_data[:100]
    id_confs = []
    id_rejections = 0

    for item in id_samples:
        try:
            img = Image.open(item["path"]).convert("RGB").resize((224, 224))
            img_arr = np.array(img, dtype=np.float32)
            _, conf, _ = get_calibrated_confidence(model, img_arr, temperature=temp)
            id_confs.append(conf)
            if conf < threshold:
                id_rejections += 1
        except Exception:
            continue

    # Tier 2: Clearly Unrelated Non-Leaf Images (30 synthetic OOD images)
    ood_paths = generate_synthetic_ood_images(30)
    ood_confs = []
    ood_rejections = 0

    for path, desc in ood_paths:
        img = Image.open(path).convert("RGB").resize((224, 224))
        img_arr = np.array(img, dtype=np.float32)
        _, conf, _ = get_calibrated_confidence(model, img_arr, temperature=temp)
        ood_confs.append(conf)
        if conf < threshold:
            ood_rejections += 1

    # Tier 3: Borderline / Blurred Plant Images (30 blurred test images)
    borderline_confs = []
    borderline_rejections = 0

    for item in test_data[100:130]:
        try:
            img = Image.open(item["path"]).convert("RGB").resize((224, 224))
            # Apply severe Gaussian blur
            img_blurred = img.filter(ImageFilter.GaussianBlur(radius=7))
            img_arr = np.array(img_blurred, dtype=np.float32)
            _, conf, _ = get_calibrated_confidence(model, img_arr, temperature=temp)
            borderline_confs.append(conf)
            if conf < threshold:
                borderline_rejections += 1
        except Exception:
            continue

    # Calculate metrics
    id_pass_rate = (len(id_confs) - id_rejections) / len(id_confs) * 100.0 if id_confs else 0
    ood_reject_rate = ood_rejections / len(ood_confs) * 100.0 if ood_confs else 0
    borderline_reject_rate = borderline_rejections / len(borderline_confs) * 100.0 if borderline_confs else 0

    ood_results = {
        "rejection_threshold_tau": threshold,
        "calibrated_temperature_T": temp,
        "tier1_in_distribution_plantvillage": {
            "total_tested": len(id_confs),
            "accepted_count": len(id_confs) - id_rejections,
            "rejected_count": id_rejections,
            "acceptance_rate_pct": round(id_pass_rate, 2),
            "mean_calibrated_confidence": round(float(np.mean(id_confs)), 4)
        },
        "tier2_unrelated_non_leaf_ood": {
            "total_tested": len(ood_confs),
            "rejected_count": ood_rejections,
            "accepted_count": len(ood_confs) - ood_rejections,
            "rejection_rate_pct": round(ood_reject_rate, 2),
            "mean_calibrated_confidence": round(float(np.mean(ood_confs)), 4)
        },
        "tier3_borderline_blurred_leaves": {
            "total_tested": len(borderline_confs),
            "rejected_count": borderline_rejections,
            "accepted_count": len(borderline_confs) - borderline_rejections,
            "rejection_rate_pct": round(borderline_reject_rate, 2),
            "mean_calibrated_confidence": round(float(np.mean(borderline_confs)), 4)
        }
    }

    out_json = os.path.join(output_dir, "ood_rejection_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ood_results, f, indent=2)

    print(f"\n================ OOD REJECTION TEST SUMMARY ================")
    print(f"Rejection Threshold (tau):       {threshold:.2f}")
    print(f"Tier 1 PlantVillage Acceptance:  {id_pass_rate:.1f}% ({len(id_confs) - id_rejections}/{len(id_confs)})")
    print(f"Tier 2 Non-Leaf OOD Rejection:  {ood_reject_rate:.1f}% ({ood_rejections}/{len(ood_confs)})")
    print(f"Tier 3 Blurred Leaf Rejection:   {borderline_reject_rate:.1f}% ({borderline_rejections}/{len(borderline_confs)})")
    print(f"============================================================")

    return ood_results

if __name__ == "__main__":
    evaluate_ood_rejection()
