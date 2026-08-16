"""
batch_processor.py
==================
Batch image evaluation module for multi-leaf field diagnostics and CSV report logging.
Includes Grad-CAM heatmap visualization for visual photo grid cards.
"""

import io
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input


def process_batch_images(
    uploaded_files,
    model,
    classes: list,
    class_descriptions: dict,
    estimate_severity_fn,
    make_gradcam_fn
):
    """
    Processes a list of uploaded leaf image files.
    Returns:
        summary_stats (dict): Field-level aggregated health metrics.
        results_list (list of dicts): Detailed per-image metadata including leaf photos & Grad-CAM overlays.
        df_export (pd.DataFrame): Structured tabular representation for CSV export.
    """
    results_list = []
    healthy_count = 0
    infected_count = 0
    high_urgency_count = 0
    total_confidence = 0.0

    for file in uploaded_files:
        try:
            image = Image.open(file).convert("RGB")
            image_resized = image.resize((224, 224))
            image_array = np.array(image_resized)
            image_batch = np.expand_dims(image_array, axis=0)
            image_batch_preprocessed = preprocess_input(image_batch.copy())

            predictions = model.predict(image_batch_preprocessed, verbose=0)
            confidence = float(np.max(predictions))
            predicted_class = classes[np.argmax(predictions)]
            readable_prediction = class_descriptions.get(predicted_class, predicted_class)
            is_healthy = "healthy" in readable_prediction.lower()

            if is_healthy:
                healthy_count += 1
            else:
                infected_count += 1

            total_confidence += confidence

            # Compute Grad-CAM and severity metric
            try:
                heatmap = make_gradcam_fn(image_batch_preprocessed, model)
                severity_metrics = estimate_severity_fn(heatmap, confidence=confidence)

                img_cv = np.array(image)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
                heatmap_uint8 = np.uint8(255 * heatmap_resized)
                heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)
                superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
            except Exception:
                severity_metrics = {
                    "affected_percentage": 0.0,
                    "severity_level": "Unknown",
                    "badge_color": "#94a3b8",
                    "urgency": "Low",
                    "is_uncertain": confidence < 0.60
                }
                superimposed_img_rgb = np.array(image)

            if "High" in severity_metrics.get("urgency", ""):
                high_urgency_count += 1

            item = {
                "Filename": file.name,
                "Detected Condition": readable_prediction,
                "Status": "HEALTHY" if is_healthy else "INFECTED",
                "Confidence Score": f"{confidence * 100:.1f}%",
                "Attention Coverage": f"{severity_metrics['affected_percentage']}%",
                "Attention Severity": severity_metrics["severity_level"],
                "Urgency": severity_metrics["urgency"],
                "raw_confidence": confidence,
                "is_healthy": is_healthy,
                "image": image,
                "superimposed_img_rgb": superimposed_img_rgb,
                "severity_metrics": severity_metrics,
                "predicted_class": predicted_class
            }
            results_list.append(item)
        except Exception as e:
            continue

    total_scanned = len(results_list)
    field_health_score = (healthy_count / total_scanned * 100.0) if total_scanned > 0 else 0.0
    avg_confidence = (total_confidence / total_scanned * 100.0) if total_scanned > 0 else 0.0

    summary_stats = {
        "total_scanned": total_scanned,
        "healthy_count": healthy_count,
        "infected_count": infected_count,
        "field_health_score": round(field_health_score, 1),
        "avg_confidence": round(avg_confidence, 1),
        "high_urgency_count": high_urgency_count
    }

    # Format DataFrame for UI display & CSV export
    df_export = pd.DataFrame([{
        "Filename": r["Filename"],
        "Detected Condition": r["Detected Condition"],
        "Status": r["Status"],
        "Confidence Score": r["Confidence Score"],
        "Attention Coverage": r["Attention Coverage"],
        "Attention Severity": r["Attention Severity"],
        "Urgency": r["Urgency"]
    } for r in results_list])

    return summary_stats, results_list, df_export
