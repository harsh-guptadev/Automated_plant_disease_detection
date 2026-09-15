"""
src/evaluation/evaluate_resnet.py
==================================
Empirical evaluation module for ResNet50 baseline model on the 15% TEST split.
Computes accuracy, precision, recall, F1 (macro + per-class), and confusion matrix.
Saves all quantitative results to results/week1/resnet50_test_metrics.json.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, models, Input

CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

def load_resnet_model(weights_path="resnet_weights.npz"):
    inputs = Input(shape=(224, 224, 3))
    resnet_base = ResNet50(weights=None, include_top=False, input_tensor=inputs)
    x = resnet_base.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation='relu', name="head_dense")(x)
    x = layers.Dropout(0.4, name="head_dropout")(x)
    outputs = layers.Dense(len(CLASSES), activation='softmax', name="predictions")(x)
    model = models.Model(inputs, outputs)

    if os.path.exists(weights_path):
        weights = np.load(weights_path, allow_pickle=True)
        model.set_weights([weights[key] for key in weights])
        print(f"[Model Loader] Successfully loaded weights from {weights_path}")
    else:
        raise FileNotFoundError(f"Weights file not found at {weights_path}")
    return model

def load_image(path):
    img = Image.open(path).convert('RGB').resize((224, 224))
    img_arr = np.array(img, dtype=np.float32)
    return img_arr

def run_resnet_evaluation(split_json="data_splits/test_split.json", output_dir="results/week1", batch_size=64):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(split_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"[ResNet50 Eval] Evaluating on {len(test_data)} test samples...")
    model = load_resnet_model()

    y_true = []
    y_pred_probs = []
    
    start_time = time.time()
    
    # Process in batches for performance
    for i in range(0, len(test_data), batch_size):
        batch_items = test_data[i:i+batch_size]
        batch_imgs = []
        for item in batch_items:
            try:
                img_arr = load_image(item["path"])
                batch_imgs.append(img_arr)
                y_true.append(item["label"])
            except Exception as e:
                print(f"[Warning] Failed to load {item['path']}: {e}")

        if not batch_imgs:
            continue
            
        batch_imgs = np.array(batch_imgs)
        batch_preprocessed = preprocess_input(batch_imgs.copy())
        probs = model.predict(batch_preprocessed, verbose=0)
        y_pred_probs.append(probs)

    y_true = np.array(y_true)
    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    elapsed_time = time.time() - start_time
    avg_latency_ms = (elapsed_time / len(y_true)) * 1000.0

    # Calculate metrics
    top1_acc = float(np.mean(y_true == y_pred))
    
    # Top-5 Accuracy
    top5_hits = 0
    for i in range(len(y_true)):
        top5_preds = np.argsort(y_pred_probs[i])[-5:]
        if y_true[i] in top5_preds:
            top5_hits += 1
    top5_acc = float(top5_hits / len(y_true))

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    weight_p, weight_r, weight_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(y_true, y_pred, average=None)

    per_class_dict = {}
    for idx, name in enumerate(CLASSES):
        if idx < len(per_class_p):
            per_class_dict[name] = {
                "precision": round(float(per_class_p[idx]), 4),
                "recall": round(float(per_class_r[idx]), 4),
                "f1_score": round(float(per_class_f1[idx]), 4),
                "support": int(per_class_supp[idx])
            }

    metrics_result = {
        "model_name": "ResNet50 Baseline",
        "evaluation_split": "TEST (15% stratified, seed=123)",
        "total_test_samples": int(len(y_true)),
        "top1_accuracy": round(top1_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1_score": round(float(macro_f1), 4),
        "weighted_precision": round(float(weight_p), 4),
        "weighted_recall": round(float(weight_r), 4),
        "weighted_f1_score": round(float(weight_f1), 4),
        "avg_inference_latency_ms": round(avg_latency_ms, 2),
        "per_class_metrics": per_class_dict
    }

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "resnet50_test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_result, f, indent=2)
    print(f"[ResNet50 Eval] Saved metrics to {metrics_path}")

    # Compute and plot Normalized Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(24, 20), dpi=150)
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Normalized Confusion Matrix - ResNet50 (Test Set)", fontsize=16, pad=20)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "resnet50_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"[ResNet50 Eval] Saved confusion matrix plot to {cm_path}")

    # Save raw logits/probs and true labels for calibration
    np.savez(
        os.path.join(output_dir, "resnet50_test_probs.npz"),
        y_true=y_true,
        y_pred_probs=y_pred_probs
    )

    print(f"\n================ RESNET50 EVALUATION SUMMARY ================")
    print(f"Top-1 Accuracy:  {top1_acc * 100:.2f}%")
    print(f"Top-5 Accuracy:  {top5_acc * 100:.2f}%")
    print(f"Macro F1 Score:  {macro_f1:.4f}")
    print(f"Weighted F1:     {weight_f1:.4f}")
    print(f"Avg Latency:     {avg_latency_ms:.2f} ms/img")
    print(f"============================================================")

    return metrics_result

if __name__ == "__main__":
    run_resnet_evaluation()
