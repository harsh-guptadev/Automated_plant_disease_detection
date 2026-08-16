"""
efficientnet_benchmark.py
==========================
Model benchmarking module comparing ResNet50 baseline against EfficientNetV2-B0
for academic research and viva defense.
"""

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetV2B0
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effnet_preprocess
from tensorflow.keras import layers, models, Input


def get_model_specs():
    """Returns architecture specs for ResNet50 vs EfficientNetV2-B0."""
    return {
        "ResNet50": {
            "parameters": 25636710,
            "params_formatted": "25.6M",
            "model_size_mb": 97.8,
            "depth_layers": 50,
            "input_resolution": "224 x 224",
            "scaling_type": "Fixed Architecture (Residual Skip)"
        },
        "EfficientNetV2-B0": {
            "parameters": 5919310,
            "params_formatted": "5.9M",
            "model_size_mb": 22.6,
            "depth_layers": 210,
            "input_resolution": "224 x 224",
            "scaling_type": "Compound Scaling (Fused-MBConv)"
        }
    }


def benchmark_single_image(image_array: np.ndarray, resnet_model, num_classes: int = 38):
    """
    Runs side-by-side inference on a single image using ResNet50 and EfficientNetV2-B0,
    measuring latency, top-1 confidence, and top-3 predicted class indices.
    """
    results = {}
    
    # ── 1. ResNet50 Inference ──
    img_resnet = np.expand_dims(image_array, axis=0)
    img_resnet = resnet_preprocess(img_resnet.copy())
    
    start_time = time.perf_counter()
    preds_resnet = resnet_model.predict(img_resnet, verbose=0)[0]
    resnet_latency_ms = (time.perf_counter() - start_time) * 1000.0
    
    top3_idx_resnet = np.argsort(preds_resnet)[::-1][:3]
    
    results["ResNet50"] = {
        "confidence": float(np.max(preds_resnet)),
        "latency_ms": round(resnet_latency_ms, 2),
        "top3_indices": top3_idx_resnet,
        "top3_probs": [float(preds_resnet[i]) for i in top3_idx_resnet],
        "all_probs": preds_resnet
    }
    
    # ── 2. EfficientNetV2-B0 Inference (Benchmarking Head) ──
    effnet_inputs = Input(shape=(224, 224, 3))
    effnet_base = EfficientNetV2B0(weights='imagenet', include_top=False, input_tensor=effnet_inputs)
    x = layers.GlobalAveragePooling2D()(effnet_base.output)
    x = layers.Dense(256, activation='relu')(x)
    effnet_outputs = layers.Dense(num_classes, activation='softmax')(x)
    effnet_model = models.Model(effnet_inputs, effnet_outputs)
    
    img_effnet = np.expand_dims(image_array, axis=0)
    img_effnet = effnet_preprocess(img_effnet.copy())
    
    start_time = time.perf_counter()
    preds_effnet = effnet_model.predict(img_effnet, verbose=0)[0]
    effnet_latency_ms = (time.perf_counter() - start_time) * 1000.0
    
    top3_idx_effnet = np.argsort(preds_effnet)[::-1][:3]
    
    results["EfficientNetV2-B0"] = {
        "confidence": float(np.max(preds_effnet)),
        "latency_ms": round(effnet_latency_ms, 2),
        "top3_indices": top3_idx_effnet,
        "top3_probs": [float(preds_effnet[i]) for i in top3_idx_effnet],
        "all_probs": preds_effnet
    }
    
    return results
