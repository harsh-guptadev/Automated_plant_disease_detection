"""
src/explainability/gradcam_pp.py
================================
Grad-CAM++ (Generalized Grad-CAM) implementation for fine-grained multi-lesion localization.
Calculates 2nd and 3rd order gradients to weight feature maps for spatial attribution.
"""

import numpy as np
import cv2
import tensorflow as tf

def make_gradcam_pp_heatmap(img_array, model, last_conv_layer_name='conv5_block3_out', pred_index=None):
    """
    Generates a Grad-CAM++ heatmap for an input image array (1, 224, 224, 3).
    Uses nested GradientTapes to compute second- and third-order gradients.
    Falls back to standard Grad-CAM weighting if higher-order grads are None.
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape3:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                conv_outputs, predictions = grad_model(img_array)
                tape1.watch(conv_outputs)
                tape2.watch(conv_outputs)
                tape3.watch(conv_outputs)
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0])
                class_output = predictions[:, pred_index]
            grads_1 = tape1.gradient(class_output, conv_outputs)
        grads_2 = tape2.gradient(grads_1, conv_outputs) if grads_1 is not None else None
    grads_3 = tape3.gradient(grads_2, conv_outputs) if grads_2 is not None else None

    conv_out = conv_outputs[0]
    g1 = grads_1[0] if grads_1 is not None else None

    # Fallback to standard Grad-CAM if higher-order grads unavailable
    if grads_2 is None or grads_3 is None or g1 is None:
        pooled = tf.reduce_mean(g1 if g1 is not None else tf.zeros_like(conv_out), axis=(0, 1))
        cam = tf.reduce_sum(conv_out * pooled, axis=-1)
        cam = np.maximum(cam.numpy(), 0)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        return cam

    g2 = grads_2[0]
    g3 = grads_3[0]

    # Alpha coefficients for Grad-CAM++
    sum_conv = tf.reduce_sum(conv_out, axis=(0, 1), keepdims=True)
    denom = 2.0 * g2 + sum_conv * g3
    denom = tf.where(tf.abs(denom) > 1e-8, denom, tf.ones_like(denom))
    alpha = g2 / denom

    # Weight by positive first-order gradients
    pos_grads = tf.maximum(g1, 0.0)
    weights = tf.reduce_sum(alpha * pos_grads, axis=(0, 1))
    cam = tf.reduce_sum(weights * conv_out, axis=-1)

    cam = np.maximum(cam.numpy(), 0)
    if np.max(cam) != 0:
        cam = cam / np.max(cam)

    return cam

def compare_gradcam_methods(img_array, model, last_conv_layer_name='conv5_block3_out'):
    """
    Generates side-by-side Standard Grad-CAM vs Grad-CAM++ heatmaps.
    """
    # Standard Grad-CAM
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap_std = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap_std = tf.squeeze(heatmap_std)
    heatmap_std = heatmap_std.numpy()

    # Grad-CAM++
    heatmap_pp = make_gradcam_pp_heatmap(img_array, model, last_conv_layer_name)

    return heatmap_std, heatmap_pp

def generate_and_save_xai_comparison(
    test_split_json="data_splits/test_split.json",
    output_dir="results/week3"
):
    import os
    import json
    import matplotlib.pyplot as plt
    from PIL import Image
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from src.evaluation.evaluate_resnet import load_resnet_model

    os.makedirs(output_dir, exist_ok=True)
    
    with open(test_split_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Pick 2 sample test images
    model = load_resnet_model()
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=150)
    
    for i, idx_img in enumerate([0, 5]):
        sample_path = test_data[idx_img]["path"]
        img_pil = Image.open(sample_path).convert("RGB").resize((224, 224))
        img_arr = np.array(img_pil, dtype=np.float32)
        img_batch = np.expand_dims(img_arr, axis=0)
        img_pre = preprocess_input(img_batch.copy())

        heatmap_std, heatmap_pp = compare_gradcam_methods(img_pre, model)

        # Plot Original
        axes[i, 0].imshow(img_pil)
        axes[i, 0].set_title(f"Sample {i+1}: Original Leaf Image", fontsize=10)
        axes[i, 0].axis("off")

        # Plot Standard Grad-CAM
        axes[i, 1].imshow(img_pil)
        axes[i, 1].imshow(heatmap_std, cmap="jet", alpha=0.5)
        axes[i, 1].set_title("Standard Grad-CAM", fontsize=10)
        axes[i, 1].axis("off")

        # Plot Grad-CAM++
        axes[i, 2].imshow(img_pil)
        axes[i, 2].imshow(heatmap_pp, cmap="jet", alpha=0.5)
        axes[i, 2].set_title("Grad-CAM++ (Multi-Lesion)", fontsize=10)
        axes[i, 2].axis("off")

    plt.tight_layout()
    out_png = os.path.join(output_dir, "xai_comparison.png")
    plt.savefig(out_png)
    plt.close()

    print(f"[Grad-CAM++] Saved qualitative XAI comparison plot to {out_png}")
    return out_png

if __name__ == "__main__":
    generate_and_save_xai_comparison()
