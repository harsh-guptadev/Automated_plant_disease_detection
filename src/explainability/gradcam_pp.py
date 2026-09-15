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
    """
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape(persistent=True) as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]

    # First-order gradients
    grads_1 = tape.gradient(class_output, conv_outputs)
    # Second-order gradients
    grads_2 = tape.gradient(grads_1, conv_outputs)
    # Third-order gradients
    grads_3 = tape.gradient(grads_2, conv_outputs)

    del tape

    # Global sum of feature maps
    conv_outputs = conv_outputs[0]
    grads_1 = grads_1[0]
    grads_2 = grads_2[0]
    grads_3 = grads_3[0]

    # Calculate alpha coefficients (weights for Grad-CAM++)
    # alpha_ij^kc = grad_2 / (2 * grad_2 + sum(conv_outputs * grad_3))
    sum_conv = tf.reduce_sum(conv_outputs, axis=(0, 1), keepdims=True)
    denom = 2.0 * grads_2 + sum_conv * grads_3
    denom = tf.where(denom != 0.0, denom, tf.ones_like(denom))
    
    alpha = grads_2 / denom

    # Positive gradients (ReLU on grads_1)
    pos_grads = tf.maximum(grads_1, 0.0)
    weights = tf.reduce_sum(alpha * pos_grads, axis=(0, 1))

    # Weighted sum of feature maps
    cam = tf.reduce_sum(weights * conv_outputs, axis=-1)

    # Apply ReLU to cam output
    cam = np.maximum(cam.numpy(), 0)

    # Normalize heatmap between 0 and 1
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
    heatmap_std = tf.maximum(heatmap_std, 0) / tf.math.reduce_max(heatmap_std)
    heatmap_std = heatmap_std.numpy()

    # Grad-CAM++
    heatmap_pp = make_gradcam_pp_heatmap(img_array, model, last_conv_layer_name)

    return heatmap_std, heatmap_pp
