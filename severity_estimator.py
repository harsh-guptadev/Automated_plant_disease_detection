import numpy as np

def estimate_disease_severity(heatmap: np.ndarray, threshold: float = 0.35, confidence: float = 1.0) -> dict:
    """
    Computes an Attention-Based Severity Heuristic derived from Grad-CAM activation area intensity.
    
    NOTE (Scientific Disclaimer):
    Grad-CAM measures CNN spatial feature attention, NOT ground-truth biological leaf lesion segmentation.
    This indicator serves as an experimental decision-support metric.
    """
    if heatmap is None or confidence < 0.60:
        return {
            "affected_percentage": 0.0,
            "severity_level": "Uncertain / Low Confidence",
            "badge_color": "#94a3b8",
            "urgency": "High Uncertainty - Manual Inspection Recommended",
            "is_uncertain": True
        }
    
    # Normalize heatmap between 0 and 1
    norm_heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    
    # Count pixels exceeding activation threshold
    active_pixels = np.sum(norm_heatmap >= threshold)
    total_pixels = norm_heatmap.size
    affected_percentage = float((active_pixels / total_pixels) * 100.0)
    
    # Determine attention-based severity tier
    if affected_percentage < 15.0:
        severity_level = "Localized Attention (Low)"
        badge_color = "#10b981"  # Emerald green
        urgency = "Low - Standard preventative care recommended"
    elif 15.0 <= affected_percentage < 40.0:
        severity_level = "Moderate Attention Area"
        badge_color = "#f59e0b"  # Amber yellow
        urgency = "Medium - Monitor crop and apply target treatment"
    else:
        severity_level = "Broad Feature Activation (High)"
        badge_color = "#ef4444"  # Coral red
        urgency = "High - Prompt agronomist consultation recommended"
        
    return {
        "affected_percentage": round(affected_percentage, 1),
        "severity_level": severity_level,
        "badge_color": badge_color,
        "urgency": urgency,
        "is_uncertain": False
    }

