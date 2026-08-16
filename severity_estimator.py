import numpy as np

def estimate_disease_severity(heatmap: np.ndarray, threshold: float = 0.35) -> dict:
    """
    Computes disease severity and affected surface area percentage from Grad-CAM activation heatmap.
    """
    if heatmap is None:
        return {
            "affected_percentage": 0.0,
            "severity_level": "Unknown",
            "badge_color": "#94a3b8",
            "urgency": "Low"
        }
    
    # Normalize heatmap between 0 and 1
    norm_heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    
    # Count pixels exceeding activation threshold
    active_pixels = np.sum(norm_heatmap >= threshold)
    total_pixels = norm_heatmap.size
    affected_percentage = float((active_pixels / total_pixels) * 100.0)
    
    # Determine severity tier
    if affected_percentage < 15.0:
        severity_level = "Mild / Early Stage"
        badge_color = "#10b981"  # Emerald green
        urgency = "Low - Preventative care recommended"
    elif 15.0 <= affected_percentage < 40.0:
        severity_level = "Moderate Infection"
        badge_color = "#f59e0b"  # Amber yellow
        urgency = "Medium - Action required within 48 hours"
    else:
        severity_level = "Severe Infection"
        badge_color = "#ef4444"  # Coral red
        urgency = "High - Immediate treatment mandatory"
        
    return {
        "affected_percentage": round(affected_percentage, 1),
        "severity_level": severity_level,
        "badge_color": badge_color,
        "urgency": urgency
    }
