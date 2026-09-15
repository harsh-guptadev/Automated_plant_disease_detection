"""
src/evaluation/benchmark_comparison.py
======================================
Comparative evaluation module producing side-by-side benchmark metrics
for ResNet50 Baseline vs EfficientNetV2-B0 on the 15% TEST set.

NOTE: This script only includes EfficientNetV2-B0 rows when a real
results file (results/week2/efficientnetv2_test_metrics.json) exists.
It does NOT fall back to hardcoded placeholder numbers.
"""

import os
import json
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def generate_comparison_table(
    resnet_json="results/week1/resnet50_test_metrics.json",
    effnet_json="results/week2/efficientnetv2_test_metrics.json",
    output_dir="results/week2"
):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(resnet_json):
        raise FileNotFoundError(
            f"ResNet50 results file not found: {resnet_json}\n"
            "Run src/evaluation/evaluate_resnet.py first."
        )

    with open(resnet_json, "r", encoding="utf-8") as f:
        res_data = json.load(f)

    comparison_list = [
        {
            "Architecture": res_data.get("model_name", "ResNet50 Baseline"),
            "Params": res_data.get("parameters_formatted", "25.6M"),
            "Top-1 Accuracy (%)": f"{res_data.get('top1_accuracy', 0) * 100:.2f}%",
            "Macro F1": f"{res_data.get('macro_f1_score', 0):.4f}",
            "Weighted F1": f"{res_data.get('weighted_f1_score', 0):.4f}",
            "Macro Precision": f"{res_data.get('macro_precision', 0):.4f}",
            "Macro Recall": f"{res_data.get('macro_recall', 0):.4f}",
            "Avg Latency (ms)": res_data.get("avg_inference_latency_ms", "N/A"),
            "Relative Compute Reduction": "Baseline (1.0x)"
        }
    ]

    effnet_status = "PENDING"
    if os.path.exists(effnet_json):
        with open(effnet_json, "r", encoding="utf-8") as f:
            eff_data = json.load(f)
        comparison_list.append({
            "Architecture": eff_data.get("model_name", "EfficientNetV2-B0"),
            "Params": eff_data.get("parameters_formatted", "5.9M"),
            "Top-1 Accuracy (%)": f"{eff_data.get('top1_accuracy', 0) * 100:.2f}%",
            "Macro F1": f"{eff_data.get('macro_f1_score', 0):.4f}",
            "Weighted F1": f"{eff_data.get('weighted_f1_score', 0):.4f}",
            "Macro Precision": f"{eff_data.get('macro_precision', 0):.4f}",
            "Macro Recall": f"{eff_data.get('macro_recall', 0):.4f}",
            "Avg Latency (ms)": eff_data.get("avg_inference_latency_ms", "N/A"),
            "Relative Compute Reduction": "~77% fewer params vs ResNet50"
        })
        effnet_status = "COMPLETE"
    else:
        print(f"[Benchmark Comparison] EfficientNetV2-B0 results not found at {effnet_json}.")
        print("[Benchmark Comparison] Generating ResNet50-only table. Run train_efficientnet.py to complete.")
        comparison_list.append({
            "Architecture": "EfficientNetV2-B0",
            "Params": "5.9M",
            "Top-1 Accuracy (%)": "PENDING — training not yet executed",
            "Macro F1": "PENDING",
            "Weighted F1": "PENDING",
            "Macro Precision": "PENDING",
            "Macro Recall": "PENDING",
            "Avg Latency (ms)": "PENDING",
            "Relative Compute Reduction": "~77% fewer params vs ResNet50 (theoretical)"
        })

    json_path = os.path.join(output_dir, "model_comparison_table.json")

    output = {
        "note": "EfficientNetV2-B0 row is PENDING until training is executed. ResNet50 results are from real evaluation.",
        "effnet_status": effnet_status,
        "comparison": comparison_list
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Also save CSV if pandas is available
    try:
        import pandas as pd
        df_comp = pd.DataFrame(comparison_list)
        csv_path = os.path.join(output_dir, "model_comparison_table.csv")
        df_comp.to_csv(csv_path, index=False)
        print(f"\n================ MODEL COMPARISON SUMMARY ================")
        print(df_comp.to_string(index=False))
        print(f"==========================================================")
        print(f"[Benchmark Comparison] CSV saved to {csv_path}")
    except ImportError:
        print("[Benchmark Comparison] pandas not available, skipping CSV export")

    print(f"[Benchmark Comparison] JSON saved to {json_path}")
    return output


if __name__ == "__main__":
    generate_comparison_table()
