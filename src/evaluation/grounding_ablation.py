"""
src/evaluation/grounding_ablation.py
=====================================
Grounding Ablation Study — Knowledge Base Coverage Evaluation.

WHAT THIS SCRIPT ACTUALLY MEASURES (honestly):
  - For each of 18 benchmark disease/question pairs, it calls retrieve_disease_context()
    and records which structured KB fields (symptoms, chemical_treatment, organic_remedy,
    prevention) are populated vs. missing.
  - This measures RAG *retrieval coverage* from the curated rag_knowledge_base.json.
  - KB coverage rate is a real, verifiable proxy for grounding quality.

WHAT THIS SCRIPT DOES NOT MEASURE:
  - Strategy B (direct LLM generation) hallucination rate. That requires running a live
    LLM inference session and having a human rater assess each response. This cannot
    be simulated deterministically. Those rows are saved as null/PENDING in the output.
  - Human ratings (previously faked as 5.0 / 2.5 on a modulo pattern) are removed.

Run this script to generate results/week2/grounding_ablation_results.json.
A second pass (human_rating_pass.py, to be run manually by a team member) will
fill in the Strategy B slots.
"""

import os
import sys
import json

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rag_engine import retrieve_disease_context

BENCHMARK_QUERIES = [
    ("Apple___Apple_scab",            "What chemical fungicide or organic remedy should I apply for Apple scab?"),
    ("Apple___Black_rot",             "How can I prevent Black rot from spreading in my apple orchard?"),
    ("Corn_(maize)___Common_rust_",   "What are the early visual symptoms and chemical dosages for Corn common rust?"),
    ("Corn_(maize)___Northern_Leaf_Blight", "Is there an organic treatment protocol for Northern Leaf Blight in maize?"),
    ("Grape___Black_rot",             "What is the recommended application timing for black rot in grapes?"),
    ("Grape___Esca_(Black_Measles)",  "Can Esca black measles in grapevines be cured with fungicides?"),
    ("Potato___Early_blight",         "What copper-based fungicide rate is verified for potato early blight?"),
    ("Potato___Late_blight",          "What is the emergency management protocol for potato late blight?"),
    ("Tomato___Bacterial_spot",       "How do I control bacterial spot on tomato leaves organically?"),
    ("Tomato___Early_blight",         "What crop rotation period is needed for tomato early blight?"),
    ("Tomato___Late_blight",          "What chemical spray is recommended for tomato late blight?"),
    ("Tomato___Leaf_Mold",            "What relative humidity control prevents tomato leaf mold in greenhouses?"),
    ("Tomato___Septoria_leaf_spot",   "What fungicides are effective for septoria leaf spot on tomatoes?"),
    ("Tomato___Spider_mites Two-spotted_spider_mite", "What miticide or neem oil concentration controls two-spotted spider mites?"),
    ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Can TYLCV virus be cured after infection, or is vector control mandatory?"),
    ("Tomato___Tomato_mosaic_virus",  "How is tomato mosaic virus transmitted and disinfected?"),
    ("Pepper,_bell___Bacterial_spot", "What bactericide protocol works for bell pepper bacterial spot?"),
    ("Peach___Bacterial_spot",        "What spray schedule controls bacterial spot on peach trees?"),
]

KB_FIELDS = ["symptoms", "chemical_treatment", "organic_remedy", "prevention"]


def run_grounding_ablation_study(
    kb_path="rag_knowledge_base.json",
    output_dir="results/week2"
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Grounding Ablation] Evaluating {len(BENCHMARK_QUERIES)} benchmark queries...")
    print(f"[Grounding Ablation] Measuring KB retrieval coverage for Strategy A (Grounded RAG).")
    print(f"[Grounding Ablation] Strategy B (Direct LLM) ratings are PENDING human evaluation.\n")

    per_item_results = []
    total_fields = len(KB_FIELDS)
    grounded_coverage_sum = 0

    for idx, (disease_key, query_text) in enumerate(BENCHMARK_QUERIES):
        ctx = retrieve_disease_context(disease_key)

        # Measure which KB fields are populated (not empty/None)
        field_coverage = {}
        populated_count = 0
        for field in KB_FIELDS:
            value = ctx.get(field, None)
            is_populated = bool(value and str(value).strip() not in ("", "N/A", "Not available"))
            field_coverage[field] = {
                "populated": is_populated,
                "value_preview": str(value)[:80] if is_populated else None
            }
            if is_populated:
                populated_count += 1

        coverage_pct = round((populated_count / total_fields) * 100.0, 1)
        grounded_coverage_sum += coverage_pct

        item = {
            "query_id": idx + 1,
            "disease_class": disease_key,
            "user_query": query_text,
            "strategy_a_grounded_rag": {
                "method": "retrieve_disease_context() from rag_knowledge_base.json",
                "kb_fields_populated": populated_count,
                "kb_fields_total": total_fields,
                "kb_coverage_pct": coverage_pct,
                "field_detail": field_coverage,
                "note": "Coverage measures whether structured KB fields are available for grounding."
            },
            "strategy_b_direct_llm": {
                "method": "PENDING — requires live LLM inference + human rating",
                "factually_consistent_with_kb": None,
                "hallucinated_unverified_dosages": None,
                "human_rating_score": None,
                "note": "Run human_rating_pass.py after collecting real LLM responses from a team member."
            }
        }
        per_item_results.append(item)
        status = "FULL" if coverage_pct == 100.0 else f"PARTIAL ({coverage_pct}%)"
        print(f"  Query {idx+1:2d} | {disease_key:<45} | KB Coverage: {status}")

    n = len(BENCHMARK_QUERIES)
    mean_kb_coverage = round(grounded_coverage_sum / n, 1)
    full_coverage_count = sum(1 for r in per_item_results
                              if r["strategy_a_grounded_rag"]["kb_coverage_pct"] == 100.0)

    summary = {
        "evaluation_type": "KB_RETRIEVAL_COVERAGE",
        "note": (
            "This file records Strategy A (grounded RAG) KB coverage only. "
            "Strategy B (direct LLM) hallucination rates are PENDING a human evaluation pass. "
            "The previously saved version of this file used mechanically-assigned fake human ratings "
            "and has been deleted. This version records only verifiable retrieval facts."
        ),
        "total_benchmark_queries": n,
        "strategy_a_mean_kb_coverage_pct": mean_kb_coverage,
        "strategy_a_full_coverage_count": full_coverage_count,
        "strategy_a_full_coverage_rate_pct": round((full_coverage_count / n) * 100.0, 1),
        "strategy_b_hallucination_rate_pct": None,
        "strategy_b_factual_accuracy_pct": None,
        "strategy_b_status": "PENDING — human evaluation required",
        "per_item_results": per_item_results
    }

    out_file = os.path.join(output_dir, "grounding_ablation_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n================ GROUNDING ABLATION SUMMARY (KB COVERAGE ONLY) ================")
    print(f"Total Benchmark Queries:                {n}")
    print(f"Strategy A — Mean KB Coverage:          {mean_kb_coverage}%")
    print(f"Strategy A — Queries with Full Coverage:{full_coverage_count}/{n} "
          f"({summary['strategy_a_full_coverage_rate_pct']}%)")
    print(f"Strategy B — Hallucination Rate:        PENDING (human evaluation)")
    print(f"================================================================================")
    print(f"[Grounding Ablation] Saved to {out_file}")

    return summary


if __name__ == "__main__":
    run_grounding_ablation_study()
