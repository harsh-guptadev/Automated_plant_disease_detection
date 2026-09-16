"""
verify_claims.py
================
Mandatory verification gate script that scans manuscript drafts (IEEE_Paper_Draft.md)
for every numerical metric claim, cross-references against saved JSON files in results/,
and outputs a strict audit report (CONFIRMED vs UNVERIFIED).
"""

import os
import re
import json
import glob
import sys

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_all_results_data(results_dir="results"):
    """Loads all JSON data files from results directory into a flattened lookup dictionary."""
    results_map = {}
    json_files = glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)
    
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                rel_path = os.path.relpath(jf, start=results_dir).replace("\\", "/")
                results_map[rel_path] = data
        except Exception as e:
            print(f"[Warning] Error reading {jf}: {e}")
            
    return results_map


def check_mechanical_cycle(sequence, min_len=6):
    """
    Checks if a sequence of numeric values repeats in a mechanical periodic cycle.
    Returns (is_cyclical, period, pattern) if a cycle of length k <= len(sequence)//2 exists.
    """
    if len(sequence) < min_len:
        return False, None, None
    n = len(sequence)
    for k in range(1, (n // 2) + 1):
        if all(sequence[i] == sequence[i % k] for i in range(n)):
            return True, k, sequence[:k]
    return False, None, None


def check_results_provenance(results_dir="results"):
    """
    Performs strict provenance audits across all results files.
    FAILS if:
      1. A results file reports human ratings while raw_response fields are empty or absent.
      2. A results file claims an experiment COMPLETE with no corresponding saved model weights,
         probability arrays (.npz), or raw outputs on disk.
      3. Rating/score values repeat in a mechanical cycle across items.
    """
    failures = []
    passes = []
    
    json_files = glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)
    
    for jf in json_files:
        rel_path = os.path.relpath(jf, start=results_dir).replace("\\", "/")
        file_dir = os.path.dirname(jf)
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            failures.append(f"[JSON ERROR] {rel_path}: Failed to parse JSON: {e}")
            continue

        # -------------------------------------------------------------
        # Check 1: Human ratings provenance (raw_response requirement)
        # -------------------------------------------------------------
        per_items = data.get("per_item_results", [])
        human_ratings_reported = False
        
        # Check summary metrics for reported human ratings
        summary = data.get("summary_metrics", {})
        for strat_key, strat_val in summary.items():
            if isinstance(strat_val, dict):
                if strat_val.get("mean_human_score") is not None:
                    human_ratings_reported = True
        if data.get("mean_human_score") is not None:
            human_ratings_reported = True

        # Check per_item_results
        rated_items_count = 0
        missing_raw_response_count = 0
        for item in per_items:
            for k, v in item.items():
                if isinstance(v, dict) and v.get("human_rating_score") is not None:
                    human_ratings_reported = True
                    rated_items_count += 1
                    raw_resp = v.get("raw_response") or v.get("raw_output") or item.get("raw_response")
                    if not raw_resp or str(raw_resp).strip() == "":
                        missing_raw_response_count += 1

        if human_ratings_reported and missing_raw_response_count > 0:
            failures.append(
                f"[PROVENANCE FAIL] {rel_path}: Reports human ratings ({rated_items_count} rated items), "
                f"but 'raw_response' field is empty or absent on {missing_raw_response_count} item(s)."
            )
        elif human_ratings_reported and rated_items_count == 0:
            failures.append(
                f"[PROVENANCE FAIL] {rel_path}: Summary claims human ratings, but no per-item rated entries exist."
            )
        elif not human_ratings_reported:
            passes.append(f"[PROVENANCE PASS] {rel_path}: No unearned human ratings reported.")

        # -------------------------------------------------------------
        # Check 2: Experiment COMPLETE status backed by saved artifacts
        # -------------------------------------------------------------
        def find_status_claims(obj, prefix=""):
            claims = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    curr_k = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, str) and v.strip().upper() == "COMPLETE":
                        claims.append((curr_k, v))
                    elif isinstance(v, (dict, list)):
                        claims.extend(find_status_claims(v, curr_k))
            elif isinstance(obj, list):
                for i, elem in enumerate(obj):
                    claims.extend(find_status_claims(elem, f"{prefix}[{i}]"))
            return claims

        status_claims = find_status_claims(data)
        for key_name, val in status_claims:
            # Determine expected artifacts
            key_lower = key_name.lower()
            if "effnet" in key_lower or "efficientnet" in key_lower:
                weights_exist = any(
                    glob.glob(os.path.join(results_dir, "**", "*efficientnet*weight*"), recursive=True) +
                    glob.glob(os.path.join(results_dir, "**", "*efficientnet*.h5"), recursive=True) +
                    glob.glob(os.path.join(results_dir, "**", "*efficientnet*.keras"), recursive=True)
                )
                probs_exist = any(
                    glob.glob(os.path.join(results_dir, "**", "*efficientnet*prob*.npz"), recursive=True)
                )
                metrics_exist = any(
                    glob.glob(os.path.join(results_dir, "**", "*efficientnet*metric*.json"), recursive=True)
                )
                if not (weights_exist and (probs_exist or metrics_exist)):
                    failures.append(
                        f"[PROVENANCE FAIL] {rel_path}: Claims '{key_name}: COMPLETE' but required model "
                        f"weights or probability arrays were not found on disk."
                    )
                else:
                    passes.append(f"[PROVENANCE PASS] {rel_path}: '{key_name}: COMPLETE' backed by saved weights and probs.")
            elif "resnet" in key_lower:
                probs_exist = any(
                    glob.glob(os.path.join(results_dir, "**", "*resnet*prob*.npz"), recursive=True)
                )
                if not probs_exist:
                    failures.append(
                        f"[PROVENANCE FAIL] {rel_path}: Claims '{key_name}: COMPLETE' but ResNet50 test probs (.npz) missing."
                    )
                else:
                    passes.append(f"[PROVENANCE PASS] {rel_path}: '{key_name}: COMPLETE' backed by saved probability array.")
            else:
                # Generic COMPLETE claim: verify at least one model/prob/artifact exists in same folder or results
                sibling_artifacts = [
                    f for f in os.listdir(file_dir)
                    if f.endswith((".npz", ".h5", ".weights.h5", ".keras", ".png", ".csv"))
                ]
                if not sibling_artifacts:
                    failures.append(
                        f"[PROVENANCE FAIL] {rel_path}: Claims '{key_name}: COMPLETE' but directory {file_dir} "
                        f"has no supporting weights, probability arrays (.npz), or evaluation artifacts."
                    )
                else:
                    passes.append(f"[PROVENANCE PASS] {rel_path}: '{key_name}: COMPLETE' verified with local artifacts.")

        # -------------------------------------------------------------
        # Check 3: Mechanical cycle detection in rating/score series
        # -------------------------------------------------------------
        if per_items:
            # Collect score series across items
            score_series_map = {}
            for item in per_items:
                for strat_name, strat_data in item.items():
                    if isinstance(strat_data, dict):
                        for field_name, field_val in strat_data.items():
                            if "score" in field_name.lower() or "rating" in field_name.lower():
                                if isinstance(field_val, (int, float)):
                                    series_key = f"{strat_name}.{field_name}"
                                    score_series_map.setdefault(series_key, []).append(field_val)

            for series_name, seq in score_series_map.items():
                is_cyclical, period, pattern = check_mechanical_cycle(seq, min_len=6)
                if is_cyclical:
                    failures.append(
                        f"[PROVENANCE FAIL] {rel_path}: Values in '{series_name}' repeat in a mechanical cycle "
                        f"(period={period}: {pattern}) across {len(seq)} items. Flagged as fabricated/unearned."
                    )
                else:
                    passes.append(f"[PROVENANCE PASS] {rel_path}: '{series_name}' has organic non-cyclical distribution.")

    return len(failures) == 0, failures, passes

def extract_numeric_tokens(text):
    """
    Extracts numerical statements (percentages, ratios, decimal numbers) from markdown text.
    Filters out numbers occurring inside URLs/DOIs.
    """
    pattern = r'(\d+\.\d+%|\d+\.\d+|\b\d+/\d+\b)'
    lines = text.splitlines()
    claims = []
    
    in_code_block = False
    for line_num, line in enumerate(lines, 1):
        line_str = line.strip()
        if line_str.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        # Strip URLs to avoid parsing DOI numbers
        line_clean_urls = re.sub(r'https?://\S+', '', line_str)
        line_clean_urls = re.sub(r'doi\.org/\S+', '', line_clean_urls)

        matches = re.findall(pattern, line_clean_urls)
        for m in matches:
            val_clean = m.replace("%", "").strip()
            try:
                num_val = float(val_clean) if "/" not in val_clean else val_clean
                claims.append({
                    "line_num": line_num,
                    "line_text": line_str,
                    "raw_token": m,
                    "clean_value": num_val
                })
            except ValueError:
                continue

    return claims

def search_value_in_json(target_val, json_obj, path_prefix=""):
    """Recursively searches for a numerical value in a JSON structure."""
    found_paths = []
    
    # Handle ratio fractions like "70/15" in "70/15/15" or "96/100"
    if isinstance(target_val, str) and "/" in target_val:
        if target_val in str(json_obj):
            found_paths.append((path_prefix, str(target_val)))
            return found_paths
        parts = target_val.split("/")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            num_a, num_b = float(parts[0]), float(parts[1])
            res_a = search_value_in_json(num_a, json_obj, path_prefix)
            res_b = search_value_in_json(num_b, json_obj, path_prefix)
            if res_a and res_b:
                found_paths.append((res_a[0][0], f"{res_a[0][1]}/{res_b[0][1]}"))
                return found_paths

    if isinstance(json_obj, dict):
        for k, v in json_obj.items():
            current_path = f"{path_prefix}.{k}" if path_prefix else k
            found_paths.extend(search_value_in_json(target_val, v, current_path))
    elif isinstance(json_obj, list):
        for idx, item in enumerate(json_obj):
            current_path = f"{path_prefix}[{idx}]"
            found_paths.extend(search_value_in_json(target_val, item, current_path))
    else:
        # Check matching float or string values
        try:
            if isinstance(target_val, float) and isinstance(json_obj, (int, float)):
                # Match floating values with tight tolerance or exact percentage scaling
                if abs(json_obj - target_val) < 1e-4 or abs((json_obj * 100.0) - target_val) < 1e-2 or abs(json_obj - (target_val / 100.0)) < 1e-4:
                    found_paths.append((path_prefix, json_obj))
            elif str(target_val) in str(json_obj):
                found_paths.append((path_prefix, json_obj))
        except Exception:
            pass
            
    return found_paths

def run_verification_gate(doc_path="IEEE_Paper_Draft.md", results_dir="results"):
    print(f"\n================ MANDATORY VERIFICATION GATE AUDIT ================")
    print(f"Manuscript Document:  {doc_path}")
    print(f"Results Directory:    {results_dir}")
    print(f"====================================================================")

    # 1. PROVENANCE INTEGRITY AUDIT
    print("\n--- STAGE 1: PROVENANCE & EARNED-DATA AUDIT ---")
    prov_ok, prov_failures, prov_passes = check_results_provenance(results_dir)
    for p in prov_passes:
        print(f"  [OK] {p}")
    if prov_failures:
        print("\n  [PROVENANCE FAILURES DETECTED]:")
        for f in prov_failures:
            print(f"  {f}")
    else:
        print("\n  [PROVENANCE SUCCESS] All results files exhibit valid provenance and earned artifacts.")

    # 2. NUMERIC CLAIMS CROSS-REFERENCE AUDIT
    print("\n--- STAGE 2: NUMERIC CLAIMS CROSS-REFERENCE AUDIT ---")
    if not os.path.exists(doc_path):
        print(f"[Error] Manuscript file {doc_path} not found!")
        sys.exit(1)

    with open(doc_path, "r", encoding="utf-8") as f:
        doc_text = f.read()

    results_data = load_all_results_data(results_dir)
    claims = extract_numeric_tokens(doc_text)

    confirmed_count = 0
    unverified_count = 0
    pending_count = 0

    audit_logs = []

    for c in claims:
        line_txt = c["line_text"]
        
        # Check if line explicitly notes PENDING or PLANNED
        if "[PENDING" in line_txt.upper() or "PLANNED" in line_txt.upper() or "NOT YET EXECUTED" in line_txt.upper():
            pending_count += 1
            audit_logs.append(f"Line {c['line_num']:<3} | PENDING ACKNOWLEDGED: '{c['raw_token']}' in '{line_txt[:70]}...'")
            continue

        # Search across all loaded JSON files
        token_found = False
        matching_citations = []

        for file_rel, data_json in results_data.items():
            matches = search_value_in_json(c["clean_value"], data_json)
            if matches:
                token_found = True
                for key_path, val in matches:
                    matching_citations.append(f"{file_rel} -> {key_path}: {val}")

        if token_found:
            confirmed_count += 1
            audit_logs.append(f"Line {c['line_num']:<3} | [CONFIRMED]: '{c['raw_token']}' -> {matching_citations[0]}")
        else:
            unverified_count += 1
            audit_logs.append(f"Line {c['line_num']:<3} | [UNVERIFIED]: '{c['raw_token']}' in '{line_txt}'")

    print("\n--- DETAILED NUMERIC AUDIT LOG ---")
    for log in audit_logs:
        print(log)

    print("\n================ VERIFICATION GATE SUMMARY ================")
    print(f"Stage 1 Provenance Status:     {'PASSED (0 failures)' if prov_ok else f'FAILED ({len(prov_failures)} failures)'}")
    print(f"Total Numeric Claims Analyzed: {len(claims)}")
    print(f"CONFIRMED Claims:              {confirmed_count}")
    print(f"PENDING/PLANNED Statements:    {pending_count}")
    print(f"UNVERIFIED Claims:             {unverified_count}")
    print(f"===========================================================")

    gate_passed = True
    if not prov_ok:
        print(f"\n[FAIL] Verification Gate Blocked on Provenance Integrity! Found {len(prov_failures)} provenance failure(s):")
        for f in prov_failures:
            print(f"  - {f}")
        gate_passed = False

    if unverified_count > 0:
        print(f"\n[FAIL] Verification Gate Blocked on Numerical Mismatch! {unverified_count} UNVERIFIED numeric claims found.")
        print("You must fix the document by removing unverified claims or citing exact results files.")
        gate_passed = False

    if gate_passed:
        print("\n[SUCCESS] Verification Gate Passed! 100% of numeric claims trace to saved results files and all provenance checks pass.")
        return True
    else:
        return False

if __name__ == "__main__":
    success = run_verification_gate()
    if not success:
        sys.exit(1)
