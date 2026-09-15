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

    print("\n--- DETAILED AUDIT LOG ---")
    for log in audit_logs:
        print(log)

    print("\n================ VERIFICATION GATE SUMMARY ================")
    print(f"Total Numeric Claims Analyzed: {len(claims)}")
    print(f"CONFIRMED Claims:           {confirmed_count}")
    print(f"PENDING/PLANNED Statements: {pending_count}")
    print(f"UNVERIFIED Claims:          {unverified_count}")
    print(f"===========================================================")

    if unverified_count > 0:
        print(f"\n[FAIL] Verification Gate Blocked! {unverified_count} UNVERIFIED numeric claims found.")
        print("You must fix the document by removing unverified claims or citing exact results files.")
        return False
    else:
        print("\n[SUCCESS] Verification Gate Passed! 100% of numeric claims trace to saved results files.")
        return True

if __name__ == "__main__":
    success = run_verification_gate()
    if not success:
        sys.exit(1)
