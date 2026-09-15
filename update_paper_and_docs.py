"""
update_paper_and_docs.py
========================
Safeguarded paper update helper. Loads verified metrics from results/
and runs verify_claims.py to guarantee zero fabrication.
"""

import sys
import subprocess

def run_verification():
    print("[update_paper_and_docs] Running mandatory verification gate (verify_claims.py)...")
    res = subprocess.run([sys.executable, "-X", "utf8", "verify_claims.py"], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("[ERROR] Verification failed! Paper contains unverified numeric claims.")
        sys.exit(1)
    else:
        print("[SUCCESS] All numeric claims in IEEE_Paper_Draft.md verified against real result files.")

if __name__ == "__main__":
    run_verification()

