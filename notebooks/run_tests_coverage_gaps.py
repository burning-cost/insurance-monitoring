# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-monitoring — Coverage Gap Tests
# MAGIC
# MAGIC Runs the new `test_coverage_gaps.py` tests (targeting thresholds boundaries,
# MAGIC drift CSI/Wasserstein, BAWS block-length internals, CUSUM LLO/_resample_mc_pool,
# MAGIC multicalibration frozen edges / JSON serialisation, and ModelMonitor
# MAGIC _make_decision all branches + gamma/tweedie distributions).
# MAGIC
# MAGIC Also runs the full suite to verify no regressions.

# COMMAND ----------

# MAGIC %pip install polars>=1.0 scipy>=1.12 sortedcontainers>=2.4 statsmodels>=0.14 scikit-learn pytest pytest-cov matplotlib

# COMMAND ----------

import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e",
     "/Workspace/insurance-monitoring", "--quiet"],
    capture_output=True, text=True
)
print(result.stdout[-2000:] if result.stdout else "")
print(result.stderr[-2000:] if result.stderr else "")

# COMMAND ----------

import subprocess
import sys

# Run only the new gap tests first for fast feedback
result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-monitoring/tests/test_coverage_gaps.py",
        "-v", "--tb=short", "--no-header",
    ],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-monitoring",
)
print(result.stdout[-8000:])
print(result.stderr[-2000:] if result.stderr else "")

# COMMAND ----------

# Full test suite — regression check
result_full = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-monitoring/tests/",
        "-v", "--tb=short", "--no-header",
        "--ignore=/Workspace/insurance-monitoring/tests/test_mlflow_tracker.py",
        "-q",
    ],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-monitoring",
)
print(result_full.stdout[-10000:])
print(result_full.stderr[-2000:] if result_full.stderr else "")

# COMMAND ----------

# Summary for notebook output (picked up by Jobs API)
gap_passed = result.returncode == 0
full_passed = result_full.returncode == 0

summary_lines = []
for line in result.stdout.split("\n"):
    if "passed" in line or "failed" in line or "error" in line.lower():
        summary_lines.append(f"[coverage_gaps] {line}")

for line in result_full.stdout.split("\n"):
    if "passed" in line or "failed" in line or "error" in line.lower():
        summary_lines.append(f"[full suite] {line}")

summary = "\n".join(summary_lines) if summary_lines else "No summary lines found."
print(summary)

if not gap_passed:
    raise RuntimeError("Coverage gap tests FAILED — see output above.")
if not full_passed:
    raise RuntimeError("Full test suite FAILED — see output above.")

dbutils.notebook.exit(summary)
