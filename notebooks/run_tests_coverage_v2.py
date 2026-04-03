# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-monitoring — Test Coverage Expansion (April 2026)
# MAGIC
# MAGIC Runs the new `test_coverage_gaps_v2.py` suite covering previously untested
# MAGIC paths: BAWS private helpers, CUSUM LLO/pool helpers, multicalibration
# MAGIC degenerate cells, conformal SPC score_samples fallback, ModelMonitor
# MAGIC _make_decision unit tests and distribution variants, threshold boundary
# MAGIC conditions, and top-level import integration.
# MAGIC
# MAGIC Also runs the full test suite as a regression check.

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

# Run the new coverage tests first
result_new = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-monitoring/tests/test_coverage_gaps_v2.py",
        "-v", "--tb=short", "--no-header",
    ],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-monitoring",
)
print(result_new.stdout[-8000:])
print(result_new.stderr[-2000:] if result_new.stderr else "")

# COMMAND ----------

# Full test suite regression check
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
print(result_full.stdout[-8000:])
print(result_full.stderr[-2000:] if result_full.stderr else "")

# COMMAND ----------

# Summary
new_passed = result_new.returncode == 0
full_passed = result_full.returncode == 0

summary_lines = []
for line in result_new.stdout.split("\n"):
    if "passed" in line or "failed" in line or "error" in line.lower():
        summary_lines.append(f"[coverage_v2] {line}")

for line in result_full.stdout.split("\n"):
    if "passed" in line or "failed" in line or "error" in line.lower():
        summary_lines.append(f"[full suite] {line}")

summary = "\n".join(summary_lines) if summary_lines else "No summary lines found."
print(summary)

if not new_passed:
    raise RuntimeError("New coverage tests FAILED — see output above.")
if not full_passed:
    raise RuntimeError("Full test suite FAILED — see output above.")

dbutils.notebook.exit(summary)
