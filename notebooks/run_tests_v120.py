# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-monitoring v0.12.0 — ConformedControlChart and ConformedProcessMonitor
# MAGIC
# MAGIC Runs the full test suite for the new `conformal_spc` module added in v0.12.0,
# MAGIC plus all existing tests to verify no regressions.
# MAGIC
# MAGIC Reference: Burger (2025), arXiv:2512.23602.

# COMMAND ----------

# MAGIC %pip install polars>=1.0 scipy>=1.12 sortedcontainers>=2.4 scikit-learn pytest pytest-cov

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

# Run only the new conformal_spc tests first for fast feedback
result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-monitoring/tests/test_conformal_spc.py",
        "-v", "--tb=short", "--no-header",
    ],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-monitoring",
)
print(result.stdout[-5000:])
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
print(result_full.stdout[-8000:])
print(result_full.stderr[-2000:] if result_full.stderr else "")

# COMMAND ----------

# Summary for notebook output (picked up by Jobs API)
passed = "passed" in result.stdout and result.returncode == 0
full_passed = result_full.returncode == 0

summary_lines = []
for line in result.stdout.split("\n"):
    if "passed" in line or "failed" in line or "error" in line.lower():
        summary_lines.append(f"[conformal_spc] {line}")

for line in result_full.stdout.split("\n"):
    if "passed" in line or "failed" in line or "error" in line.lower():
        summary_lines.append(f"[full suite] {line}")

summary = "\n".join(summary_lines) if summary_lines else "No summary lines found."
print(summary)

if not passed:
    raise RuntimeError("conformal_spc tests FAILED — see output above.")
if not full_passed:
    raise RuntimeError("Full test suite FAILED — see output above.")

dbutils.notebook.exit(summary)
