# Databricks notebook source
# COMMAND ----------
# MAGIC %pip install polars scipy matplotlib pytest pytest-cov sortedcontainers pandas

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import subprocess
import sys
import os

# Add the src directory to sys.path so insurance_monitoring is importable
sys.path.insert(0, "/Workspace/insurance-monitoring/src")

# Verify import works
import insurance_monitoring
print(f"insurance_monitoring loaded from: {insurance_monitoring.__file__}")
print(f"Version: {insurance_monitoring.__version__}")

# Verify BAWSMonitor importable from top-level package
from insurance_monitoring import BAWSMonitor, BAWSResult
print(f"BAWSMonitor importable: {BAWSMonitor}")
print(f"BAWSResult importable: {BAWSResult}")

# Verify scoring functions importable directly
from insurance_monitoring.baws import fissler_ziegel_score, asymm_abs_loss
print(f"fissler_ziegel_score importable: {fissler_ziegel_score}")
print(f"asymm_abs_loss importable: {asymm_abs_loss}")

# COMMAND ----------
# Databricks workspace filesystem does not support __pycache__ creation.
# Disable bytecode writing and use a /tmp cache directory for pytest.
env = {
    **os.environ,
    "PYTHONPATH": "/Workspace/insurance-monitoring/src",
    "PYTHONDONTWRITEBYTECODE": "1",
}

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-monitoring/tests/test_baws.py",
        "-v", "--tb=long",
        "--import-mode=importlib",
        "-p", "no:cacheprovider",
        f"--rootdir=/Workspace/insurance-monitoring",
    ],
    capture_output=True,
    text=True,
    env=env,
)
output = result.stdout + "\n" + result.stderr
print(output)
dbutils.notebook.exit(output[-5000:] if len(output) > 5000 else output)
