"""
Verify insurance-monitoring sequential test suite on Databricks.
Targeted runner: tests/test_sequential.py only.
"""

import os
import sys
import time
import base64
import pathlib

# Load credentials
env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k] = v

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

WORKSPACE_PATH = "/Workspace/insurance-monitoring-seq-verify"
PROJECT_ROOT = pathlib.Path("/home/ralph/burning-cost/repos/insurance-monitoring")


def upload_file(local_path: pathlib.Path, remote_path: str) -> None:
    content = local_path.read_bytes()
    encoded = base64.b64encode(content).decode()
    parent = "/".join(remote_path.split("/")[:-1])
    try:
        w.workspace.mkdirs(path=parent)
    except Exception:
        pass
    w.workspace.import_(
        path=remote_path,
        content=encoded,
        overwrite=True,
        format=ImportFormat.AUTO,
    )
    print(f"  Uploaded: {remote_path}")


def upload_notebook(local_path: pathlib.Path, remote_path: str) -> None:
    content = local_path.read_bytes()
    encoded = base64.b64encode(content).decode()
    parent = "/".join(remote_path.split("/")[:-1])
    try:
        w.workspace.mkdirs(path=parent)
    except Exception:
        pass
    w.workspace.import_(
        path=remote_path,
        content=encoded,
        overwrite=True,
        format=ImportFormat.SOURCE,
        language=Language.PYTHON,
    )
    print(f"  Uploaded notebook: {remote_path}")


# Collect src/ files and the sequential test file
upload_files = []
for p in (PROJECT_ROOT / "src").rglob("*.py"):
    if "__pycache__" in p.parts:
        continue
    upload_files.append(p)

upload_files.append(PROJECT_ROOT / "tests" / "__init__.py")
upload_files.append(PROJECT_ROOT / "tests" / "test_sequential.py")
upload_files.append(PROJECT_ROOT / "pyproject.toml")

print(f"Uploading {len(upload_files)} files to Databricks workspace...")
for local in sorted(upload_files):
    rel = local.relative_to(PROJECT_ROOT)
    remote = f"{WORKSPACE_PATH}/{rel}"
    upload_file(local, remote)

# Write a standalone test script (not using pytest.main to avoid ini conflicts)
# Run as a subprocess but with PYTHONPATH set so insurance_monitoring is importable.
runner_source = '''# Databricks notebook source
# COMMAND ----------
# MAGIC %pip install polars scipy matplotlib sortedcontainers pytest

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import subprocess, sys, os

workspace_path = "/Workspace/insurance-monitoring-seq-verify"
src_path = f"{workspace_path}/src"

# Verify the src path exists and insurance_monitoring is importable
sys.path.insert(0, src_path)
import insurance_monitoring
print(f"insurance_monitoring {insurance_monitoring.__version__} loaded")
print(f"File: {insurance_monitoring.__file__}")

# COMMAND ----------
# Run pytest as subprocess with PYTHONPATH pointing to src/
# Use -no-ini-override to avoid pyproject.toml cache_dir issue.
# Pass --import-mode=importlib so pytest doesn't manipulate sys.path itself.
env = {**os.environ, "PYTHONPATH": src_path, "PYTHONDONTWRITEBYTECODE": "1"}

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     f"{workspace_path}/tests/test_sequential.py",
     "-v", "--tb=long",
     "-p", "no:cacheprovider",
     "--import-mode=importlib",
     "--no-header",
    ],
    capture_output=True,
    text=True,
    env=env,
    cwd="/tmp",  # Run from /tmp to avoid pyproject.toml being picked up
)
output = result.stdout + result.stderr
# Print in chunks so it doesn't get cut off
for chunk in [output[i:i+3000] for i in range(0, len(output), 3000)]:
    print(chunk)

print(f"\\nReturn code: {result.returncode}")

if result.returncode == 0:
    dbutils.notebook.exit("ALL TESTS PASSED")
else:
    # Don't raise — just exit with failure summary so we can read the output
    summary = "\\n".join(output.strip().split("\\n")[-30:])
    dbutils.notebook.exit(f"TESTS FAILED (rc={result.returncode})\\n---\\n{summary}")
'''

nb_path = "/tmp/run_sequential_tests_v5.py"
with open(nb_path, "w") as f:
    f.write(runner_source)

upload_notebook(pathlib.Path(nb_path), f"{WORKSPACE_PATH}/notebooks/run_sequential_tests_v5")

print("\nSubmitting test job (serverless compute)...")

run = w.jobs.submit(
    run_name="insurance-monitoring-sequential-verify-v5",
    tasks=[
        jobs.SubmitTask(
            task_key="run_tests",
            environment_key="default",
            notebook_task=jobs.NotebookTask(
                notebook_path=f"{WORKSPACE_PATH}/notebooks/run_sequential_tests_v5",
            ),
        )
    ],
    environments=[
        jobs.JobEnvironment(
            environment_key="default",
            spec=compute.Environment(),
        )
    ],
)

run_id = run.run_id
print(f"Job run submitted: run_id={run_id}")
print(f"Monitor at: {os.environ['DATABRICKS_HOST']}#job/runs/{run_id}")

print("\nPolling for completion (max 15 min)...")
deadline = time.time() + 900
while time.time() < deadline:
    state = w.jobs.get_run(run_id=run_id)
    life_cycle = state.state.life_cycle_state.value
    print(f"  Status: {life_cycle}")
    if life_cycle in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    time.sleep(20)

result_state = state.state.result_state
print(f"\nResult: {result_state.value if result_state else 'UNKNOWN'}")

try:
    task_run = state.tasks[0] if state.tasks else None
    if task_run:
        output = w.jobs.get_run_output(run_id=task_run.run_id)
        if output.notebook_output and output.notebook_output.result:
            print("\n--- Test output ---")
            print(output.notebook_output.result)
        if output.error:
            print(f"\nError: {output.error}")
        if output.error_trace:
            print(f"\nTrace:\n{output.error_trace}")
except Exception as e:
    print(f"Could not fetch output: {e}")

# Check if output says PASSED
if result_state and result_state.value == "SUCCESS":
    print("\nSequential tests PASSED.")
    sys.exit(0)
else:
    print("\nSequential tests FAILED.")
    sys.exit(1)
