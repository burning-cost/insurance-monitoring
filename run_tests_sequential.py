"""
Run insurance-monitoring sequential test suite on Databricks serverless compute.
Validates all 10 test cases from the mSPRT sequential testing spec.
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

WORKSPACE_PATH = "/Workspace/insurance-monitoring-seq"
PROJECT_ROOT = pathlib.Path("/home/ralph/repos/insurance-monitoring")


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


# Build the notebook content inline
NOTEBOOK_CONTENT = '''\
# Databricks notebook source
# COMMAND ----------
# MAGIC %pip install polars scipy matplotlib sortedcontainers pytest

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import subprocess
import sys
import os

workspace_path = "/Workspace/insurance-monitoring-seq"

# Add the src directory to sys.path so insurance_monitoring is importable
sys.path.insert(0, f"{workspace_path}/src")

# Verify import works
import insurance_monitoring
print(f"insurance_monitoring loaded from: {insurance_monitoring.__file__}")
print(f"Version: {insurance_monitoring.__version__}")

from insurance_monitoring.sequential import SequentialTest, SequentialTestResult, sequential_test_from_df
print("SequentialTest, SequentialTestResult, sequential_test_from_df all importable")

# COMMAND ----------
# Run only the sequential tests
env = {
    **os.environ,
    "PYTHONPATH": f"{workspace_path}/src",
    "PYTHONDONTWRITEBYTECODE": "1",
}

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        f"{workspace_path}/tests/test_sequential.py",
        "-v", "--tb=long",
        "--import-mode=importlib",
        "-p", "no:cacheprovider",
        f"--rootdir={workspace_path}",
    ],
    capture_output=True,
    text=True,
    env=env,
)
output = result.stdout + "\\n" + result.stderr
print(output)
dbutils.notebook.exit(output[-6000:] if len(output) > 6000 else output)
'''

# Write the notebook to a temp file
notebook_local = PROJECT_ROOT / "notebooks" / "run_tests_sequential.py"
notebook_local.write_text(NOTEBOOK_CONTENT)
print(f"Wrote notebook to {notebook_local}")

# Collect all .py files from src/ and tests/
upload_files = []
for rel_dir in ["src", "tests"]:
    for p in (PROJECT_ROOT / rel_dir).rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        upload_files.append(p)

# Also upload pyproject.toml
upload_files.append(PROJECT_ROOT / "pyproject.toml")

print(f"Uploading {len(upload_files)} files to Databricks workspace...")
for local in sorted(upload_files):
    rel = local.relative_to(PROJECT_ROOT)
    remote = f"{WORKSPACE_PATH}/{rel}"
    upload_file(local, remote)

# Upload the notebook
upload_notebook(notebook_local, f"{WORKSPACE_PATH}/notebooks/run_tests_sequential")

print("\nSubmitting test job (serverless compute)...")

run = w.jobs.submit(
    run_name="insurance-monitoring-sequential-tests",
    tasks=[
        jobs.SubmitTask(
            task_key="run_sequential_tests",
            environment_key="default",
            notebook_task=jobs.NotebookTask(
                notebook_path=f"{WORKSPACE_PATH}/notebooks/run_tests_sequential",
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

print("\nPolling for completion...")
while True:
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
            print("\n--- Notebook output ---")
            print(output.notebook_output.result)
        if output.error:
            print(f"\nError: {output.error}")
        if output.error_trace:
            print(f"\nTrace:\n{output.error_trace}")
except Exception as e:
    print(f"Could not fetch output: {e}")

if result_state and result_state.value == "SUCCESS":
    print("\nAll sequential tests PASSED.")
    sys.exit(0)
else:
    print("\nTests FAILED.")
    sys.exit(1)
