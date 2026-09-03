"""Key-less import: the engine imports cleanly with OPENAI_API_KEY unset
and the OpenAI client is not constructed at import time."""
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_engine_imports_without_openai_key():
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    # Hide the developer's real .env so the subprocess is genuinely key-less.
    env["JOB_FIT_ENV_PATH"] = os.path.join(REPO_ROOT, ".no.env.for.tests")
    proc = subprocess.run(
        [sys.executable, "-c",
         "import source_registry, job_fit_finder, feedback_server; "
         "import job_fit_finder as jf; "
         "print('client:', jf._client)"],
        env=env, capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert "client: None" in proc.stdout  # lazy: no OpenAI client at import


def test_lazy_client_raises_clear_error_on_first_use():
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    env["JOB_FIT_ENV_PATH"] = os.path.join(REPO_ROOT, ".no.env.for.tests")
    proc = subprocess.run(
        [sys.executable, "-c", "import job_fit_finder as jf; jf._get_openai_client()"],
        env=env, capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode != 0
    assert "OPENAI_API_KEY is not set" in proc.stderr
