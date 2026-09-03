"""Pytest conftest: put the repo root on sys.path so tests can import the
repo's flat modules (source_registry, job_fit_finder, feedback_server)
regardless of the directory pytest is invoked from."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
