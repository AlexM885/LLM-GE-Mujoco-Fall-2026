"""Shared fixtures for the MESB tests.

These tests need only NumPy (plus DEAP for the shadow/full-mode integration
tests, which skip themselves if it is missing). No MuJoCo, PPO or LLM.
"""

import os
import sys
from pathlib import Path

import pytest

# tests/conftest.py (shared, not modified here) starts an LLM server at session
# start - on a Slurm cluster via `sbatch server.sh` (2 GPUs). The MESB tests use
# no LLM, so when they are run on their own (`pytest tests/mesb`) default that
# off. An explicit LLMGE_AUTO_START_SERVER in the environment still wins.
os.environ.setdefault("LLMGE_AUTO_START_SERVER", "0")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sota.MujocoRL.mesb_archive import ArchiveRecord  # noqa: E402


def make_record(gene_id, objective, measures, generation=0):
    return ArchiveRecord(gene_id=gene_id, generation=generation,
                         objective=objective, measures=tuple(measures))


@pytest.fixture
def rec():
    return make_record
