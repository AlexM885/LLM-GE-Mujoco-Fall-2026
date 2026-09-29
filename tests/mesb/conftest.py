"""Shared fixtures for the MESB tests.

These tests need only NumPy (plus DEAP for the shadow/full-mode integration
tests, which skip themselves if it is missing). No MuJoCo, PPO or LLM.
"""

import sys
from pathlib import Path

import pytest

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
