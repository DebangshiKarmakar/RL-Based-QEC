"""Tests for run_baseline CLI script."""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_baseline.py"


def test_cli_zero_failures_does_not_crash() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--shots", "10", "--p", "0.00001", "--seed", "1"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0
    assert "MWPM failures: 0" in result.stdout
    assert "Block LER: 0.000000" in result.stdout
    assert "Shots for about 100 MWPM failures" in result.stdout


def test_cli_json_output() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--shots", "100", "--p", "0.005", "--seed", "2", "--json"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "mwpm_failures" in data
    assert "block_ler" in data
    assert "wilson_ci_95" in data
    assert "per_round_ler" in data
    assert "config" in data
    assert data["config"]["distance"] == 3

