#!/usr/bin/env python3
"""Rank/nullity checks for structured uniqueness evidence.

This script runs the exact Clifford-algebra linear solves that were used to
derive the finite N=3, N=4, and N=5 tables.  The uniqueness evidence is
full-rank solvability in the specified structured ansatz spaces.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "HamFam_Resources" / "python"


@dataclass(frozen=True)
class CheckSpec:
    label: str
    script: Path
    block_title: str
    expected_unknowns: int
    expected_rank: int


CHECKS = (
    CheckSpec(
        label="N=3 structured uniqueness",
        script=PYTHON_ROOT / "solve_n3_constant_terms.py",
        block_title="Joint solve: PDF B33 polynomial terms plus O_3 dressed terms",
        expected_unknowns=13,
        expected_rank=13,
    ),
    CheckSpec(
        label="N=4 fourth-flow uniqueness",
        script=PYTHON_ROOT / "solve_n4_fourth_flow.py",
        block_title="N=4 fourth-flow solve with forced sign corrections",
        expected_unknowns=448,
        expected_rank=448,
    ),
    CheckSpec(
        label="N=5 structured uniqueness",
        script=PYTHON_ROOT / "solve_n5_structured_ansatz.py",
        block_title="N=5 structured ansatz solve",
        expected_unknowns=10,
        expected_rank=10,
    ),
)


def _run_script(path: Path) -> str:
    env = os.environ.copy()
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PYTHON_ROOT) if not previous else f"{PYTHON_ROOT}{os.pathsep}{previous}"
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=Path(__file__).resolve().parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        print(result.stdout)
        raise SystemExit(result.returncode)
    return result.stdout


def _parse_block(output: str, title: str) -> tuple[int, int, int]:
    start = output.find(title)
    if start < 0:
        raise ValueError(f"could not find block title: {title}")
    block = output[start:]
    unknowns = _parse_int(block, r"unknowns\s*:\s*(\d+)")
    rank = _parse_int(block, r"rank\s*:\s*(\d+)")
    inconsistent = _parse_int(block, r"inconsistent\s*:\s*(\d+)")
    return unknowns, rank, inconsistent


def _parse_int(text: str, pattern: str) -> int:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"could not parse pattern: {pattern}")
    return int(match.group(1))


def main() -> int:
    for spec in CHECKS:
        output = _run_script(spec.script)
        unknowns, rank, inconsistent = _parse_block(output, spec.block_title)
        nullity = unknowns - rank
        ok = (
            unknowns == spec.expected_unknowns
            and rank == spec.expected_rank
            and inconsistent == 0
            and nullity == 0
        )
        status = "PASS" if ok else "FAIL"
        print(
            f"{spec.label}: {status} "
            f"rank={rank} unknowns={unknowns} nullity={nullity} inconsistent={inconsistent}"
        )
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
