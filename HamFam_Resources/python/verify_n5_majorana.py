#!/usr/bin/env python3
"""Exact check for the solved N=5 Majorana Hamiltonian family."""

from __future__ import annotations

from hamfam.models import failure_counts, integrability_failures, n5_majorana_table


def main() -> int:
    _, matrix = n5_majorana_table()
    failures = integrability_failures(matrix, ("t", "eps", "eta", "zeta", "xi"))

    print("N=5 Majorana structured ansatz check")
    counts = failure_counts(failures)
    for name in ("symmetry", "eq13", "eq14", "eq00", "eq10", "full"):
        print(f"{name:8s}: {counts.get(name, 0)}")

    if failures:
        print("FAIL: nonzero affine integrability constraints remain.")
        for failure in failures[:10]:
            print(f"  {failure.name} {failure.indices} terms={len(failure.expr.terms)}")
        return 1

    print("PASS: all affine integrability constraints vanish.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
