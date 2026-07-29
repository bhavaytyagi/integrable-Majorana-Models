#!/usr/bin/env python3
"""Verify the N=3 even-Majorana table from the conjecture note.

This intentionally checks the full affine family, not just the Hessian block.
The source of truth is the PDF transcription:
B33 = 2 beta1 beta2 beta3 A1 A2 A3 + T,
O30 = -sum_j g_j D_j sum_{k != j} beta_k A_k.
"""

from __future__ import annotations

from hamfam.models import failure_counts, integrability_failures, n3_majorana_table


def main() -> int:
    _, matrix = n3_majorana_table()
    failures = integrability_failures(matrix, ("t", "eps", "eta"))
    counts = failure_counts(failures)

    print("N=3 Majorana transcription check")
    for name in ("symmetry", "eq13", "eq14", "eq00", "eq10", "full"):
        print(f"{name:8s}: {counts.get(name, 0)}")

    if not failures:
        print("PASS: all affine integrability constraints vanish.")
        return 0

    print("\nFirst nonzero failures:")
    for failure in failures[:8]:
        print(f"\n[{failure.name}] indices={failure.indices}, terms={len(failure.expr.terms)}")
        print(failure.expr.format_terms(limit=12))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
