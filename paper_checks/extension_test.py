#!/usr/bin/env python3.12
"""Test whether the closed-form family extends to more flows than N (sites).

Builds an (n_flows)-row table from the arbitrary-N closed formula
(Table 6 / Eq. (A.16) of the draft) on n_sites Majorana sites, then runs the
full exact integrability check. n_flows = n_sites reproduces the verified
family; n_flows = n_sites + 1 is the candidate (N+1)-st Hamiltonian.
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HamFam_Resources" / "python"))

from hamfam.clifford import minus, sum_expr  # noqa: E402
from hamfam.models import (  # noqa: E402
    a_bilinear,
    v_bilinear,
    constant_dressing,
    d0_op,
    d_op,
    d0_elementary_a,
    elementary_a,
    failure_counts,
    gvar,
    integrability_failures,
    inv_beta,
    majorana_algebra,
    square_weighted_a,
    term,
)


def signed(expr, sign):
    return expr if sign > 0 else minus(expr)


def a_index(mu: int) -> int:
    # a_1 = 0, a_mu = mu - 2 for mu >= 3 (mu = 2 is the D0 row, special)
    return 0 if mu == 1 else mu - 2


def k_block(alg, n_sites: int, mu: int, nu: int):
    """K^{(N)}_{a_mu, a_nu} with sign (-1)^{a_mu + a_nu}: the Hessian block."""
    a, b = a_index(mu), a_index(nu)
    expr = alg.zero()
    for s in range(0, min(a, b) + 1):
        coeff = comb(a + b - 2 * s, a - s)
        if coeff == 0:
            continue
        expr = expr + square_weighted_a(alg, n_sites, a + b + 1 - 2 * s, s).scale(coeff)
    return signed(expr, 1 if (a + b) % 2 == 0 else -1)


def build_table(n_sites: int, n_flows: int):
    tau_names = tuple(f"tau{i}" for i in range(1, n_flows + 1))
    alg = majorana_algebra(n_sites, extra_variables=tau_names)
    matrix = [[alg.zero() for _ in range(n_flows + 1)] for _ in range(n_flows)]
    const_col = n_flows

    rows = [1] + list(range(3, n_flows + 1))  # the script-I index set (skip 2)

    # Hessian block for mu, nu in I
    for mu in rows:
        for nu in rows:
            if nu < mu:
                continue
            entry = k_block(alg, n_sites, mu, nu)
            matrix[mu - 1][nu - 1] = entry
            matrix[nu - 1][mu - 1] = entry

    # Row/column 2 (D0 direction)
    for mu in rows:
        a = a_index(mu)
        entry = signed(d0_elementary_a(alg, n_sites, a), 1 if a % 2 == 0 else -1)
        matrix[1][mu - 1] = entry
        matrix[mu - 1][1] = entry
    matrix[1][1] = sum_expr(
        (term(inv_beta(alg, j), a_bilinear(alg, n_sites, j), alg=alg) for j in range(1, n_sites + 1)),
        alg,
    )

    # Constant column
    for mu in rows:
        a = a_index(mu)
        matrix[mu - 1][const_col] = constant_dressing(
            alg, n_sites, a, 1 if a % 2 == 0 else -1
        )
    matrix[1][const_col] = minus(
        sum_expr(
            (
                term(gvar(alg, j) * inv_beta(alg, j), v_bilinear(alg, n_sites, j), alg=alg)
                for j in range(1, n_sites + 1)
            ),
            alg,
        )
    )
    return alg, matrix, tau_names


def entry_summary(matrix):
    lines = []
    for i, row in enumerate(matrix, start=1):
        cells = []
        for j, e in enumerate(row):
            label = "0" if j == len(row) - 1 else str(j + 1)
            cells.append(f"O[{i}][{label}]:{'zero' if e.is_zero else len(e.terms)}")
        lines.append("  " + "  ".join(cells))
    return "\n".join(lines)


def run(n_sites: int, n_flows: int, show_entries=False, show_failures=8):
    alg, matrix, tau_names = build_table(n_sites, n_flows)
    failures = integrability_failures(matrix, tau_names)
    tag = f"N={n_sites} sites, {n_flows} flows"
    if failures:
        print(f"{tag}: FAIL {failure_counts(failures)}")
        for f in failures[:show_failures]:
            print(f"  {f.name} {f.indices} nonzero-terms={len(f.expr.terms)}")
            print("    " + f.expr.format_terms(limit=4).replace("\n", "\n    "))
    else:
        print(f"{tag}: PASS (all conditions hold exactly)")
    if show_entries:
        print(entry_summary(matrix))
    return failures


if __name__ == "__main__":
    # Controls: must PASS (reproduce verified families)
    run(2, 2)
    run(3, 3)
    # The candidate (N+1)-st flow
    run(2, 3, show_entries=True)
    run(3, 4, show_entries=True)
    run(4, 5)
    # Beyond: N+2 flows
    run(2, 4, show_entries=True)
    run(3, 5, show_entries=True)
