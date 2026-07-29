#!/usr/bin/env python3
"""Project-level exact commutator tests for the Hamiltonian family results.

Run from this folder:

    python3 commutator_tests.py

The script uses the dependency-free exact Clifford algebra package in
../HamFam_Resources/python. It verifies:

1. The stored corrected N=3 table.
2. The stored corrected N=4 table.
3. The stored structured N=5 table.
4. The arbitrary-N closed formula projected to N=3, N=4, and N=5.
"""

from __future__ import annotations

import sys
from pathlib import Path
from math import comb


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "HamFam_Resources" / "python"
sys.path.insert(0, str(PYTHON_ROOT))

from hamfam.clifford import CliffordExpr, minus, sum_expr  # noqa: E402
from hamfam.models import (  # noqa: E402
    a_bilinear,
    v_bilinear,
    beta_product,
    beta_square_product,
    d0_op,
    d_op,
    failure_counts,
    gvar,
    integrability_failures,
    inv_beta,
    majorana_algebra,
    n3_majorana_table,
    n4_majorana_table,
    n5_majorana_table,
    term,
)


PARAMETER_NAMES = {
    3: ("t", "eps", "eta"),
    4: ("t", "eps", "eta", "zeta"),
    5: ("t", "eps", "eta", "zeta", "xi"),
}


def _signed(expr: CliffordExpr, sign: int) -> CliffordExpr:
    return expr if sign > 0 else minus(expr)


def _s_rs(alg, n_sites: int, r_degree: int, square_degree: int) -> CliffordExpr:
    import itertools

    if r_degree < 0 or square_degree < 0 or r_degree > n_sites or square_degree > n_sites - r_degree:
        return alg.zero()

    parts = []
    for subset in itertools.combinations(range(1, n_sites + 1), r_degree):
        complement = [index for index in range(1, n_sites + 1) if index not in subset]
        square_subsets = [()] if square_degree == 0 else itertools.combinations(complement, square_degree)
        for square_subset in square_subsets:
            parts.append(
                term(
                    beta_product(alg, subset) * beta_square_product(alg, square_subset),
                    *(a_bilinear(alg, n_sites, index) for index in subset),
                    alg=alg,
                )
            )
    return sum_expr(parts, alg) if parts else alg.zero()


def _d0_e(alg, n_sites: int, degree: int) -> CliffordExpr:
    return term(alg.const_poly(1), d0_op(alg), _s_rs(alg, n_sites, degree, 0), alg=alg)


def _constant_dressing(alg, n_sites: int, degree: int, sign: int) -> CliffordExpr:
    return _signed(
        sum_expr(
            (
                term(
                    gvar(alg, j),
                    d_op(alg, j),
                    _complement_e(alg, n_sites, j, degree),
                    alg=alg,
                )
                for j in range(1, n_sites + 1)
            ),
            alg,
        ),
        sign,
    )


def _complement_e(alg, n_sites: int, excluded: int, degree: int) -> CliffordExpr:
    import itertools

    complement = [index for index in range(1, n_sites + 1) if index != excluded]
    if degree < 0 or degree > len(complement):
        return alg.zero()
    parts = [
        term(
            beta_product(alg, subset),
            *(a_bilinear(alg, n_sites, index) for index in subset),
            alg=alg,
        )
        for subset in itertools.combinations(complement, degree)
    ]
    return sum_expr(parts, alg) if parts else alg.zero()


def arbitrary_formula_table(n_sites: int) -> tuple[object, list[list[CliffordExpr]]]:
    """Build the conjectural closed-form table for N=3,4,5 projections."""

    tau_names = PARAMETER_NAMES.get(n_sites, tuple(f"tau{index}" for index in range(1, n_sites + 1)))
    alg = majorana_algebra(n_sites, extra_variables=tau_names)
    matrix = [[alg.zero() for _ in range(n_sites + 1)] for _ in range(n_sites)]
    constant_col = n_sites

    matrix[0][0] = _s_rs(alg, n_sites, 1, 0)
    matrix[0][1] = d0_op(alg)
    matrix[1][0] = matrix[0][1]
    matrix[1][1] = sum_expr(
        (term(inv_beta(alg, j), a_bilinear(alg, n_sites, j), alg=alg) for j in range(1, n_sites + 1)),
        alg,
    )

    for nu in range(3, n_sites + 1):
        matrix[0][nu - 1] = _signed(_s_rs(alg, n_sites, nu - 1, 0), -1 if nu % 2 else 1)
        matrix[nu - 1][0] = matrix[0][nu - 1]

        matrix[1][nu - 1] = _signed(_d0_e(alg, n_sites, nu - 2), -1 if nu % 2 else 1)
        matrix[nu - 1][1] = matrix[1][nu - 1]

    for mu in range(3, n_sites + 1):
        for nu in range(mu, n_sites + 1):
            h = mu - 2
            k = nu - 2
            expr = alg.zero()
            for square_degree in range(0, min(h, k) + 1):
                coeff = comb(h + k - 2 * square_degree, h - square_degree)
                expr = expr + _s_rs(alg, n_sites, mu + nu - 3 - 2 * square_degree, square_degree).scale(coeff)
            expr = _signed(expr, 1 if (mu + nu) % 2 == 0 else -1)
            matrix[mu - 1][nu - 1] = expr
            matrix[nu - 1][mu - 1] = expr

    matrix[0][constant_col] = sum_expr(
        (term(gvar(alg, j), d_op(alg, j), alg=alg) for j in range(1, n_sites + 1)),
        alg,
    )
    matrix[1][constant_col] = minus(
        sum_expr(
            (term(gvar(alg, j) * inv_beta(alg, j), v_bilinear(alg, n_sites, j), alg=alg) for j in range(1, n_sites + 1)),
            alg,
        )
    )
    for mu in range(3, n_sites + 1):
        matrix[mu - 1][constant_col] = _constant_dressing(
            alg,
            n_sites,
            mu - 2,
            1 if mu % 2 == 0 else -1,
        )

    return alg, matrix


def _assert_no_failures(label: str, matrix: list[list[CliffordExpr]], tau_names: tuple[str, ...]) -> None:
    failures = integrability_failures(matrix, tau_names)
    if failures:
        print(f"{label}: FAIL {failure_counts(failures)}")
        for failure in failures[:8]:
            print(f"  {failure.name} {failure.indices} terms={len(failure.expr.terms)}")
        raise SystemExit(1)
    print(f"{label}: PASS")


def _assert_same_table(label: str, expected: list[list[CliffordExpr]], actual: list[list[CliffordExpr]]) -> None:
    for row_index, (expected_row, actual_row) in enumerate(zip(expected, actual), start=1):
        for col_index, (expected_expr, actual_expr) in enumerate(zip(expected_row, actual_row), start=1):
            if not (expected_expr - actual_expr).is_zero:
                print(f"{label}: FAIL at entry ({row_index},{col_index})")
                print((expected_expr - actual_expr).format_terms(limit=12))
                raise SystemExit(1)
    print(f"{label}: PASS")


def main() -> int:
    verified_builders = {
        3: n3_majorana_table,
        4: n4_majorana_table,
        5: n5_majorana_table,
    }

    for n_sites, builder in verified_builders.items():
        _, table = builder()
        _assert_no_failures(f"N={n_sites} verified table", table, PARAMETER_NAMES[n_sites])

    for n_sites, builder in verified_builders.items():
        _, expected = builder()
        _, projected = arbitrary_formula_table(n_sites)
        _assert_same_table(f"general formula projection N={n_sites}", expected, projected)
        _assert_no_failures(f"general formula commutators N={n_sites}", projected, PARAMETER_NAMES[n_sites])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
