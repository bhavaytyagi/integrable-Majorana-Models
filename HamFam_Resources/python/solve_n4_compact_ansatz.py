#!/usr/bin/env python3
"""Check the corrected N=4 base family and test the compact B34/B44 ansatz."""

from __future__ import annotations

from hamfam.clifford import CliffordExpr
from hamfam.exact import ComplexRat, ZERO_C, cneg, fmt_complex
from hamfam.linear import solve_linear
from hamfam.models import (
    elementary_a,
    failure_counts,
    integrability_failures,
    n4_compact_base_table,
    n4_three_parameter_table,
    weighted_a2,
    weighted_a3,
)


def _coefficients(expr: CliffordExpr) -> dict[tuple[int, tuple[int, ...]], ComplexRat]:
    out = {}
    for mask, poly in expr.terms.items():
        for monomial, coeff in poly.terms.items():
            out[(mask, monomial)] = coeff
    return out


def _with_basis(matrix, target, expr):
    out = [row[:] for row in matrix]
    row, col = target
    out[row][col] = out[row][col] + expr
    if row != col and col < len(matrix):
        out[col][row] = out[row][col]
    return out


def _targeted_builders():
    builders = []
    for mu in (0, 1, 2):
        label = f"eta-sector mu={mu + 1}"

        def eta_builder(matrix, mu=mu):
            return matrix[mu][4].commutator(matrix[2][3]) + matrix[mu][2].commutator(matrix[3][4])

        builders.append((label, eta_builder))

    for mu in (0, 1, 2):
        label = f"zeta-sector mu={mu + 1}"

        def zeta_builder(matrix, mu=mu):
            return matrix[mu][4].commutator(matrix[3][3]) + matrix[mu][3].commutator(matrix[3][4])

        builders.append((label, zeta_builder))

    return builders


def _linear_system(matrix, basis, builders):
    rows: list[list[ComplexRat]] = []
    rhs: list[ComplexRat] = []

    for _, builder in builders:
        const_expr = builder(matrix)
        const_coeffs = _coefficients(const_expr)
        column_coeffs = [
            _coefficients(builder(_with_basis(matrix, target, expr)) - const_expr)
            for target, _, expr in basis
        ]
        keys = set(const_coeffs)
        for coeffs in column_coeffs:
            keys.update(coeffs)
        for key in sorted(keys):
            row = [coeffs.get(key, ZERO_C) for coeffs in column_coeffs]
            value = cneg(const_coeffs.get(key, ZERO_C))
            if any(coeff != ZERO_C for coeff in row) or value != ZERO_C:
                rows.append(row)
                rhs.append(value)
    return rows, rhs


def main() -> int:
    _, subfamily = n4_three_parameter_table()
    sub_failures = integrability_failures(subfamily, ("t", "eps", "eta"))
    print("N=4 restriction to (t, eps, eta)")
    print(failure_counts(sub_failures))
    if sub_failures:
        for failure in sub_failures[:6]:
            print(f"{failure.name} {failure.indices} terms={len(failure.expr.terms)}")
        return 1

    alg, matrix = n4_compact_base_table()
    basis = [
        ((2, 3), "B34:e4", elementary_a(alg, 4, 4)),
        ((2, 3), "B34:weighted_a2", weighted_a2(alg, 4)),
        ((3, 3), "B44:e3", elementary_a(alg, 4, 3)),
        ((3, 3), "B44:weighted_a3", weighted_a3(alg, 4)),
    ]

    rows, rhs = _linear_system(matrix, basis, _targeted_builders())
    result = solve_linear(rows, rhs, len(basis))
    print("\nCompact N=4 B34/B44 ansatz")
    print(f"unknowns     : {len(basis)}")
    print(f"equations    : {len(rows)}")
    print(f"rank         : {result.rank}")
    print(f"inconsistent : {len(result.inconsistent_rows)}")
    if result.inconsistent_rows:
        print("No exact solution in the original four-coefficient compact ansatz.")
        print("The exact fourth-flow solve forces sign corrections and an extra beta-degree-5 B44 structure.")
        print("See solve_n4_fourth_flow.py and verify_n4_majorana.py.")
        return 1

    print("solution:")
    for (_, name, _), value in zip(basis, result.solution):
        print(f"  {name} = {fmt_complex(value)}")

    solved = [row[:] for row in matrix]
    for (target, _, expr), value in zip(basis, result.solution):
        solved = _with_basis(solved, target, expr.scale(value))
    failures = integrability_failures(solved, ("t", "eps", "eta", "zeta"))
    print("full affine check:", failure_counts(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
