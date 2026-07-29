#!/usr/bin/env python3
"""Solve the N=5 structured ansatz suggested by the N=3 and N=4 tables."""

from __future__ import annotations

from hamfam.clifford import CliffordExpr, minus, sum_expr
from hamfam.exact import ComplexRat, ZERO_C, cneg, fmt_complex
from hamfam.linear import solve_linear
from hamfam.models import (
    a_bilinear,
    v_bilinear,
    beta,
    constant_dressing,
    d0_elementary_a,
    d0_op,
    d_op,
    elementary_a,
    failure_counts,
    gvar,
    integrability_failures,
    inv_beta,
    majorana_algebra,
    square_weighted_a,
    term,
)


def _coefficients(expr: CliffordExpr) -> dict[tuple[int, tuple[int, ...]], ComplexRat]:
    out = {}
    for mask, poly in expr.terms.items():
        for monomial, coeff in poly.terms.items():
            out[(mask, monomial)] = coeff
    return out


def _n5_base():
    n_sites = 5
    alg = majorana_algebra(n_sites, extra_variables=("t", "eps", "eta", "zeta", "xi"))
    matrix = [[alg.zero() for _ in range(6)] for _ in range(5)]

    matrix[0][0] = elementary_a(alg, n_sites, 1)
    matrix[0][1] = d0_op(alg)
    matrix[0][2] = minus(elementary_a(alg, n_sites, 2))
    matrix[0][3] = elementary_a(alg, n_sites, 3)
    matrix[0][4] = minus(elementary_a(alg, n_sites, 4))
    matrix[0][5] = sum_expr(
        (term(gvar(alg, j), d_op(alg, j), alg=alg) for j in range(1, n_sites + 1)),
        alg,
    )

    matrix[1][0] = matrix[0][1]
    matrix[1][1] = sum_expr(
        (term(inv_beta(alg, j), a_bilinear(alg, n_sites, j), alg=alg) for j in range(1, n_sites + 1)),
        alg,
    )
    matrix[1][2] = minus(d0_elementary_a(alg, n_sites, 1))
    matrix[1][3] = d0_elementary_a(alg, n_sites, 2)
    matrix[1][4] = minus(d0_elementary_a(alg, n_sites, 3))
    matrix[1][5] = minus(
        sum_expr(
            (
                term(gvar(alg, j) * inv_beta(alg, j), v_bilinear(alg, n_sites, j), alg=alg)
                for j in range(1, n_sites + 1)
            ),
            alg,
        )
    )

    matrix[2][0] = matrix[0][2]
    matrix[2][1] = matrix[1][2]
    matrix[2][2] = elementary_a(alg, n_sites, 3).scale(2) + square_weighted_a(alg, n_sites, 1, 1)
    matrix[2][3] = minus(elementary_a(alg, n_sites, 4).scale(3) + square_weighted_a(alg, n_sites, 2, 1))
    matrix[2][5] = constant_dressing(alg, n_sites, 1, -1)

    matrix[3][0] = matrix[0][3]
    matrix[3][1] = matrix[1][3]
    matrix[3][2] = matrix[2][3]
    matrix[3][5] = constant_dressing(alg, n_sites, 2, 1)

    matrix[4][0] = matrix[0][4]
    matrix[4][1] = matrix[1][4]
    matrix[4][5] = constant_dressing(alg, n_sites, 3, -1)

    return alg, matrix


def _basis(alg):
    return [
        ((3, 3), "B44:S(1,2)", square_weighted_a(alg, 5, 1, 2)),
        ((3, 3), "B44:S(3,1)", square_weighted_a(alg, 5, 3, 1)),
        ((3, 3), "B44:S(5,0)", square_weighted_a(alg, 5, 5, 0)),
        ((2, 4), "B35:S(1,2)", square_weighted_a(alg, 5, 1, 2)),
        ((2, 4), "B35:S(3,1)", square_weighted_a(alg, 5, 3, 1)),
        ((2, 4), "B35:S(5,0)", square_weighted_a(alg, 5, 5, 0)),
        ((3, 4), "B45:S(2,2)", square_weighted_a(alg, 5, 2, 2)),
        ((3, 4), "B45:S(4,1)", square_weighted_a(alg, 5, 4, 1)),
        ((4, 4), "B55:S(1,3)", square_weighted_a(alg, 5, 1, 3)),
        ((4, 4), "B55:S(3,2)", square_weighted_a(alg, 5, 3, 2)),
    ]


def _with_basis(matrix, target, expr):
    out = [row[:] for row in matrix]
    row, col = target
    out[row][col] = out[row][col] + expr
    if row != col and col < len(matrix):
        out[col][row] = out[row][col]
    return out


def _linear_system(matrix, basis):
    rows: list[list[ComplexRat]] = []
    rhs: list[ComplexRat] = []
    n_params = 5

    builders = []
    for mu in range(n_params):
        for nu in range(mu + 1, n_params):
            builders.append(lambda m, mu=mu, nu=nu: m[mu][n_params].commutator(m[nu][n_params]))
            for alpha in range(n_params):
                builders.append(
                    lambda m, mu=mu, nu=nu, alpha=alpha: m[mu][alpha].commutator(m[nu][n_params])
                    + m[mu][n_params].commutator(m[nu][alpha])
                )

    for builder in builders:
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
    alg, matrix = _n5_base()
    basis = _basis(alg)
    rows, rhs = _linear_system(matrix, basis)
    result = solve_linear(rows, rhs, len(basis))

    print("N=5 structured ansatz solve")
    print(f"unknowns     : {len(basis)}")
    print(f"equations    : {len(rows)}")
    print(f"rank         : {result.rank}")
    print(f"inconsistent : {len(result.inconsistent_rows)}")
    if result.inconsistent_rows:
        print("No exact solution in the structured N=5 basis.")
        return 1

    solved = [row[:] for row in matrix]
    print("solution:")
    for (target, name, expr), value in zip(basis, result.solution):
        print(f"  {name} = {fmt_complex(value)}")
        solved = _with_basis(solved, target, expr.scale(value))

    failures = integrability_failures(solved, ("t", "eps", "eta", "zeta", "xi"))
    print("full affine check:", failure_counts(failures))
    if failures:
        for failure in failures[:8]:
            print(f"  {failure.name} {failure.indices} terms={len(failure.expr.terms)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
