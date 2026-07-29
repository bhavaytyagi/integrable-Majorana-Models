#!/usr/bin/env python3
"""Solve the N=3 constant-column coefficients from the exact constraints.

Two exact linear systems are solved against the PDF ansatz:

1. Keep the corrected B33 fixed and solve
   O_3 = sum_{j != k} c[j,k] g_j beta_k D_j A_k.

2. Solve B33 and O_3 together, using the coefficient basis that appears in the
   PDF. The solution should recover B33's coefficients and c[j,k] = -1.
"""

from __future__ import annotations

from dataclasses import dataclass

from hamfam.clifford import CliffordExpr, minus, sum_expr
from hamfam.exact import ComplexRat, ZERO_C, cneg, fmt_complex
from hamfam.linear import solve_linear
from hamfam.models import (
    a_bilinear,
    beta,
    v_bilinear,
    d0_op,
    d_op,
    gvar,
    inv_beta,
    majorana_algebra,
    q_op,
    term,
    top_op,
)


@dataclass(frozen=True)
class BasisTerm:
    target: tuple[int, int]
    name: str
    expr: CliffordExpr


def _base_n3_without_o33_o30():
    n_sites = 3
    alg = majorana_algebra(n_sites, extra_variables=("t", "eps", "eta"))
    matrix = [[alg.zero() for _ in range(4)] for _ in range(3)]

    matrix[0][0] = sum_expr((term(beta(alg, j), a_bilinear(alg, n_sites, j), alg=alg) for j in range(1, 4)), alg)
    matrix[0][1] = d0_op(alg)
    matrix[0][2] = minus(q_op(alg, n_sites))
    matrix[0][3] = sum_expr((term(gvar(alg, j), d_op(alg, j), alg=alg) for j in range(1, 4)), alg)

    matrix[1][0] = matrix[0][1]
    matrix[1][1] = sum_expr((term(inv_beta(alg, j), a_bilinear(alg, n_sites, j), alg=alg) for j in range(1, 4)), alg)
    matrix[1][2] = minus(
        sum_expr((term(beta(alg, j), d0_op(alg), a_bilinear(alg, n_sites, j), alg=alg) for j in range(1, 4)), alg)
    )
    matrix[1][3] = minus(
        sum_expr((term(gvar(alg, j) * inv_beta(alg, j), v_bilinear(alg, n_sites, j), alg=alg) for j in range(1, 4)), alg)
    )

    matrix[2][0] = matrix[0][2]
    matrix[2][1] = matrix[1][2]
    return alg, matrix


def _current_o33(alg) -> CliffordExpr:
    return (
        term(
            alg.const_poly(2) * beta(alg, 1) * beta(alg, 2) * beta(alg, 3),
            a_bilinear(alg, 3, 1),
            a_bilinear(alg, 3, 2),
            a_bilinear(alg, 3, 3),
            alg=alg,
        )
        + top_op(alg, 3)
    )


def _o30_basis(alg) -> list[BasisTerm]:
    basis = []
    for j in range(1, 4):
        for k in range(1, 4):
            if k == j:
                continue
            basis.append(
                BasisTerm(
                    target=(2, 3),
                    name=f"O_3:g{j}*beta{k}*D{j}*A{k}",
                    expr=term(gvar(alg, j) * beta(alg, k), d_op(alg, j), a_bilinear(alg, 3, k), alg=alg),
                )
            )
    return basis


def _wide_o33_basis(alg) -> list[BasisTerm]:
    basis = [
        BasisTerm(
            target=(2, 2),
            name="B33:beta1*beta2*beta3*A1*A2*A3",
            expr=term(
                beta(alg, 1) * beta(alg, 2) * beta(alg, 3),
                a_bilinear(alg, 3, 1),
                a_bilinear(alg, 3, 2),
                a_bilinear(alg, 3, 3),
                alg=alg,
            ),
        )
    ]
    for j in range(1, 4):
        for k in range(1, 4):
            if k == j:
                continue
            basis.append(
                BasisTerm(
                    target=(2, 2),
                    name=f"B33:beta{j}*beta{k}^2*A{j}",
                    expr=term(beta(alg, j) * beta(alg, k, 2), a_bilinear(alg, 3, j), alg=alg),
                )
            )
    return basis


def _with_basis(matrix, basis_term: BasisTerm | None) -> list[list[CliffordExpr]]:
    out = [row[:] for row in matrix]
    if basis_term is not None:
        row, col = basis_term.target
        out[row][col] = out[row][col] + basis_term.expr
    return out


def _coefficients(expr: CliffordExpr) -> dict[tuple[int, tuple[int, ...]], ComplexRat]:
    out = {}
    for mask, poly in expr.terms.items():
        for monomial, coeff in poly.terms.items():
            out[(mask, monomial)] = coeff
    return out


def _linear_system(matrix, basis: list[BasisTerm], constraint_builders):
    rows: list[list[ComplexRat]] = []
    rhs: list[ComplexRat] = []
    row_labels = []

    for label, builder in constraint_builders:
        const_expr = builder(matrix)
        col_exprs = [builder(_with_basis(matrix, item)) - const_expr for item in basis]
        keys = set(_coefficients(const_expr))
        for col_expr in col_exprs:
            keys.update(_coefficients(col_expr))
        for key in sorted(keys):
            row = [_coefficients(col_expr).get(key, ZERO_C) for col_expr in col_exprs]
            value = cneg(_coefficients(const_expr).get(key, ZERO_C))
            if any(coeff != ZERO_C for coeff in row) or value != ZERO_C:
                rows.append(row)
                rhs.append(value)
                row_labels.append((label, key))
    return rows, rhs, row_labels


def _eq10_builders():
    builders = []
    for mu, nu in ((0, 2), (1, 2)):
        for alpha in range(3):
            label = f"eq10(mu={mu + 1},nu={nu + 1},alpha={alpha + 1})"

            def builder(matrix, mu=mu, nu=nu, alpha=alpha):
                return matrix[mu][alpha].commutator(matrix[nu][3]) + matrix[mu][3].commutator(matrix[nu][alpha])

            builders.append((label, builder))
    return builders


def _all_coefficient_builders():
    builders = []
    for mu in range(3):
        for nu in range(mu + 1, 3):
            for alpha in range(3):
                label = f"eq13(mu={mu + 1},nu={nu + 1},alpha={alpha + 1})"

                def eq13(matrix, mu=mu, nu=nu, alpha=alpha):
                    return matrix[mu][alpha].commutator(matrix[nu][alpha])

                builders.append((label, eq13))

            for alpha in range(3):
                for beta_col in range(alpha + 1, 3):
                    label = f"eq14(mu={mu + 1},nu={nu + 1},alpha={alpha + 1},beta={beta_col + 1})"

                    def eq14(matrix, mu=mu, nu=nu, alpha=alpha, beta_col=beta_col):
                        return matrix[mu][alpha].commutator(matrix[nu][beta_col]) + matrix[mu][beta_col].commutator(
                            matrix[nu][alpha]
                        )

                    builders.append((label, eq14))

            label = f"eq00(mu={mu + 1},nu={nu + 1})"

            def eq00(matrix, mu=mu, nu=nu):
                return matrix[mu][3].commutator(matrix[nu][3])

            builders.append((label, eq00))

            for alpha in range(3):
                label = f"eq10(mu={mu + 1},nu={nu + 1},alpha={alpha + 1})"

                def eq10(matrix, mu=mu, nu=nu, alpha=alpha):
                    return matrix[mu][alpha].commutator(matrix[nu][3]) + matrix[mu][3].commutator(matrix[nu][alpha])

                builders.append((label, eq10))

    return builders


def _print_result(title: str, basis: list[BasisTerm], rows, rhs, labels) -> None:
    result = solve_linear(rows, rhs, len(basis))
    print(f"\n{title}")
    print(f"unknowns     : {len(basis)}")
    print(f"equations    : {len(rows)}")
    print(f"rank         : {result.rank}")
    print(f"inconsistent : {len(result.inconsistent_rows)}")
    if result.inconsistent_rows:
        print("no exact solution in this ansatz")
        print("note         : inconsistent rows are RREF certificates, i.e. linear combinations of input equations")
        return
    print("solution:")
    for item, value in zip(basis, result.solution):
        print(f"  {item.name} = {fmt_complex(value)}")


def main() -> int:
    alg, base = _base_n3_without_o33_o30()

    restricted = [row[:] for row in base]
    restricted[2][2] = _current_o33(alg)
    restricted_basis = _o30_basis(alg)
    rows, rhs, labels = _linear_system(restricted, restricted_basis, _eq10_builders())
    _print_result("Restricted solve: O_3 only, corrected B33 fixed", restricted_basis, rows, rhs, labels)

    wide = [row[:] for row in base]
    wide_basis = _wide_o33_basis(alg) + _o30_basis(alg)
    rows, rhs, labels = _linear_system(wide, wide_basis, _all_coefficient_builders())
    _print_result("Joint solve: PDF B33 polynomial terms plus O_3 dressed terms", wide_basis, rows, rhs, labels)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
