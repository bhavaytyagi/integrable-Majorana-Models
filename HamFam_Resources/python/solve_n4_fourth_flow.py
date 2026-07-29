#!/usr/bin/env python3
"""Solve the remaining N=4 fourth-flow entry after the forced sign corrections."""

from __future__ import annotations

import itertools

from hamfam.clifford import CliffordExpr, minus, sum_expr
from hamfam.exact import ComplexRat, LaurentPoly, ONE_C, ZERO_C, cneg, fmt_complex
from hamfam.linear import solve_linear
from hamfam.models import (
    a_bilinear,
    beta,
    d0_op,
    d_op,
    dressed_a_complement,
    elementary_a,
    failure_counts,
    gvar,
    integrability_failures,
    n4_compact_base_table,
    term,
    weighted_a2,
)


def _coefficients(expr: CliffordExpr) -> dict[tuple[int, tuple[int, ...]], ComplexRat]:
    out = {}
    for mask, poly in expr.terms.items():
        for monomial, coeff in poly.terms.items():
            out[(mask, monomial)] = coeff
    return out


def _beta_monomial(alg, powers: tuple[int, int, int, int]) -> LaurentPoly:
    exp = [0 for _ in alg.variables]
    for index, power in enumerate(powers, start=1):
        exp[alg.variables.index(f"beta{index}")] = power
    return LaurentPoly(alg.variables, {tuple(exp): ONE_C})


def _compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


def _a_subset(alg, indices: tuple[int, ...]) -> CliffordExpr:
    return term(alg.const_poly(1), *(a_bilinear(alg, 4, index) for index in indices), alg=alg)


def _d0_e2(alg) -> CliffordExpr:
    return sum_expr(
        (
            term(beta(alg, i) * beta(alg, j), d0_op(alg), a_bilinear(alg, 4, i), a_bilinear(alg, 4, j), alg=alg)
            for i, j in itertools.combinations(range(1, 5), 2)
        ),
        alg,
    )


def _g_d_dress2(alg) -> CliffordExpr:
    return sum_expr(
        (
            term(gvar(alg, j), d_op(alg, j), dressed_a_complement(alg, 4, j, 2), alg=alg)
            for j in range(1, 5)
        ),
        alg,
    )


def _forced_fourth_flow_base():
    alg, matrix = n4_compact_base_table()
    out = [row[:] for row in matrix]

    out[1][3] = _d0_e2(alg)
    out[3][1] = out[1][3]
    out[2][3] = minus(elementary_a(alg, 4, 4).scale(3) + weighted_a2(alg, 4))
    out[3][2] = out[2][3]
    out[3][4] = _g_d_dress2(alg)

    return alg, out


def _basis_o44(alg) -> list[tuple[str, CliffordExpr]]:
    basis = []
    odd_subsets = [
        combo
        for degree in (1, 3)
        for combo in itertools.combinations(range(1, 5), degree)
    ]
    for subset in odd_subsets:
        a_part = _a_subset(alg, subset)
        subset_s = "".join(str(index) for index in subset)
        for powers in _compositions(5, 4):
            poly = _beta_monomial(alg, powers)
            powers_s = ",".join(str(power) for power in powers)
            basis.append((f"A{subset_s}:beta({powers_s})", a_part.scale(poly)))
    return basis


def _linear_system(matrix, basis):
    rows: list[list[ComplexRat]] = []
    rhs: list[ComplexRat] = []

    builders = [
        lambda m, mu=mu: m[mu][4].commutator(m[3][3]) + m[mu][3].commutator(m[3][4])
        for mu in range(3)
    ]
    for builder in builders:
        const_expr = builder(matrix)
        const_coeffs = _coefficients(const_expr)
        column_coeffs = []
        for _, expr in basis:
            trial = [row[:] for row in matrix]
            trial[3][3] = trial[3][3] + expr
            column_coeffs.append(_coefficients(builder(trial) - const_expr))

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
    alg, matrix = _forced_fourth_flow_base()
    basis = _basis_o44(alg)
    rows, rhs = _linear_system(matrix, basis)
    result = solve_linear(rows, rhs, len(basis))

    print("N=4 fourth-flow solve with forced sign corrections")
    print(f"unknowns     : {len(basis)}")
    print(f"equations    : {len(rows)}")
    print(f"rank         : {result.rank}")
    print(f"inconsistent : {len(result.inconsistent_rows)}")
    if result.inconsistent_rows:
        print("No exact B44 solution in the beta-degree-5 odd-A basis.")
        return 1

    solved = [row[:] for row in matrix]
    print("closed form:")
    print("  O24 = +D0 e2")
    print("  B34 = -3 e4 - weighted_a2")
    print("  O40 = +sum_j g_j D_j e2(A_{k != j})")
    print("  B44 = sum_j beta_j A_j e2({beta_k^2 : k != j}) + 2 weighted_a3")
    print("nonzero B44 coefficients:")
    for (name, expr), value in zip(basis, result.solution):
        if value == ZERO_C:
            continue
        print(f"  {name} = {fmt_complex(value)}")
        solved[3][3] = solved[3][3] + expr.scale(value)

    failures = integrability_failures(solved, ("t", "eps", "eta", "zeta"))
    print("full affine check:", failure_counts(failures))
    if failures:
        for failure in failures[:8]:
            print(f"  {failure.name} {failure.indices} terms={len(failure.expr.terms)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
