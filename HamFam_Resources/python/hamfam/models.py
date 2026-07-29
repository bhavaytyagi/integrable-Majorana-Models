"""Hamiltonian-family model builders and exact integrability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from .clifford import CliffordAlgebra, CliffordExpr, i_times, minus, sum_expr


def majorana_algebra(n_sites: int, extra_variables: tuple[str, ...] = ()) -> CliffordAlgebra:
    variables = tuple(f"beta{j}" for j in range(1, n_sites + 1))
    variables += tuple(f"g{j}" for j in range(1, n_sites + 1))
    variables += extra_variables
    return CliffordAlgebra(labels=(-1, 0, *range(1, 2 * n_sites + 1)), variables=variables)


def beta(alg: CliffordAlgebra, j: int, power: int = 1):
    return alg.var(f"beta{j}", power)


def inv_beta(alg: CliffordAlgebra, j: int):
    return alg.var(f"beta{j}", -1)


def gvar(alg: CliffordAlgebra, j: int):
    return alg.var(f"g{j}")


def a_bilinear(alg: CliffordAlgebra, n_sites: int, j: int) -> CliffordExpr:
    return i_times(alg.product((n_sites + j, j)))


def v_bilinear(alg: CliffordAlgebra, n_sites: int, j: int) -> CliffordExpr:
    return i_times(alg.product((n_sites + j, 0)))


def d0_op(alg: CliffordAlgebra) -> CliffordExpr:
    return i_times(alg.product((-1, 0)))


def d_op(alg: CliffordAlgebra, j: int) -> CliffordExpr:
    return i_times(alg.product((-1, j)))


def term(poly, *ops: CliffordExpr, alg: CliffordAlgebra) -> CliffordExpr:
    out = alg.scalar(poly)
    for op in ops:
        out = out * op
    return out


def q_op(alg: CliffordAlgebra, n_sites: int) -> CliffordExpr:
    return sum_expr(
        (
            term(beta(alg, i) * beta(alg, j), a_bilinear(alg, n_sites, i), a_bilinear(alg, n_sites, j), alg=alg)
            for i in range(1, n_sites + 1)
            for j in range(1, i)
        ),
        alg,
    )


def top_op(alg: CliffordAlgebra, n_sites: int) -> CliffordExpr:
    return sum_expr(
        (
            term(
                beta(alg, j)
                * sum_poly(beta(alg, k, 2) for k in range(1, n_sites + 1) if k != j),
                a_bilinear(alg, n_sites, j),
                alg=alg,
            )
            for j in range(1, n_sites + 1)
        ),
        alg,
    )


def sum_poly(parts):
    parts = list(parts)
    if not parts:
        raise ValueError("sum_poly needs at least one term")
    out = parts[0]
    for part in parts[1:]:
        out = out + part
    return out


def beta_product(alg: CliffordAlgebra, indices) -> object:
    out = alg.const_poly(1)
    for index in indices:
        out = out * beta(alg, index)
    return out


def beta_square_product(alg: CliffordAlgebra, indices) -> object:
    out = alg.const_poly(1)
    for index in indices:
        out = out * beta(alg, index, 2)
    return out


def elementary_a(alg: CliffordAlgebra, n_sites: int, degree: int) -> CliffordExpr:
    import itertools

    return sum_expr(
        (
            term(
                beta_product(alg, combo),
                *(a_bilinear(alg, n_sites, index) for index in combo),
                alg=alg,
            )
            for combo in itertools.combinations(range(1, n_sites + 1), degree)
        ),
        alg,
    )


def weighted_a1(alg: CliffordAlgebra, n_sites: int) -> CliffordExpr:
    return sum_expr(
        (
            term(
                beta(alg, j) * sum_poly(beta(alg, k, 2) for k in range(1, n_sites + 1) if k != j),
                a_bilinear(alg, n_sites, j),
                alg=alg,
            )
            for j in range(1, n_sites + 1)
        ),
        alg,
    )


def square_elementary_excluding(alg: CliffordAlgebra, n_sites: int, excluded: tuple[int, ...], degree: int):
    import itertools

    if degree == 0:
        return alg.const_poly(1)
    complement = [index for index in range(1, n_sites + 1) if index not in excluded]
    parts = [beta_square_product(alg, combo) for combo in itertools.combinations(complement, degree)]
    return sum_poly(parts) if parts else alg.zero_poly()


def weighted_a1_square_elementary(alg: CliffordAlgebra, n_sites: int, degree: int) -> CliffordExpr:
    return sum_expr(
        (
            term(
                beta(alg, j) * square_elementary_excluding(alg, n_sites, (j,), degree),
                a_bilinear(alg, n_sites, j),
                alg=alg,
            )
            for j in range(1, n_sites + 1)
        ),
        alg,
    )


def square_weighted_a(
    alg: CliffordAlgebra,
    n_sites: int,
    a_degree: int,
    square_degree: int,
) -> CliffordExpr:
    import itertools

    parts = []
    for subset in itertools.combinations(range(1, n_sites + 1), a_degree):
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


def weighted_a2(alg: CliffordAlgebra, n_sites: int) -> CliffordExpr:
    import itertools

    return sum_expr(
        (
            term(
                beta(alg, i)
                * beta(alg, j)
                * sum_poly(beta(alg, k, 2) for k in range(1, n_sites + 1) if k not in (i, j)),
                a_bilinear(alg, n_sites, i),
                a_bilinear(alg, n_sites, j),
                alg=alg,
            )
            for i, j in itertools.combinations(range(1, n_sites + 1), 2)
        ),
        alg,
    )


def weighted_a3(alg: CliffordAlgebra, n_sites: int) -> CliffordExpr:
    import itertools

    return sum_expr(
        (
            term(
                beta_product(alg, combo)
                * sum_poly(beta(alg, k, 2) for k in range(1, n_sites + 1) if k not in combo),
                *(a_bilinear(alg, n_sites, index) for index in combo),
                alg=alg,
            )
            for combo in itertools.combinations(range(1, n_sites + 1), 3)
        ),
        alg,
    )


def dressed_a_complement(alg: CliffordAlgebra, n_sites: int, j: int, degree: int) -> CliffordExpr:
    import itertools

    complement = [index for index in range(1, n_sites + 1) if index != j]
    return sum_expr(
        (
            term(
                beta_product(alg, combo),
                *(a_bilinear(alg, n_sites, index) for index in combo),
                alg=alg,
            )
            for combo in itertools.combinations(complement, degree)
        ),
        alg,
    )


def d0_elementary_a(alg: CliffordAlgebra, n_sites: int, degree: int) -> CliffordExpr:
    return term(alg.const_poly(1), d0_op(alg), elementary_a(alg, n_sites, degree), alg=alg)


def constant_dressing(alg: CliffordAlgebra, n_sites: int, degree: int, sign: int) -> CliffordExpr:
    expr = sum_expr(
        (
            term(gvar(alg, j), d_op(alg, j), dressed_a_complement(alg, n_sites, j, degree), alg=alg)
            for j in range(1, n_sites + 1)
        ),
        alg,
    )
    return expr if sign > 0 else minus(expr)


def n3_majorana_table() -> tuple[CliffordAlgebra, list[list[CliffordExpr]]]:
    """Return the N=3 (O_mu | B_{mu nu}) table from the PDF Majorana-family transcription.

    Columns 1..3 are the Hessian block, and column 4 is the constant column.
    """

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
    matrix[2][2] = (
        term(
            alg.const_poly(2) * beta(alg, 1) * beta(alg, 2) * beta(alg, 3),
            a_bilinear(alg, n_sites, 1),
            a_bilinear(alg, n_sites, 2),
            a_bilinear(alg, n_sites, 3),
            alg=alg,
        )
        + top_op(alg, n_sites)
    )
    dressings = (
        term(
            gvar(alg, j),
            d_op(alg, j),
            sum_expr((term(beta(alg, k), a_bilinear(alg, n_sites, k), alg=alg) for k in range(1, 4) if k != j), alg),
            alg=alg,
        )
        for j in range(1, 4)
    )
    matrix[2][3] = minus(sum_expr(dressings, alg))

    return alg, matrix


def n4_three_parameter_table() -> tuple[CliffordAlgebra, list[list[CliffordExpr]]]:
    """N=4 restriction to the verified (t, eps, eta) family."""

    n_sites = 4
    alg = majorana_algebra(n_sites, extra_variables=("t", "eps", "eta"))
    matrix = [[alg.zero() for _ in range(4)] for _ in range(3)]

    matrix[0][0] = elementary_a(alg, n_sites, 1)
    matrix[0][1] = d0_op(alg)
    matrix[0][2] = minus(elementary_a(alg, n_sites, 2))
    matrix[0][3] = sum_expr((term(gvar(alg, j), d_op(alg, j), alg=alg) for j in range(1, n_sites + 1)), alg)

    matrix[1][0] = matrix[0][1]
    matrix[1][1] = sum_expr(
        (term(inv_beta(alg, j), a_bilinear(alg, n_sites, j), alg=alg) for j in range(1, n_sites + 1)),
        alg,
    )
    matrix[1][2] = minus(
        sum_expr(
            (term(beta(alg, j), d0_op(alg), a_bilinear(alg, n_sites, j), alg=alg) for j in range(1, n_sites + 1)),
            alg,
        )
    )
    matrix[1][3] = minus(
        sum_expr(
            (term(gvar(alg, j) * inv_beta(alg, j), v_bilinear(alg, n_sites, j), alg=alg) for j in range(1, n_sites + 1)),
            alg,
        )
    )

    matrix[2][0] = matrix[0][2]
    matrix[2][1] = matrix[1][2]
    matrix[2][2] = elementary_a(alg, n_sites, 3).scale(2) + weighted_a1(alg, n_sites)
    matrix[2][3] = minus(
        sum_expr(
            (
                term(gvar(alg, j), d_op(alg, j), dressed_a_complement(alg, n_sites, j, 1), alg=alg)
                for j in range(1, n_sites + 1)
            ),
            alg,
        )
    )

    return alg, matrix


def n4_compact_base_table() -> tuple[CliffordAlgebra, list[list[CliffordExpr]]]:
    """N=4 table with B34/B44 left as zero for ansatz solving."""

    n_sites = 4
    alg = majorana_algebra(n_sites, extra_variables=("t", "eps", "eta", "zeta"))
    matrix = [[alg.zero() for _ in range(5)] for _ in range(4)]

    matrix[0][0] = elementary_a(alg, n_sites, 1)
    matrix[0][1] = d0_op(alg)
    matrix[0][2] = minus(elementary_a(alg, n_sites, 2))
    matrix[0][3] = elementary_a(alg, n_sites, 3)
    matrix[0][4] = sum_expr((term(gvar(alg, j), d_op(alg, j), alg=alg) for j in range(1, n_sites + 1)), alg)

    matrix[1][0] = matrix[0][1]
    matrix[1][1] = sum_expr(
        (term(inv_beta(alg, j), a_bilinear(alg, n_sites, j), alg=alg) for j in range(1, n_sites + 1)),
        alg,
    )
    matrix[1][2] = minus(
        sum_expr(
            (term(beta(alg, j), d0_op(alg), a_bilinear(alg, n_sites, j), alg=alg) for j in range(1, n_sites + 1)),
            alg,
        )
    )
    matrix[1][3] = minus(
        sum_expr(
            (
                term(
                    beta(alg, i) * beta(alg, j),
                    d0_op(alg),
                    a_bilinear(alg, n_sites, i),
                    a_bilinear(alg, n_sites, j),
                    alg=alg,
                )
                for i in range(1, n_sites + 1)
                for j in range(i + 1, n_sites + 1)
            ),
            alg,
        )
    )
    matrix[1][4] = minus(
        sum_expr(
            (term(gvar(alg, j) * inv_beta(alg, j), v_bilinear(alg, n_sites, j), alg=alg) for j in range(1, n_sites + 1)),
            alg,
        )
    )

    matrix[2][0] = matrix[0][2]
    matrix[2][1] = matrix[1][2]
    matrix[2][2] = elementary_a(alg, n_sites, 3).scale(2) + weighted_a1(alg, n_sites)
    matrix[2][4] = minus(
        sum_expr(
            (
                term(gvar(alg, j), d_op(alg, j), dressed_a_complement(alg, n_sites, j, 1), alg=alg)
                for j in range(1, n_sites + 1)
            ),
            alg,
        )
    )

    matrix[3][0] = matrix[0][3]
    matrix[3][1] = matrix[1][3]
    matrix[3][4] = minus(
        sum_expr(
            (
                term(gvar(alg, j), d_op(alg, j), dressed_a_complement(alg, n_sites, j, 2), alg=alg)
                for j in range(1, n_sites + 1)
            ),
            alg,
        )
    )

    return alg, matrix


def n4_majorana_table() -> tuple[CliffordAlgebra, list[list[CliffordExpr]]]:
    """Return the exact N=4 Majorana table found by the fourth-flow solve."""

    n_sites = 4
    alg, matrix = n4_compact_base_table()

    matrix[1][3] = sum_expr(
        (
            term(
                beta(alg, i) * beta(alg, j),
                d0_op(alg),
                a_bilinear(alg, n_sites, i),
                a_bilinear(alg, n_sites, j),
                alg=alg,
            )
            for i in range(1, n_sites + 1)
            for j in range(i + 1, n_sites + 1)
        ),
        alg,
    )
    matrix[3][1] = matrix[1][3]

    matrix[2][3] = minus(elementary_a(alg, n_sites, 4).scale(3) + weighted_a2(alg, n_sites))
    matrix[3][2] = matrix[2][3]

    matrix[3][3] = weighted_a1_square_elementary(alg, n_sites, 2) + weighted_a3(alg, n_sites).scale(2)
    matrix[3][4] = sum_expr(
        (
            term(gvar(alg, j), d_op(alg, j), dressed_a_complement(alg, n_sites, j, 2), alg=alg)
            for j in range(1, n_sites + 1)
        ),
        alg,
    )

    return alg, matrix


def n5_majorana_table() -> tuple[CliffordAlgebra, list[list[CliffordExpr]]]:
    """Return the exact N=5 table found by the structured ansatz solve."""

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
    matrix[2][4] = elementary_a(alg, n_sites, 5).scale(4) + square_weighted_a(alg, n_sites, 3, 1)
    matrix[2][5] = constant_dressing(alg, n_sites, 1, -1)

    matrix[3][0] = matrix[0][3]
    matrix[3][1] = matrix[1][3]
    matrix[3][2] = matrix[2][3]
    matrix[3][3] = (
        square_weighted_a(alg, n_sites, 1, 2)
        + square_weighted_a(alg, n_sites, 3, 1).scale(2)
        + elementary_a(alg, n_sites, 5).scale(6)
    )
    matrix[3][4] = minus(square_weighted_a(alg, n_sites, 2, 2) + square_weighted_a(alg, n_sites, 4, 1).scale(3))
    matrix[3][5] = constant_dressing(alg, n_sites, 2, 1)

    matrix[4][0] = matrix[0][4]
    matrix[4][1] = matrix[1][4]
    matrix[4][2] = matrix[2][4]
    matrix[4][3] = matrix[3][4]
    matrix[4][4] = square_weighted_a(alg, n_sites, 1, 3) + square_weighted_a(alg, n_sites, 3, 2).scale(2)
    matrix[4][5] = constant_dressing(alg, n_sites, 3, -1)

    return alg, matrix


@dataclass(frozen=True)
class ConstraintFailure:
    name: str
    indices: tuple[int, ...]
    expr: CliffordExpr


def build_hamiltonians(matrix: list[list[CliffordExpr]], tau_names: tuple[str, ...]) -> list[CliffordExpr]:
    alg = matrix[0][0].algebra
    out = []
    for row in matrix:
        h = row[len(tau_names)]
        for col, name in enumerate(tau_names):
            h = h + row[col].scale(alg.var(name))
        out.append(h)
    return out


def integrability_failures(matrix: list[list[CliffordExpr]], tau_names: tuple[str, ...]) -> list[ConstraintFailure]:
    failures: list[ConstraintFailure] = []
    m = len(tau_names)

    for mu in range(m):
        for nu in range(mu + 1, m):
            if not (matrix[mu][nu] - matrix[nu][mu]).is_zero:
                failures.append(ConstraintFailure("symmetry", (mu + 1, nu + 1), matrix[mu][nu] - matrix[nu][mu]))

    for mu in range(m):
        for nu in range(mu + 1, m):
            for alpha in range(m):
                expr = matrix[mu][alpha].commutator(matrix[nu][alpha])
                if not expr.is_zero:
                    failures.append(ConstraintFailure("eq13", (mu + 1, nu + 1, alpha + 1), expr))
            for alpha in range(m):
                for beta_col in range(alpha + 1, m):
                    expr = matrix[mu][alpha].commutator(matrix[nu][beta_col]) + matrix[mu][beta_col].commutator(
                        matrix[nu][alpha]
                    )
                    if not expr.is_zero:
                        failures.append(ConstraintFailure("eq14", (mu + 1, nu + 1, alpha + 1, beta_col + 1), expr))

    zcol = m
    for mu in range(m):
        for nu in range(mu + 1, m):
            expr = matrix[mu][zcol].commutator(matrix[nu][zcol])
            if not expr.is_zero:
                failures.append(ConstraintFailure("eq00", (mu + 1, nu + 1), expr))
            for alpha in range(m):
                expr = matrix[mu][alpha].commutator(matrix[nu][zcol]) + matrix[mu][zcol].commutator(matrix[nu][alpha])
                if not expr.is_zero:
                    failures.append(ConstraintFailure("eq10", (mu + 1, nu + 1, alpha + 1), expr))

    hamiltonians = build_hamiltonians(matrix, tau_names)
    for mu in range(m):
        for nu in range(mu + 1, m):
            expr = hamiltonians[mu].commutator(hamiltonians[nu])
            if not expr.is_zero:
                failures.append(ConstraintFailure("full", (mu + 1, nu + 1), expr))

    return failures


def failure_counts(failures: list[ConstraintFailure]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for failure in failures:
        counts[failure.name] = counts.get(failure.name, 0) + 1
    return counts
