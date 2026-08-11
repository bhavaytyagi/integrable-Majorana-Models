#!/usr/bin/env python3.12
"""Exact mathematical-consistency check of the v2 draft.

Every displayed formula of A_Search_for_a_Hamiltonians_Family_Revised_V2.tex
that makes an algebraic claim is transcribed here *literally, from the paper*
and compared against the closed-form table (Table 3, Eq. (4.21)) in the exact
Clifford algebra over Laurent polynomials in (beta_j, g_j, tau_nu).

This is deliberately an INDEPENDENT transcription: the closed form is
re-implemented from the printed Table 3 rather than imported, and then also
cross-checked against the verified builder in extension_test.py.  A mismatch
means the paper and the verified algebra disagree.

Run:  python3 check_paper_consistency.py
Exit status is nonzero if any check fails.
"""

from __future__ import annotations

import itertools
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HamFam_Resources" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hamfam.clifford import CliffordExpr, minus, one, sum_expr  # noqa: E402
from hamfam.models import (  # noqa: E402
    a_bilinear,
    v_bilinear,
    beta,
    d0_op,
    d_op,
    gvar,
    integrability_failures,
    inv_beta,
    majorana_algebra,
    term,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# Paper-notation building blocks (Eqs. (2.2), (2.3))
# ---------------------------------------------------------------------------

def A_bil(alg, N, j):
    """A_j = i beta_j gamma_{N+j} gamma_j."""
    return term(beta(alg, j), a_bilinear(alg, N, j), alg=alg)


def V(alg, N, j):
    """V_j = (i/beta_j) gamma_{N+j} gamma_0."""
    return term(inv_beta(alg, j), v_bilinear(alg, N, j), alg=alg)


def D0(alg):
    return d0_op(alg)


def D(alg, j):
    return d_op(alg, j)


def A_subset(alg, N, I):
    """A_I = prod_{i in I} A_i."""
    out = one(alg)
    for i in I:
        out = out * A_bil(alg, N, i)
    return out


def e_r(alg, N, r, universe=None):
    """e_r(A) over the given index universe (default {1..N})."""
    if universe is None:
        universe = range(1, N + 1)
    universe = list(universe)
    if r < 0 or r > len(universe):
        return alg.zero()
    return sum_expr((A_subset(alg, N, I) for I in itertools.combinations(universe, r)), alg)


def e_beta2(alg, s, universe):
    """e_s({beta_k^2 : k in universe}) as a scalar polynomial."""
    universe = list(universe)
    if s < 0 or s > len(universe):
        return alg.zero_poly()
    out = alg.zero_poly()
    for J in itertools.combinations(universe, s):
        p = alg.const_poly(1)
        for k in J:
            p = p * beta(alg, k, 2)
        out = out + p
    return out


def coeff_of(alg, expr, powers):
    """Extract the CliffordExpr coefficient of the monomial given by `powers`."""
    idx = {name: alg.variables.index(name) for name in powers}
    out: dict[int, object] = {}
    for mask, poly in expr.terms.items():
        acc = alg.zero_poly()
        for exp, c in poly.terms.items():
            if all(exp[idx[n]] == p for n, p in powers.items()):
                e = list(exp)
                for n in powers:
                    e[idx[n]] = 0
                acc = acc + type(poly)(poly.variables, {tuple(e): c})
        if not acc.is_zero:
            out[mask] = acc
    return CliffordExpr(alg, out)


# ---------------------------------------------------------------------------
# The closed form, transcribed from printed Table 3 + Eqs. (4.20), (4.21)
# ---------------------------------------------------------------------------

def a_index(mu):
    """Eq. (4.20): a_1 = 0, a_mu = mu - 2 for mu >= 3."""
    return 0 if mu == 1 else mu - 2


def K_paper(alg, N, a, b):
    """Eq. (4.21):
    K^{(N)}_{a,b} = sum_s C(a+b-2s, a-s) sum_{|I|=a+b+1-2s} A_I e_s({beta_k^2 : k not in I}).
    """
    out = alg.zero()
    for s in range(0, max(a, b) + 2):
        c = comb(a + b - 2 * s, a - s) if 0 <= a - s <= a + b - 2 * s else 0
        if c == 0:
            continue
        d = a + b + 1 - 2 * s
        if d < 0 or d > N:
            continue
        for I in itertools.combinations(range(1, N + 1), d):
            compl = [k for k in range(1, N + 1) if k not in I]
            es = e_beta2(alg, s, compl)
            if es.is_zero:
                continue
            out = out + A_subset(alg, N, I).scale(es).scale(c)
    return out


def paper_table(N, rows, extra=()):
    """Build the (O_mu | B_{mu nu}) table from printed Table 3 for the given row index set.

    `rows` is the index set (1, 2, 3, ..., top).  Returns (alg, matrix, tau_names)
    with matrix[mu-1][nu-1] = B_{mu,nu} and matrix[mu-1][-1] = O_mu.
    """
    top = max(rows)
    tau_names = tuple(f"tau{i}" for i in range(1, top + 1))
    alg = majorana_algebra(N, extra_variables=tau_names + tuple(extra))
    M = [[alg.zero() for _ in range(top + 1)] for _ in range(top)]
    script_I = [m for m in rows if m != 2]

    for mu in script_I:
        for nu in script_I:
            a, b = a_index(mu), a_index(nu)
            e = K_paper(alg, N, a, b)
            if (a + b) % 2:
                e = minus(e)
            M[mu - 1][nu - 1] = e

    for mu in script_I:                       # Table 3, row/column 2
        a = a_index(mu)
        e = D0(alg) * e_r(alg, N, a)
        M[1][mu - 1] = M[mu - 1][1] = e if a % 2 == 0 else minus(e)
    M[1][1] = sum_expr(
        (A_bil(alg, N, j).scale(beta(alg, j, -2)) for j in range(1, N + 1)), alg
    )

    for mu in script_I:                       # Table 3, constant column
        a = a_index(mu)
        e = sum_expr(
            (
                term(gvar(alg, j), D(alg, j), e_r(alg, N, a, [k for k in range(1, N + 1) if k != j]), alg=alg)
                for j in range(1, N + 1)
            ),
            alg,
        )
        M[mu - 1][top] = e if a % 2 == 0 else minus(e)
    M[1][top] = minus(sum_expr((term(gvar(alg, j), V(alg, N, j), alg=alg) for j in range(1, N + 1)), alg))

    return alg, M, tau_names


def H_of(alg, M, mu, tau_names):
    """H_mu = O_mu + sum_nu tau_nu B_{mu nu}."""
    top = len(M)
    out = M[mu - 1][top]
    for nu in range(1, top + 1):
        out = out + M[mu - 1][nu - 1].scale(alg.var(tau_names[nu - 1]))
    return out


# ---------------------------------------------------------------------------

def check_table_vs_code():
    section("Table 3 (paper transcription) vs the verified builder / integrability")
    from extension_test import build_table

    for N, top in ((2, 2), (3, 3), (4, 4), (2, 3), (3, 4), (4, 5), (2, 4), (3, 5)):
        alg_p, Mp, xs = paper_table(N, range(1, top + 1))
        alg_c, Mc, xc = build_table(N, top)
        same = all(
            (Mp[i][j] - Mc[i][j]).is_zero
            for i in range(top)
            for j in range(top + 1)
        )
        check(f"N={N}, {top} flows: printed Table 3 == extension_test builder", same)
        if top <= N + 1:
            fails = integrability_failures(Mp, xs)
            check(f"N={N}, {top} flows: printed Table 3 satisfies (3.12)-(3.15)", not fails,
                  str(fails[:1]))


def check_section2_four_lines():
    section("Section 2.2 -- the family 'in four lines' (Eqs. (2.6)-(2.10))")
    for N in (2, 3, 4):
        alg, M, xs = paper_table(N, range(1, N + 1), extra=("z", "w"))
        z, w = alg.var("z"), alg.var("w")
        script_I = [m for m in range(1, N + 1) if m != 2]

        # Eq. (2.6): B_{mu nu} = (-1)^{a+b} [z^a w^b] (E(z)E(w) - prod(1+beta^2 zw))/(z+w).
        # The quotient's coefficients are extracted by the recursion K_{a,b} = R_{a,b} - K_{a-1,b},
        # which is exactly polynomial division by (z+w); we verify the division is exact.
        E = lambda u: _E(alg, N, u)
        R = E(z) * E(w)
        P0 = one(alg)
        for j in range(1, N + 1):
            P0 = P0 * (one(alg) + alg.scalar(beta(alg, j, 2) * z * w))
        R = R - P0

        # (z+w)G = R  <=>  R_{a,b} = K_{a-1,b} + K_{a,b-1}  (K_{-1,b} = K_{a,-1} = 0),
        # which solves uniquely as K_{a,b} = R_{a+1,b} - K_{a+1,b-1}.
        Kq = {}
        for b in range(0, N + 2):
            for a in range(N + 1, -1, -1):
                Rab = coeff_of(alg, R, {"z": a + 1, "w": b})
                Kq[(a, b)] = Rab - Kq.get((a + 1, b - 1), alg.zero())
        # exactness of the division: (z+w) * sum K z^a w^b == R
        G = alg.zero()
        for (a, b), k in Kq.items():
            if not k.is_zero:
                G = G + k.scale(alg.var("z", a) * alg.var("w", b))
        check(f"N={N}: (z+w) G(z,w) == E(z)E(w) - prod(1+beta_j^2 zw)  [Eq. (2.6) well posed]",
              (G.scale(z + w) - R).is_zero)

        ok = True
        for mu in script_I:
            for nu in script_I:
                a, b = a_index(mu), a_index(nu)
                lhs = M[mu - 1][nu - 1]
                rhs = Kq[(a, b)]
                if (a + b) % 2:
                    rhs = minus(rhs)
                ok &= (lhs - rhs).is_zero
        check(f"N={N}: Eq. (2.6) coefficient extraction == Table 3 block", ok)

        # Eq. (2.7): B_{2 nu} = (-1)^{a_nu} D0 e_{a_nu}(A);  B_22 = sum A_j/beta_j^2
        ok = all(
            (M[1][nu - 1] - (D0(alg) * e_r(alg, N, a_index(nu))).scale((-1) ** a_index(nu))).is_zero
            for nu in script_I
        )
        check(f"N={N}: Eq. (2.7) row 2 == Table 3", ok)
        check(f"N={N}: Eq. (2.7) B_22 == sum_j A_j/beta_j^2",
              (M[1][1] - sum_expr((A_bil(alg, N, j).scale(beta(alg, j, -2)) for j in range(1, N + 1)), alg)).is_zero)

        # Eq. (2.8): O_mu = (-1)^{a_mu} sum_j g_j D_j e_{a_mu}(A^{(j)});  O_2 = -sum g_j V_j
        ok = True
        for mu in script_I:
            a = a_index(mu)
            rhs = sum_expr(
                (term(gvar(alg, j), D(alg, j), e_r(alg, N, a, [k for k in range(1, N + 1) if k != j]), alg=alg)
                 for j in range(1, N + 1)), alg).scale((-1) ** a)
            ok &= (M[mu - 1][N] - rhs).is_zero
        check(f"N={N}: Eq. (2.8) constant column == Table 3", ok)

        # Eqs. (2.9), (2.10): the written-out H_1 and H_2.
        tau = [alg.var(x) for x in xs]
        H1_paper = (
            D0(alg).scale(tau[1])
            + sum_expr((A_bil(alg, N, j) for j in range(1, N + 1)), alg).scale(tau[0])
            + sum_expr((term(gvar(alg, j), D(alg, j), alg=alg) for j in range(1, N + 1)), alg)
            + sum_expr((e_r(alg, N, nu - 1).scale(tau[nu - 1]).scale((-1) ** nu)
                        for nu in range(3, N + 1)), alg)
        )
        check(f"N={N}: Eq. (2.9) H_1 == table row 1", (H1_paper - H_of(alg, M, 1, xs)).is_zero)

        H2_paper = (
            D0(alg).scale(tau[0])
            + sum_expr((A_bil(alg, N, j).scale(beta(alg, j, -2)) for j in range(1, N + 1)), alg).scale(tau[1])
            - sum_expr((term(gvar(alg, j), V(alg, N, j), alg=alg) for j in range(1, N + 1)), alg)
            + sum_expr(((D0(alg) * e_r(alg, N, nu - 2)).scale(tau[nu - 1]).scale((-1) ** nu)
                        for nu in range(3, N + 1)), alg)
        )
        check(f"N={N}: Eq. (2.10) H_2 == table row 2", (H2_paper - H_of(alg, M, 2, xs)).is_zero)


def _E(alg, N, u):
    """E(u) = prod_j (1 + A_j u), Eq. (2.4)."""
    out = one(alg)
    for j in range(1, N + 1):
        out = out * (one(alg) + A_bil(alg, N, j).scale(u))
    return out


# ---------------------------------------------------------------------------
# Section 2.3 -- fermionization (Eqs. (2.11)-(2.14))
# ---------------------------------------------------------------------------

def check_section2_fermions():
    section("Section 2.3 -- N standard fermions coupled to a Majorana")
    for N in (2, 3, 4):
        alg, M, xs = paper_table(N, range(1, N + 1))
        g = lambda k: alg.gen(k)
        half = alg.scalar((__import__("fractions").Fraction(1, 2), __import__("fractions").Fraction(0)))

        def f(j):        # Eq. (2.11): f_j = (gamma_j - i gamma_{N+j})/2
            return (g(j) - g(N + j).scale((0, 1))) * half

        def fd(j):       # f_j^dagger = (gamma_j + i gamma_{N+j})/2
            return (g(j) + g(N + j).scale((0, 1))) * half

        ok = all((f(j) * fd(k) + fd(k) * f(j) - (one(alg) if j == k else alg.zero())).is_zero
                 for j in range(1, N + 1) for k in range(1, N + 1))
        check(f"N={N}: Eq. (2.11) {{f_j, f_k^dag}} = delta_jk", ok)
        ok = all((f(j) * f(k) + f(k) * f(j)).is_zero
                 for j in range(1, N + 1) for k in range(1, N + 1))
        check(f"N={N}: Eq. (2.11) {{f_j, f_k}} = 0", ok)

        # Eq. (2.12): i gamma_{N+j} gamma_j = 2 n_j - 1,  A_j = beta_j (2 n_j - 1)
        n = lambda j: fd(j) * f(j)
        ok = all((a_bilinear(alg, N, j) - (n(j).scale(2) - one(alg))).is_zero for j in range(1, N + 1))
        check(f"N={N}: Eq. (2.12) i gamma_(N+j) gamma_j == 2 n_j - 1", ok)
        ok = all((A_bil(alg, N, j) - (n(j).scale(2) - one(alg)).scale(beta(alg, j))).is_zero
                 for j in range(1, N + 1))
        check(f"N={N}: Eq. (2.12) A_j == beta_j (2 n_j - 1)", ok)
        ok = all((g(j) - (f(j) + fd(j))).is_zero and
                 (g(N + j) - (f(j) - fd(j)).scale((0, 1))).is_zero for j in range(1, N + 1))
        check(f"N={N}: Eq. (2.12) gamma_j = f+f^dag, gamma_(N+j) = i(f-f^dag)", ok)

        # Eq. (2.13): H_1 in fermionic variables (tau_nu = 0 for nu >= 4, tau_3 = eta)
        t, eps = alg.var(xs[0]), alg.var(xs[1])
        eta = alg.var(xs[2]) if N >= 3 else None
        c0, gamma0 = g(-1), g(0)
        H1_ferm = (
            (c0 * gamma0).scale((0, 1)).scale(eps)
            + sum_expr((((n(j).scale(2) - one(alg)).scale(beta(alg, j) * t))
                        + ((c0 * (f(j) + fd(j))).scale((0, 1)).scale(gvar(alg, j)))
                        for j in range(1, N + 1)), alg)
        )
        if N >= 3:
            H1_ferm = H1_ferm - sum_expr(
                (((n(i).scale(2) - one(alg)) * (n(j).scale(2) - one(alg)))
                 .scale(beta(alg, i) * beta(alg, j) * eta)
                 for i, j in itertools.combinations(range(1, N + 1), 2)), alg)
        H1_table = H_of(alg, M, 1, xs)
        if N >= 4:      # switch off tau_nu, nu >= 4
            for nu in range(4, N + 1):
                H1_table = H1_table - M[0][nu - 1].scale(alg.var(xs[nu - 1]))
        check(f"N={N}: Eq. (2.13) H_1 in fermion variables == table row 1", (H1_ferm - H1_table).is_zero)

        # Eq. (2.14): d = (gamma_0 - i c_0)/2
        d = (gamma0 - c0.scale((0, 1))) * half
        dd = (gamma0 + c0.scale((0, 1))) * half
        check(f"N={N}: Eq. (2.14) i c_0 gamma_0 == 2 d^dag d - 1",
              ((c0 * gamma0).scale((0, 1)) - (dd * d).scale(2) + one(alg)).is_zero)
        check(f"N={N}: Eq. (2.14) i c_0 == d^dag - d", (c0.scale((0, 1)) - (dd - d)).is_zero)
        check(f"N={N}: Eq. (2.14) (d^dag-d)(f+f^dag) expansion",
              all(((dd - d) * (f(j) + fd(j))
                   - (dd * f(j) - d * fd(j) + dd * fd(j) - d * f(j))).is_zero
                  for j in range(1, N + 1)))


# ---------------------------------------------------------------------------
# Section 4 -- the gamma-magnet family (Eqs. (4.1)-(4.13), Table 2)
# ---------------------------------------------------------------------------

def fermionize(alg, expr):
    """The rule of Section 4: odd monomials M (in the gammas) -> i c_0 M."""
    c0_bit = 1 << alg.label_to_bit[-1]
    out = alg.zero()
    ic0 = alg.gen(-1).scale((0, 1))
    for mask, poly in expr.terms.items():
        piece = CliffordExpr(alg, {mask: poly})
        if (mask & ~c0_bit).bit_count() % 2:
            piece = ic0 * piece
        out = out + piece
    return out


def check_section4():
    section("Section 4 -- gamma-magnet family, fermionization, N=3 table, "
            "and the Appendix C N=4 members")
    for N in (3, 4):
        alg, M, xs = paper_table(N, range(1, N + 1))
        t, eps = alg.var(xs[0]), alg.var(xs[1])
        eta = alg.var(xs[2])
        gam = lambda k: alg.gen(k)
        c0, gamma0 = gam(-1), gam(0)

        # Eqs. (4.8)-(4.13): the local commutator identities
        As = [None] + [A_bil(alg, N, j) for j in range(1, N + 1)]
        Vs = [None] + [V(alg, N, j) for j in range(1, N + 1)]
        Ds = [None] + [D(alg, j) for j in range(1, N + 1)]
        d0 = D0(alg)
        ok = all(As[i].commutator(As[j]).is_zero for i in range(1, N + 1) for j in range(1, N + 1))
        check(f"N={N}: Eq. (4.8) [A_i,A_j] = 0", ok)
        check(f"N={N}: Eq. (4.9) [A_i,D_0] = 0",
              all(As[i].commutator(d0).is_zero for i in range(1, N + 1)))
        check(f"N={N}: Eq. (4.10) [A_i,D_j] = 0 (i != j)",
              all(As[i].commutator(Ds[j]).is_zero
                  for i in range(1, N + 1) for j in range(1, N + 1) if i != j))
        check(f"N={N}: Eq. (4.11) [A_j,D_j] + beta_j^2 [D_0,V_j] = 0",
              all((As[j].commutator(Ds[j]) + d0.commutator(Vs[j]).scale(beta(alg, j, 2))).is_zero
                  for j in range(1, N + 1)))
        check(f"N={N}: Eq. (4.12) [A_j,V_j] + [D_0,D_j] = 0",
              all((As[j].commutator(Vs[j]) + d0.commutator(Ds[j])).is_zero
                  for j in range(1, N + 1)))
        check(f"N={N}: Eq. (4.13) [V_j,D_j] = 0",
              all(Vs[j].commutator(Ds[j]).is_zero for j in range(1, N + 1)))

        # Eqs. (4.1), (4.2): the spin/gamma forms, and their fermionic images (4.4), (4.5)
        H1g = (
            gamma0.scale(eps)
            + sum_expr((((gam(N + j) * gam(j)).scale((0, 1)).scale(beta(alg, j) * t))
                        + gam(j).scale(gvar(alg, j)) for j in range(1, N + 1)), alg)
            + sum_expr((((gam(N + i) * gam(i)) * (gam(N + j) * gam(j)))
                        .scale(beta(alg, i) * beta(alg, j) * eta)
                        for i, j in itertools.combinations(range(1, N + 1), 2)), alg)
        )
        H2g = (
            gamma0.scale(t)
            + sum_expr((
                (gam(N + j) * gam(j)).scale((0, 1)).scale(eps * beta(alg, j, -1))
                - (gam(N + j) * gamma0).scale((0, 1)).scale(gvar(alg, j) * beta(alg, j, -1))
                - (gamma0 * (gam(N + j) * gam(j)).scale((0, 1))).scale(eta * beta(alg, j))
                for j in range(1, N + 1)), alg)
        )
        H1f = (
            (c0 * gamma0).scale((0, 1)).scale(eps)
            + sum_expr((((gam(N + j) * gam(j)).scale((0, 1)).scale(beta(alg, j) * t))
                        + (c0 * gam(j)).scale((0, 1)).scale(gvar(alg, j)) for j in range(1, N + 1)), alg)
            + sum_expr((((gam(N + i) * gam(i)) * (gam(N + j) * gam(j)))
                        .scale(beta(alg, i) * beta(alg, j) * eta)
                        for i, j in itertools.combinations(range(1, N + 1), 2)), alg)
        )
        H2f = (
            (c0 * gamma0).scale((0, 1)).scale(t)
            + sum_expr((
                (gam(N + j) * gam(j)).scale((0, 1)).scale(eps * beta(alg, j, -1))
                - (gam(N + j) * gamma0).scale((0, 1)).scale(gvar(alg, j) * beta(alg, j, -1))
                + ((c0 * gamma0) * (gam(N + j) * gam(j))).scale(eta * beta(alg, j))
                for j in range(1, N + 1)), alg)
        )
        check(f"N={N}: fermionization rule maps Eq. (4.1) -> Eq. (4.4)",
              (fermionize(alg, H1g) - H1f).is_zero)
        check(f"N={N}: fermionization rule maps Eq. (4.2) -> Eq. (4.5)",
              (fermionize(alg, H2g) - H2f).is_zero)

        # H_1, H_2 of the table, with tau_nu = 0 for nu >= 4
        def truncated(mu):
            out = H_of(alg, M, mu, xs)
            for nu in range(4, N + 1):
                out = out - M[mu - 1][nu - 1].scale(alg.var(xs[nu - 1]))
            return out

        check(f"N={N}: Eq. (4.4) H_1 == table row 1 (tau_nu=0, nu>=4)", (H1f - truncated(1)).is_zero)
        check(f"N={N}: Eq. (4.5) H_2 == table row 2 (tau_nu=0, nu>=4)", (H2f - truncated(2)).is_zero)

    # Eq. (4.3)/(4.6): the N=3 third member, gamma form and fermionic form
    N = 3
    alg, M, xs = paper_table(N, range(1, N + 1))
    t, eps, eta = (alg.var(x) for x in xs)
    gam = lambda k: alg.gen(k)
    c0, gamma0 = gam(-1), gam(0)
    bil = lambda j: gam(N + j) * gam(j)
    ibil = lambda j: bil(j).scale((0, 1))

    common_t = sum_expr(((bil(i) * bil(j)).scale(beta(alg, i) * beta(alg, j) * t)
                         for i, j in itertools.combinations((1, 2, 3), 2)), alg)
    common_eta = (
        sum_expr(((bil(i) * bil(j) * bil(k))
                  .scale(beta(alg, i) * beta(alg, j) * beta(alg, k) * eta).scale((0, -2))
                  for i, j, k in itertools.combinations((1, 2, 3), 3)), alg)
        + sum_expr((ibil(j).scale(beta(alg, j) * eta
                                  * _sum_poly(alg, [beta(alg, k, 2) for k in (1, 2, 3) if k != j]))
                    for j in (1, 2, 3)), alg)
    )
    H3g = (
        common_t
        - sum_expr(((gamma0 * ibil(j)).scale(beta(alg, j) * eps) for j in (1, 2, 3)), alg)
        + common_eta
        - sum_expr((gam(j).scale(gvar(alg, j))
                    * sum_expr((ibil(k).scale(beta(alg, k)) for k in (1, 2, 3) if k != j), alg)
                    for j in (1, 2, 3)), alg)
    )
    H3f = (
        common_t
        + sum_expr((((c0 * gamma0) * bil(j)).scale(beta(alg, j) * eps) for j in (1, 2, 3)), alg)
        + common_eta
        - sum_expr(((c0 * gam(j)).scale((0, 1)).scale(gvar(alg, j))
                    * sum_expr((ibil(k).scale(beta(alg, k)) for k in (1, 2, 3) if k != j), alg)
                    for j in (1, 2, 3)), alg)
    )
    check("N=3: fermionization rule maps Eq. (4.3) -> Eq. (4.6)", (fermionize(alg, H3g) - H3f).is_zero)
    check("N=3: Eq. (4.6) H_3 == table row 3", (H3f - H_of(alg, M, 3, xs)).is_zero)

    # Eqs. (4.14)-(4.16): the N=3 bilinear forms, and Table 2
    Aj = [None] + [A_bil(alg, N, j) for j in (1, 2, 3)]
    Vj = [None] + [V(alg, N, j) for j in (1, 2, 3)]
    Dj = [None] + [D(alg, j) for j in (1, 2, 3)]
    d0 = D0(alg)
    b2 = lambda j: beta(alg, j, 2)
    H1b = (d0.scale(eps) + (Aj[1] + Aj[2] + Aj[3]).scale(t)
           + sum_expr((Dj[j].scale(gvar(alg, j)) for j in (1, 2, 3)), alg)
           - (Aj[1] * Aj[2] + Aj[1] * Aj[3] + Aj[2] * Aj[3]).scale(eta))
    H2b = (d0.scale(t)
           + sum_expr((Aj[j].scale(beta(alg, j, -2)) for j in (1, 2, 3)), alg).scale(eps)
           - sum_expr((Vj[j].scale(gvar(alg, j)) for j in (1, 2, 3)), alg)
           - sum_expr(((d0 * Aj[j]) for j in (1, 2, 3)), alg).scale(eta))
    H3b = (
        -(Aj[1] * Aj[2] + Aj[1] * Aj[3] + Aj[2] * Aj[3]).scale(t)
        - sum_expr(((d0 * Aj[j]) for j in (1, 2, 3)), alg).scale(eps)
        + ((Aj[1] * Aj[2] * Aj[3]).scale(2)
           + Aj[1].scale(b2(2) + b2(3)) + Aj[2].scale(b2(1) + b2(3)) + Aj[3].scale(b2(1) + b2(2))
           ).scale(eta)
        - (Dj[1].scale(gvar(alg, 1)) * (Aj[2] + Aj[3])
           + Dj[2].scale(gvar(alg, 2)) * (Aj[1] + Aj[3])
           + Dj[3].scale(gvar(alg, 3)) * (Aj[1] + Aj[2]))
    )
    for mu, Hb in ((1, H1b), (2, H2b), (3, H3b)):
        check(f"N=3: Eq. (4.1{2+mu}) H_{mu} in bilinears == table row {mu}",
              (Hb - H_of(alg, M, mu, xs)).is_zero)

    # Table 2 entries, read literally
    tab2 = {
        (1, 1): Aj[1] + Aj[2] + Aj[3],
        (1, 2): d0,
        (1, 3): minus(Aj[1] * Aj[2] + Aj[1] * Aj[3] + Aj[2] * Aj[3]),
        (1, 0): sum_expr((Dj[j].scale(gvar(alg, j)) for j in (1, 2, 3)), alg),
        (2, 2): sum_expr((Aj[j].scale(beta(alg, j, -2)) for j in (1, 2, 3)), alg),
        (2, 3): minus(sum_expr(((d0 * Aj[j]) for j in (1, 2, 3)), alg)),
        (2, 0): minus(sum_expr((Vj[j].scale(gvar(alg, j)) for j in (1, 2, 3)), alg)),
        (3, 3): ((Aj[1] * Aj[2] * Aj[3]).scale(2) + Aj[1].scale(b2(2) + b2(3))
                 + Aj[2].scale(b2(1) + b2(3)) + Aj[3].scale(b2(1) + b2(2))),
        (3, 0): minus(Dj[1].scale(gvar(alg, 1)) * (Aj[2] + Aj[3])
                      + Dj[2].scale(gvar(alg, 2)) * (Aj[1] + Aj[3])
                      + Dj[3].scale(gvar(alg, 3)) * (Aj[1] + Aj[2])),
    }
    ok = all((tab2[(mu, nu)] - M[mu - 1][(N if nu == 0 else nu - 1)]).is_zero for mu, nu in tab2)
    check("N=3: Table 2 entries == closed-form Table 3", ok)

    # Eq. (4.17): the generating potential Phi_3 vs the general Eq. (3.11)
    half = (__import__("fractions").Fraction(1, 2), __import__("fractions").Fraction(0))
    Phi_general = sum_expr(
        (M[mu - 1][nu - 1].scale(alg.var(xs[mu - 1]) * alg.var(xs[nu - 1])) for mu in (1, 2, 3) for nu in (1, 2, 3)),
        alg).scale(half) + sum_expr((M[mu - 1][3].scale(alg.var(xs[mu - 1])) for mu in (1, 2, 3)), alg)
    Phi_paper = (
        (Aj[1] + Aj[2] + Aj[3]).scale(t * t).scale(half)
        + d0.scale(t * eps)
        - (Aj[1] * Aj[2] + Aj[1] * Aj[3] + Aj[2] * Aj[3]).scale(t * eta)
        + sum_expr((Dj[j].scale(gvar(alg, j)) for j in (1, 2, 3)), alg).scale(t)
        + sum_expr((Aj[j].scale(beta(alg, j, -2)) for j in (1, 2, 3)), alg).scale(eps * eps).scale(half)
        - sum_expr(((d0 * Aj[j]) for j in (1, 2, 3)), alg).scale(eps * eta)
        - sum_expr((Vj[j].scale(gvar(alg, j)) for j in (1, 2, 3)), alg).scale(eps)
        + ((Aj[1] * Aj[2] * Aj[3]).scale(2) + Aj[1].scale(b2(2) + b2(3))
           + Aj[2].scale(b2(1) + b2(3)) + Aj[3].scale(b2(1) + b2(2))).scale(eta * eta).scale(half)
        - (Dj[1].scale(gvar(alg, 1)) * (Aj[2] + Aj[3])
           + Dj[2].scale(gvar(alg, 2)) * (Aj[1] + Aj[3])
           + Dj[3].scale(gvar(alg, 3)) * (Aj[1] + Aj[2])).scale(eta)
    )
    check("N=3: Eq. (4.17) Phi_3 == the general quadratic potential Eq. (3.11)",
          (Phi_paper - Phi_general).is_zero)

    # Appendix C: the explicit N=4 Hamiltonians
    N = 4
    alg, M, xs = paper_table(N, range(1, N + 1))
    t, eps, eta, zeta = (alg.var(x) for x in xs)
    Aj = [None] + [A_bil(alg, N, j) for j in range(1, 5)]
    Vj = [None] + [V(alg, N, j) for j in range(1, 5)]
    Dj = [None] + [D(alg, j) for j in range(1, 5)]
    d0 = D0(alg)
    idx = (1, 2, 3, 4)
    e1 = sum_expr((Aj[j] for j in idx), alg)
    e2 = sum_expr((Aj[i] * Aj[j] for i, j in itertools.combinations(idx, 2)), alg)
    e3 = sum_expr((Aj[i] * Aj[j] * Aj[k] for i, j, k in itertools.combinations(idx, 3)), alg)
    e4 = Aj[1] * Aj[2] * Aj[3] * Aj[4]
    def sumb2(excl):
        parts = [beta(alg, k, 2) for k in idx if k not in excl]
        out = alg.zero_poly()
        for p in parts:
            out = out + p
        return out

    H1_4 = (e1.scale(t) + d0.scale(eps) - e2.scale(eta) + e3.scale(zeta)
            + sum_expr((Dj[j].scale(gvar(alg, j)) for j in idx), alg))
    H2_4 = (d0.scale(t) + sum_expr((Aj[j].scale(beta(alg, j, -2)) for j in idx), alg).scale(eps)
            - (d0 * e1).scale(eta) + (d0 * e2).scale(zeta)
            - sum_expr((Vj[j].scale(gvar(alg, j)) for j in idx), alg))
    H3_4 = (
        -e2.scale(t) - (d0 * e1).scale(eps)
        + (e3.scale(2) + sum_expr((Aj[j].scale(sumb2((j,))) for j in idx), alg)).scale(eta)
        - (e4.scale(3) + sum_expr(((Aj[i] * Aj[j]).scale(sumb2((i, j)))
                                   for i, j in itertools.combinations(idx, 2)), alg)).scale(zeta)
        - sum_expr((Dj[j].scale(gvar(alg, j)) * sum_expr((Aj[k] for k in idx if k != j), alg)
                    for j in idx), alg)
    )
    H4_4 = (
        e3.scale(t) + (d0 * e2).scale(eps)
        - (e4.scale(3) + sum_expr(((Aj[i] * Aj[j]).scale(sumb2((i, j)))
                                   for i, j in itertools.combinations(idx, 2)), alg)).scale(eta)
        + (sum_expr((Aj[j].scale(_e2_beta2(alg, [k for k in idx if k != j])) for j in idx), alg)
           + sum_expr(((Aj[i] * Aj[j] * Aj[k]).scale(
               beta(alg, [l for l in idx if l not in (i, j, k)][0], 2)).scale(2)
               for i, j, k in itertools.combinations(idx, 3)), alg)).scale(zeta)
        + sum_expr((Dj[j].scale(gvar(alg, j))
                    * sum_expr((Aj[k] * Aj[l] for k, l in itertools.combinations(
                        [m for m in idx if m != j], 2)), alg)
                    for j in idx), alg)
    )
    for mu, Hp in ((1, H1_4), (2, H2_4), (3, H3_4), (4, H4_4)):
        check(f"N=4: Appendix C explicit H_{mu} == closed-form table row {mu}",
              (Hp - H_of(alg, M, mu, xs)).is_zero)


def _e2_beta2(alg, universe):
    out = alg.zero_poly()
    for k, l in itertools.combinations(universe, 2):
        out = out + beta(alg, k, 2) * beta(alg, l, 2)
    return out


def _sum_poly(alg, parts):
    out = alg.zero_poly()
    for p in parts:
        out = out + p
    return out


# ---------------------------------------------------------------------------
# Appendix D -- generating function (Eqs. (D.1)-(D.9))
# ---------------------------------------------------------------------------

def check_section5():
    section("Appendix D -- generating function for B_{mu nu}")
    for N in (2, 3, 4):
        # tables extended to N+2 rows so the full-series claims can be tested
        alg, M, xs = paper_table(N, range(1, N + 3), extra=("z", "w", "T", "U", "s"))
        z, w, T, U, s = (alg.var(v) for v in ("z", "w", "T", "U", "s"))

        # Eq. (D.1) and Eq. (D.2)
        P = one(alg)
        for j in range(1, N + 1):
            P = P * (one(alg) + A_bil(alg, N, j).scale(T) + alg.scalar(beta(alg, j, 2) * U))
        E_rs = alg.zero()
        for r in range(0, N + 1):
            for sd in range(0, N + 1 - r):
                for I in itertools.combinations(range(1, N + 1), r):
                    rest = [k for k in range(1, N + 1) if k not in I]
                    for J in itertools.combinations(rest, sd):
                        p = alg.const_poly(1)
                        for j in J:
                            p = p * beta(alg, j, 2)
                        E_rs = E_rs + A_subset(alg, N, I).scale(p).scale(
                            alg.var("T", r) * alg.var("U", sd))
        check(f"N={N}: Eq. (D.2) E_(r,s) are the coefficients of Eq. (D.1) P(T,U)",
              (P - E_rs).is_zero)

        # factorization of each factor, and Eq. (D.4)
        ok = all((one(alg) + A_bil(alg, N, j).scale(z + w) + alg.scalar(beta(alg, j, 2) * z * w)
                  - (one(alg) + A_bil(alg, N, j).scale(z)) * (one(alg) + A_bil(alg, N, j).scale(w))).is_zero
                 for j in range(1, N + 1))
        check(f"N={N}: 1 + A_j(z+w) + beta_j^2 zw == (1+A_j z)(1+A_j w)", ok)
        P_sub = one(alg)
        for j in range(1, N + 1):
            P_sub = P_sub * (one(alg) + A_bil(alg, N, j).scale(z + w) + alg.scalar(beta(alg, j, 2) * z * w))
        check(f"N={N}: Eq. (D.4) P(z+w, zw) == E(z) E(w)",
              (P_sub - _E(alg, N, z) * _E(alg, N, w)).is_zero)

        # kernel G and its coefficients K_{a,b}
        P0 = one(alg)
        for j in range(1, N + 1):
            P0 = P0 * (one(alg) + alg.scalar(beta(alg, j, 2) * z * w))
        R = _E(alg, N, z) * _E(alg, N, w) - P0
        Kq = {}
        for b in range(0, N + 3):
            for a in range(N + 2, -1, -1):
                Kq[(a, b)] = coeff_of(alg, R, {"z": a + 1, "w": b}) - Kq.get((a + 1, b - 1), alg.zero())

        # Eqs. (D.5)/(D.6): the mu,nu >= 3 block, and the mu = 1 row from a = 0
        ok = True
        for mu in range(3, N + 3):
            for nu in range(3, N + 3):
                a, b = mu - 2, nu - 2
                rhs = Kq[(a, b)].scale((-1) ** (mu + nu))
                ok &= (M[mu - 1][nu - 1] - rhs).is_zero
        check(f"N={N}: Eq. (D.6) B_(mu nu) = (-1)^(mu+nu)[z^(mu-2) w^(nu-2)]G, mu,nu>=3", ok)
        ok = all((M[0][nu - 1] - Kq[(0, a_index(nu))].scale((-1) ** a_index(nu))).is_zero
                 for nu in [1] + list(range(3, N + 2)))
        check(f"N={N}: the a=0 coefficients of Eq. (D.6) give row 1", ok)

        # support of the kernel: K_{a,b} != 0 only for max(a,b) <= N-1
        ok = all(Kq[(a, b)].is_zero
                 for a in range(0, N + 3) for b in range(0, N + 3) if max(a, b) > N - 1)
        check(f"N={N}: Theorem 6.6(i) support -- K_(a,b) = 0 unless max(a,b) <= N-1", ok)

        # Eq. (D.7): row 1 series, extended range nu = 3..N+1
        lhs = sum_expr((M[0][nu - 1].scale(alg.var("s", nu - 1)).scale((-1) ** nu)
                        for nu in range(3, N + 2)), alg)
        rhs = _E(alg, N, s) - one(alg) - sum_expr((A_bil(alg, N, j) for j in range(1, N + 1)), alg).scale(s)
        check(f"N={N}: Eq. (D.7) row-1 series over nu=3..N+1", (lhs - rhs).is_zero)
        if N >= 2:      # and it genuinely fails if truncated at nu = N
            trunc = sum_expr((M[0][nu - 1].scale(alg.var("s", nu - 1)).scale((-1) ** nu)
                              for nu in range(3, N + 1)), alg)
            check(f"N={N}: (control) truncating Eq. (D.7) at nu=N does NOT close",
                  not (trunc - rhs).is_zero)

        # Eq. (D.8): row 2 series, extended range nu = 3..N+2
        lhs = sum_expr((M[1][nu - 1].scale(alg.var("s", nu - 2)).scale((-1) ** nu)
                        for nu in range(3, N + 3)), alg)
        rhs = D0(alg) * (_E(alg, N, s) - one(alg))
        check(f"N={N}: Eq. (D.8) row-2 series over nu=3..N+2", (lhs - rhs).is_zero)

        # Eq. (D.9): constant column over the extended index set I' = {1,3,...,N+1}
        lhs = sum_expr((M[mu - 1][N + 2].scale(alg.var("s", a_index(mu))).scale((-1) ** a_index(mu))
                        for mu in [1] + list(range(3, N + 2))), alg)
        rhs = alg.zero()
        for j in range(1, N + 1):
            Ej = one(alg)
            for i in range(1, N + 1):
                if i != j:
                    Ej = Ej * (one(alg) + A_bil(alg, N, i).scale(s))
            rhs = rhs + term(gvar(alg, j), D(alg, j), Ej, alg=alg)
        check(f"N={N}: Eq. (D.9) constant-column series over I' = {{1,3,..,N+1}}", (lhs - rhs).is_zero)

        # the three named top coefficients quoted in the text after (D.7)-(D.9)
        X = e_r(alg, N, N) * D0(alg)
        check(f"N={N}: B_(1,N+1) == (-1)^(N-1) e_N(A)",
              (M[0][N] - e_r(alg, N, N).scale((-1) ** (N - 1))).is_zero)
        check(f"N={N}: B_(2,N+2) == (-1)^N X",
              (M[1][N + 1] - X.scale((-1) ** N)).is_zero)
        Ahat = lambda j: A_subset(alg, N, [k for k in range(1, N + 1) if k != j])
        check(f"N={N}: O_(N+1) == (-1)^(N-1) sum_j g_j D_j A_hat(j)",
              (M[N][N + 2] - sum_expr((term(gvar(alg, j), D(alg, j), Ahat(j), alg=alg)
                                       for j in range(1, N + 1)), alg).scale((-1) ** (N - 1))).is_zero)


# ---------------------------------------------------------------------------
# Section 5 -- symmetric-polynomial structure
# ---------------------------------------------------------------------------

def check_section6():
    section("Section 5 -- symmetric-polynomial structure")
    for N in (2, 3, 4):
        alg, M, xs = paper_table(N, range(1, N + 1), extra=("z", "s"))
        z, s = alg.var("z"), alg.var("s")
        script_I = [m for m in range(1, N + 1) if m != 2]

        # Section 5.1: the abelian core
        entries = [M[mu - 1][nu - 1] for mu in range(1, N + 1) for nu in range(1, N + 1)]
        check(f"N={N}: Section 5.1 all Hessian entries commute pairwise",
              all(p.commutator(q).is_zero for p in entries for q in entries))

        # row 1, row 2, constant column in the e-basis
        check(f"N={N}: B_(1 nu) == (-1)^(a_nu) e_(a_nu+1)(A)",
              all((M[0][nu - 1] - e_r(alg, N, a_index(nu) + 1).scale((-1) ** a_index(nu))).is_zero
                  for nu in script_I))
        named = [(1, 1, 1), (3, 2, -1), (4, 3, 1)]
        check(f"N={N}: B_11=e_1, B_13=-e_2, B_14=e_3 as printed",
              all((M[0][nu - 1] - e_r(alg, N, r).scale(sg)).is_zero
                  for nu, r, sg in named if nu <= N))
        check(f"N={N}: B_(1N) == (-1)^N e_(N-1)",
              N < 3 or (M[0][N - 1] - e_r(alg, N, N - 1).scale((-1) ** N)).is_zero)

        # Eq. (5.5)
        e1 = e_r(alg, N, 1)
        check(f"N={N}: Eq. (5.5) e_1^2 == e_1(beta^2) + 2 e_2",
              (e1 * e1 - alg.scalar(e_beta2(alg, 1, range(1, N + 1)))
               - e_r(alg, N, 2).scale(2)).is_zero)

        # Proposition 5.1, Eq. (5.6): linearization of e_a e_b
        ok = True
        for a in range(0, N + 2):
            for b in range(0, N + 2):
                lhs = e_r(alg, N, a) * e_r(alg, N, b)
                rhs = alg.zero()
                for sd in range(0, min(a, b) + 1):
                    c = comb(a + b - 2 * sd, a - sd) if 0 <= a - sd <= a + b - 2 * sd else 0
                    if c == 0:
                        continue
                    d = a + b - 2 * sd
                    if d < 0 or d > N:
                        continue
                    for I in itertools.combinations(range(1, N + 1), d):
                        compl = [k for k in range(1, N + 1) if k not in I]
                        es = e_beta2(alg, sd, compl)
                        if es.is_zero:
                            continue
                        rhs = rhs + A_subset(alg, N, I).scale(es).scale(c)
                ok &= (lhs - rhs).is_zero
        check(f"N={N}: Proposition 5.1 / Eq. (5.6) linearization of e_a e_b", ok)

        # Eq. (5.7): the three-term recursion
        ok = True
        for a in range(0, N + 2):
            for b in range(0, N + 2):
                lhs = (K_paper(alg, N, a - 1, b) if a >= 1 else alg.zero()) + \
                      (K_paper(alg, N, a, b - 1) if b >= 1 else alg.zero())
                rhs = e_r(alg, N, a) * e_r(alg, N, b)
                if a == b:
                    rhs = rhs - alg.scalar(e_beta2(alg, a, range(1, N + 1)))
                ok &= (lhs - rhs).is_zero
        check(f"N={N}: Eq. (5.7) K_(a-1,b) + K_(a,b-1) = e_a e_b - delta_ab e_a(beta^2)", ok)

        # Boundary conventions. The recursion is stated for all a,b >= 0, so it
        # silently relies on K_(-1,b) = K_(a,-1) = 0. Pin both, and pin the
        # N=2 edge case where index 2 is the exceptional row and 1N collides
        # with the row-2 entry.
        ok_bd = all(K_paper(alg, N, -1, b).is_zero for b in range(0, N + 2)) and \
                all(K_paper(alg, N, a, -1).is_zero for a in range(0, N + 2))
        check(f"N={N}: boundary convention K_(-1,b) == K_(a,-1) == 0", ok_bd)

        # first column is e_{a+1}, which must vanish once a+1 > N
        ok_top = all(K_paper(alg, N, a, 0).is_zero for a in range(N, N + 3))
        check(f"N={N}: K_(a,0) == e_(a+1) vanishes for a >= N", ok_top)

        # Eq. (E.1): the kernel identity
        G = alg.zero()
        for a in range(0, N + 1):
            for b in range(0, N + 1):
                k = K_paper(alg, N, a, b)
                if not k.is_zero:
                    G = G + k.scale(alg.var("z", a) * alg.var("s", b))
        P0 = one(alg)
        for j in range(1, N + 1):
            P0 = P0 * (one(alg) + alg.scalar(beta(alg, j, 2) * z * s))
        check(f"N={N}: Eq. (E.1) (z+w) sum K z^a w^b == E(z)E(w) - prod(1+beta_j^2 zw)",
              (G.scale(z + s) - (_E(alg, N, z) * _E(alg, N, s) - P0)).is_zero)

        # the worked example after Eq. (E.1)
        check(f"N={N}: K_01 == K_10 == e_2",
              (K_paper(alg, N, 0, 1) - e_r(alg, N, 2)).is_zero
              and (K_paper(alg, N, 1, 0) - e_r(alg, N, 2)).is_zero)

        # Eq. (5.8): the reflection identity
        prod = one(alg)
        for j in range(1, N + 1):
            prod = prod * (one(alg) - alg.scalar(beta(alg, j, 2) * z * z))
        check(f"N={N}: Eq. (5.8) E(z)E(-z) == prod(1 - beta_j^2 z^2)",
              (_E(alg, N, z) * _E(alg, N, alg.const_poly(0) - z) - prod).is_zero)

    # Krawtchouk evaluation on the isotropic hypercube.  The identity is no
    # longer printed in the paper (the named-polynomial discussion was cut),
    # but the check is kept: it guards the hypercube spectrum, Eq. (5.3).
    ok = True
    for N in range(1, 8):
        for k in range(0, N + 1):
            sigma = [-1] * k + [1] * (N - k)
            for r in range(0, N + 1):
                direct = sum(
                    _prod([sigma[i] for i in I]) for I in itertools.combinations(range(N), r)
                )
                krawt = sum((-1) ** i * comb(k, i) * comb(N - k, r - i) for i in range(0, r + 1))
                ok &= direct == krawt
    check("e_r on a k-minus sign vector == binary Krawtchouk K_r(k;N) "
          "[identity no longer printed in the paper; it guards the "
          "hypercube spectrum, Eq. (5.3)]", ok)


def _prod(xs):
    out = 1
    for x in xs:
        out *= x
    return out


# ---------------------------------------------------------------------------
# Section 6 -- (N+1)-st flow, parity duality, termination
# ---------------------------------------------------------------------------

def check_section7():
    section("Section 6 -- (N+1)-st flow, parity duality, maximality")
    for N in (2, 3, 4):
        top = N + 2
        alg, M, xs = paper_table(N, range(1, top + 1))
        tau = [alg.var(x) for x in xs]
        X = e_r(alg, N, N) * D0(alg)
        prod_b2 = alg.const_poly(1)
        for j in range(1, N + 1):
            prod_b2 = prod_b2 * beta(alg, j, 2)
        Ahat = lambda j: A_subset(alg, N, [k for k in range(1, N + 1) if k != j])

        # Lemma 7.3
        check(f"N={N}: Lemma 7.3 X^2 == prod beta_j^2", (X * X - alg.scalar(prod_b2)).is_zero)
        check(f"N={N}: Lemma 7.3 X commutes with every table entry",
              all(X.commutator(e).is_zero for row in M for e in row))
        # Eqs. (F.1)-(F.2): Gamma = c_0 gamma_0 gamma_1...gamma_2N, Gamma^2 = (-1)^(N+1),
        # and X = kappa_N (prod beta_j) Gamma with kappa_N = 1 (N odd) / i (N even).
        # v1 printed "X = +/- (prod beta_j) Gamma", which is wrong for even N:
        # the discrepancy is a phase i, not a sign (erratum V8).
        Gamma = alg.product([-1, 0] + list(range(1, 2 * N + 1)))
        prod_b = alg.const_poly(1)
        for j in range(1, N + 1):
            prod_b = prod_b * beta(alg, j)
        kappa = (1, 0) if N % 2 else (0, 1)
        check(f"N={N}: Eq. (F.1) Gamma^2 == (-1)^(N+1)",
              (Gamma * Gamma - one(alg).scale((-1) ** (N + 1))).is_zero)
        check(f"N={N}: Eq. (F.2) X == kappa_N (prod beta_j) Gamma, kappa_N = "
              f"{'1' if N % 2 else 'i'}",
              (X - Gamma.scale(prod_b).scale(kappa)).is_zero)
        check(f"N={N}: (control) the v1 form X = +/- (prod beta_j) Gamma "
              f"{'holds' if N % 2 else 'does not hold'} at this N",
              ((X - Gamma.scale(prod_b)).is_zero or (X + Gamma.scale(prod_b)).is_zero)
              == bool(N % 2))

        # Eq. (6.1): the explicit (N+1)-st member
        HN1 = (
            e_r(alg, N, N).scale(tau[0])
            + (D0(alg) * e_r(alg, N, N - 1)).scale(tau[1])
            + sum_expr((
                sum_expr((A_subset(alg, N, I).scale(e_beta2(alg, nu - 2, [k for k in range(1, N + 1) if k not in I]))
                          for I in itertools.combinations(range(1, N + 1), N + 2 - nu)), alg)
                .scale(tau[nu - 1]).scale((-1) ** nu)
                for nu in range(3, N + 2)), alg)
            + sum_expr((term(gvar(alg, j), D(alg, j), Ahat(j), alg=alg) for j in range(1, N + 1)), alg)
        ).scale((-1) ** (N - 1))
        # H_{N+1} from the table, with the level-(N+2) time switched off
        HN1_table = H_of(alg, M, N + 1, xs) - M[N][N + 1].scale(tau[N + 1])
        check(f"N={N}: Eq. (6.1) explicit H_(N+1) == table row N+1", (HN1 - HN1_table).is_zero)

        # Theorem 7.2 proof: the t-column is e_1, D_0, -e_2, e_3, ..., (-1)^(N+1) e_N
        col = [M[mu - 1][0] for mu in range(1, N + 2)]
        ok = (col[0] - e_r(alg, N, 1)).is_zero and (col[1] - D0(alg)).is_zero
        for mu in range(3, N + 2):
            ok &= (col[mu - 1] - e_r(alg, N, mu - 1).scale((-1) ** mu)).is_zero
        check(f"N={N}: Conjecture 6.1 t-column == e_1, D_0, -e_2, e_3, ..., (-1)^(N+1) e_N", ok)
        check(f"N={N}: Conjecture 6.1 last t-column entry sign is (-1)^(N+1) (errata item V4)",
              (M[N][0] - e_r(alg, N, N).scale((-1) ** (N + 1))).is_zero)

        # Theorem 7.4: the column-by-column duality (nu = 1..N+1 and the constant column)
        sgn = (-1) ** (N - 1)
        ok = all((M[N][nu - 1] - (X * M[1][nu - 1]).scale(sgn)).is_zero for nu in range(1, N + 2))
        check(f"N={N}: Theorem 6.3 B_(N+1,nu) = (-1)^(N-1) X B_(2 nu), nu=1..N+1", ok)
        check(f"N={N}: Theorem 6.3 O_(N+1) = (-1)^(N-1) X O_2",
              (M[N][top] - (X * M[1][top]).scale(sgn)).is_zero)
        # the auxiliary identities used in the proof
        check(f"N={N}: proof (a) A_hat(j) A_j == e_N(A), i.e. A_hat(j) = e_N A_j / beta_j^2",
              all((Ahat(j) * A_bil(alg, N, j) - e_r(alg, N, N)).is_zero for j in range(1, N + 1)))
        check(f"N={N}: proof (b) X B_22 == D_0 e_(N-1)(A)",
              (X * M[1][1] - D0(alg) * e_r(alg, N, N - 1)).is_zero)
        # proof (d): D_0 V_j = c_0 gamma_(N+j)/beta_j ; A_j c_0 gamma_(N+j) = -beta_j D_j
        ok = all((D0(alg) * V(alg, N, j)
                  - (alg.gen(-1) * alg.gen(N + j)).scale(beta(alg, j, -1))).is_zero
                 for j in range(1, N + 1))
        check(f"N={N}: proof (d) D_0 V_j == c_0 gamma_(N+j) / beta_j", ok)
        ok = all((A_bil(alg, N, j) * (alg.gen(-1) * alg.gen(N + j))
                  + D(alg, j).scale(beta(alg, j))).is_zero for j in range(1, N + 1))
        check(f"N={N}: proof (d) A_j c_0 gamma_(N+j) == -beta_j D_j", ok)
        ok = all((e_r(alg, N, N) * D0(alg) * V(alg, N, j) + D(alg, j) * Ahat(j)).is_zero
                 for j in range(1, N + 1))
        check(f"N={N}: proof (d) e_N D_0 V_j == -D_j A_hat(j)", ok)

        # Remark 7.6: the level-(N+2) time
        lhs = H_of(alg, M, N + 1, xs)
        rhs = (X * (H_of(alg, M, 2, xs) - (X.scale(tau[N + 1])).scale((-1) ** N))).scale(sgn)
        check(f"N={N}: Remark 7.6 H_(N+1) = (-1)^(N-1) X (H_2 - (-1)^N tau_(N+2) X)",
              (lhs - rhs).is_zero)

        # Theorem 7.5 -- termination
        alg2, M2, xs2 = paper_table(N, range(1, N + 5))
        X2 = e_r(alg2, N, N) * D0(alg2)
        big = [m for m in range(1, N + 5) if m != 2 and m >= N + 2]
        ok = all(M2[mu - 1][nu - 1].is_zero
                 for mu in [m for m in range(1, N + 5) if m != 2]
                 for nu in [m for m in range(1, N + 5) if m != 2]
                 if max(mu, nu) >= N + 2)
        check(f"N={N}: Theorem 6.6(i) block entries vanish for max(mu,nu) >= N+2", ok)
        check(f"N={N}: Theorem 6.6(ii) first and constant columns vanish for mu >= N+2",
              all(M2[mu - 1][0].is_zero and M2[mu - 1][N + 4].is_zero for mu in big))
        row = M2[N + 1]
        nz = [(c, e) for c, e in enumerate(row) if not e.is_zero]
        check(f"N={N}: Theorem 6.6(iii) row N+2 has the single entry B_(N+2,2) = (-1)^N X",
              len(nz) == 1 and nz[0][0] == 1 and (nz[0][1] - X2.scale((-1) ** N)).is_zero)
        check(f"N={N}: Theorem 6.6(iii) rows mu >= N+3 vanish identically",
              all(e.is_zero for mu in range(N + 3, N + 5) for e in M2[mu - 1]))
        check(f"N={N}: Theorem 6.6(iii) H_(N+2) == (-1)^N eps X",
              (H_of(alg2, M2, N + 2, xs2)
               - X2.scale(alg2.var(xs2[1])).scale((-1) ** N)).is_zero)

        # Remark 7.10: B_22 and O_2 as X-duals of the top row
        check(f"N={N}: Remark 7.10 B_22 = (-1)^(N-1) X B_(N+1,2) / prod beta_j^2",
              (M[1][1].scale(prod_b2) - (X * M[N][1]).scale((-1) ** (N - 1))).is_zero)
        check(f"N={N}: Remark 7.10 O_2 = (-1)^(N-1) X O_(N+1) / prod beta_j^2",
              (M[1][top].scale(prod_b2) - (X * M[N][top]).scale((-1) ** (N - 1))).is_zero)

    # Eqs. (F.1)-(F.2) over the whole range claimed in the paper.  This needs no
    # table, so it is cheap enough to run well past N=4.
    ok_g, ok_k, ok_v1 = True, True, True
    for N in range(1, 7):
        alg = majorana_algebra(N)
        X = e_r(alg, N, N) * D0(alg)
        Gamma = alg.product([-1, 0] + list(range(1, 2 * N + 1)))
        prod_b = alg.const_poly(1)
        for j in range(1, N + 1):
            prod_b = prod_b * beta(alg, j)
        ok_g &= (Gamma * Gamma - one(alg).scale((-1) ** (N + 1))).is_zero
        ok_k &= (X - Gamma.scale(prod_b).scale((1, 0) if N % 2 else (0, 1))).is_zero
        v1_holds = (X - Gamma.scale(prod_b)).is_zero or (X + Gamma.scale(prod_b)).is_zero
        ok_v1 &= (v1_holds == bool(N % 2))
    check("N=1..6: Eq. (F.1) Gamma^2 == (-1)^(N+1)", ok_g)
    check("N=1..6: Eq. (F.2) X == kappa_N (prod beta_j) Gamma, kappa_N = 1/i for N odd/even", ok_k)
    check("N=1..6: (control) the v1 form X = +/- (prod beta_j) Gamma holds iff N is odd", ok_v1)

    # Eq. (6.2): the explicit N=3 fourth member
    N = 3
    alg, M, xs = paper_table(N, range(1, N + 2))
    t, eps, eta, zetap = (alg.var(x) for x in xs)
    Aj = [None] + [A_bil(alg, N, j) for j in (1, 2, 3)]
    Dj = [None] + [D(alg, j) for j in (1, 2, 3)]
    d0 = D0(alg)
    other = lambda pair: [m for m in (1, 2, 3) if m not in pair]
    H4 = (
        (Aj[1] * Aj[2] * Aj[3]).scale(t)
        + (d0 * (Aj[1] * Aj[2] + Aj[1] * Aj[3] + Aj[2] * Aj[3])).scale(eps)
        - sum_expr(((Aj[i] * Aj[j]).scale(beta(alg, other((i, j))[0], 2))
                    for i, j in itertools.combinations((1, 2, 3), 2)), alg).scale(eta)
        + sum_expr((Aj[j].scale(_prod_poly(alg, [beta(alg, k, 2) for k in (1, 2, 3) if k != j]))
                    for j in (1, 2, 3)), alg).scale(zetap)
        + sum_expr((term(gvar(alg, j), Dj[j],
                         A_subset(alg, N, [k for k in (1, 2, 3) if k != j]), alg=alg)
                    for j in (1, 2, 3)), alg)
    )
    check("N=3: Eq. (6.2) explicit H_4 == table row 4", (H4 - H_of(alg, M, 4, xs)).is_zero)

    # Appendix F.1: the printed N=2 extended table (F.3) and its duality lines
    N = 2
    alg, M, xs = paper_table(N, range(1, 5))
    A1, A2 = A_bil(alg, N, 1), A_bil(alg, N, 2)
    V1, V2 = V(alg, N, 1), V(alg, N, 2)
    D1, D2 = D(alg, 1), D(alg, 2)
    d0 = D0(alg)
    g1, g2 = gvar(alg, 1), gvar(alg, 2)
    b1s, b2s = beta(alg, 1, 2), beta(alg, 2, 2)
    Xe = A1 * A2 * d0
    printed = {
        (1, 1): A1 + A2, (1, 2): d0, (1, 3): minus(A1 * A2), (1, 4): alg.zero(),
        (1, 0): D1.scale(g1) + D2.scale(g2),
        (2, 1): d0, (2, 2): A1.scale(beta(alg, 1, -2)) + A2.scale(beta(alg, 2, -2)),
        (2, 3): minus(d0 * (A1 + A2)), (2, 4): Xe,
        (2, 0): minus(V1.scale(g1) + V2.scale(g2)),
        (3, 1): minus(A1 * A2), (3, 2): minus(d0 * (A1 + A2)),
        (3, 3): A1.scale(b2s) + A2.scale(b1s), (3, 4): alg.zero(),
        (3, 0): minus((D1 * A2).scale(g1) + (D2 * A1).scale(g2)),
        (4, 1): alg.zero(), (4, 2): Xe, (4, 3): alg.zero(), (4, 4): alg.zero(),
        (4, 0): alg.zero(),
    }
    ok = all((printed[(mu, nu)] - M[mu - 1][(4 if nu == 0 else nu - 1)]).is_zero for mu, nu in printed)
    check("N=2: printed extended table Eq. (F.3) == closed form", ok)
    check("N=2: Eq. (F.4) X = A_1 A_2 D_0 with X^2 = beta_1^2 beta_2^2",
          (Xe * Xe - alg.scalar(b1s * b2s)).is_zero)
    check("N=2: worked B_33 == beta_2^2 A_1 + beta_1^2 A_2",
          (M[2][2] - (A1.scale(b2s) + A2.scale(b1s))).is_zero)
    check("N=2: worked B_24 == D_0 e_2(A) == X", (M[1][3] - Xe).is_zero)
    ok = all((M[2][nu - 1] + Xe * M[1][nu - 1]).is_zero for nu in (1, 2, 3)) and \
        (M[2][4] + Xe * M[1][4]).is_zero
    check("N=2: Appendix F.1 column-by-column duality B_(3 nu) = -X B_(2 nu)", ok)
    H3 = H_of(alg, M, 3, xs)
    H2 = H_of(alg, M, 2, xs) - M[1][3].scale(alg.var(xs[3]))
    check("N=2: Appendix F.1 H_3 = -X H_2 (with tau_4 = 0)",
          (H3 - minus(Xe * H2)).is_zero)
    check("N=2: Appendix F.1 H_4 = eps X", (H_of(alg, M, 4, xs) - Xe.scale(alg.var(xs[1]))).is_zero)


def _prod_poly(alg, parts):
    out = alg.const_poly(1)
    for p in parts:
        out = out * p
    return out


# ---------------------------------------------------------------------------
# Exact complex-rational matrices, for the representation-level claims
# ---------------------------------------------------------------------------

from fractions import Fraction  # noqa: E402


class Mat:
    """Square matrix over the complex rationals (exact)."""

    __slots__ = ("n", "a")

    def __init__(self, n, a=None):
        self.n = n
        self.a = a if a is not None else [[(Fraction(0), Fraction(0))] * n for _ in range(n)]

    @staticmethod
    def eye(n):
        m = Mat(n)
        m.a = [[(Fraction(1), Fraction(0)) if i == j else (Fraction(0), Fraction(0))
                for j in range(n)] for i in range(n)]
        return m

    def __add__(self, o):
        r = Mat(self.n)
        r.a = [[(x[0] + y[0], x[1] + y[1]) for x, y in zip(ra, rb)] for ra, rb in zip(self.a, o.a)]
        return r

    def __sub__(self, o):
        r = Mat(self.n)
        r.a = [[(x[0] - y[0], x[1] - y[1]) for x, y in zip(ra, rb)] for ra, rb in zip(self.a, o.a)]
        return r

    def __mul__(self, o):
        n = self.n
        r = Mat(n)
        for i in range(n):
            row = self.a[i]
            out = r.a[i]
            for k in range(n):
                c = row[k]
                if c[0] == 0 and c[1] == 0:
                    continue
                ok_ = o.a[k]
                for j in range(n):
                    d = ok_[j]
                    out[j] = (out[j][0] + c[0] * d[0] - c[1] * d[1],
                              out[j][1] + c[0] * d[1] + c[1] * d[0])
        return r

    def scale(self, c):
        if not isinstance(c, tuple):
            c = (Fraction(c), Fraction(0))
        r = Mat(self.n)
        r.a = [[(x[0] * c[0] - x[1] * c[1], x[0] * c[1] + x[1] * c[0]) for x in row] for row in self.a]
        return r

    @property
    def is_zero(self):
        return all(x == (0, 0) or (x[0] == 0 and x[1] == 0) for row in self.a for x in row)

    def comm(self, o):
        return self * o - o * self


def kron(A, B):
    n = A.n * B.n
    r = Mat(n)
    for i in range(A.n):
        for j in range(A.n):
            c = A.a[i][j]
            if c[0] == 0 and c[1] == 0:
                continue
            for k in range(B.n):
                for l in range(B.n):
                    d = B.a[k][l]
                    r.a[i * B.n + k][j * B.n + l] = (
                        c[0] * d[0] - c[1] * d[1], c[0] * d[1] + c[1] * d[0])
    return r


F = lambda re, im=0: (Fraction(re), Fraction(im))
_SX = Mat(2, [[F(0), F(1)], [F(1), F(0)]])
_SY = Mat(2, [[F(0), (Fraction(0), Fraction(-1))], [(Fraction(0), Fraction(1)), F(0)]])
_SZ = Mat(2, [[F(1), F(0)], [F(0), F(-1)]])
_I2 = Mat.eye(2)


def site_op(N, j, P):
    """P acting on site j (1-based) of an N-qubit chain."""
    out = Mat.eye(1)
    out.a = [[F(1)]]
    for k in range(1, N + 1):
        out = kron(out, P if k == j else _I2)
    return out


def check_sector_signs():
    """Section 2.4: the reduction to spins is sector dependent.

    On the sector where the normalized central involution takes the value
    Xhat = s, the fermionic generators reduce as
        A_j -> beta_j sigma^z_j,   D_0 -> s prod_k sigma^z_k,   D_j -> s gamma_j,
    so the epsilon and g_j terms of H_1 reverse between the two sectors while
    the t-dependent and density terms do not. The printed spin Hamiltonian is
    the s=+1 sector. This is checked here on the full 2^(N+1)-dimensional
    fermionic representation, not asserted.
    """
    section("Section 2.4 -- the two central-character sectors carry different signs")
    for N in (2, 3):
        nq = N + 1
        n = 2 ** nq

        def maj(k):
            i = (k + 1) // 2
            op = _SX if k % 2 == 1 else _SY
            out = Mat.eye(1); out.a = [[F(1)]]
            for q in range(1, nq + 1):
                out = kron(out, _SZ if q < i else (op if q == i else _I2))
            return out

        g = [None] + [maj(k) for k in range(1, 2 * nq + 1)]
        gam = {j: g[j] for j in range(1, N + 1)}
        gamN = {j: g[N + j] for j in range(1, N + 1)}
        gamma0, c0 = g[2 * N + 1], g[2 * N + 2]
        I = Mat.eye(n)
        Ah = {j: (gamN[j] * gam[j]).scale((Fraction(0), Fraction(1)))
              for j in range(1, N + 1)}                      # A_j / beta_j
        D0 = (c0 * gamma0).scale((Fraction(0), Fraction(1)))
        D = {j: (c0 * gam[j]).scale((Fraction(0), Fraction(1)))
             for j in range(1, N + 1)}
        prodA = Ah[1]
        for j in range(2, N + 1):
            prodA = prodA * Ah[j]
        Xh = prodA * D0

        check(f"N={N}: Xhat^2 == 1 on the doubled space", (Xh * Xh - I).is_zero)
        ok_sec, ok_dj, ok_free = True, True, True
        for s in (1, -1):
            P = (I + Xh.scale((Fraction(s), Fraction(0)))).scale(
                (Fraction(1, 2), Fraction(0)))
            # D_0 -> s prod_j (A_j/beta_j)
            ok_sec &= (P * D0 - P * prodA.scale((Fraction(s), Fraction(0)))).is_zero
            # D_j -> s gamma_j-image: D_j = s * (D_0-stripped) partner, i.e.
            # D_j and D_0 carry the SAME sector sign, so D_0 D_j is sign free
            for j in range(1, N + 1):
                ok_dj &= (P * (D0 * D[j]) -
                          P * (prodA * D[j]).scale((Fraction(s), Fraction(0)))).is_zero
                # A_j is sector independent: it commutes with Xhat and needs no s
                ok_free &= (Xh * Ah[j] - Ah[j] * Xh).is_zero
        check(f"N={N}: on Xhat=s, D_0 == s * prod_j(A_j/beta_j) for BOTH s", ok_sec)
        check(f"N={N}: the D_j images carry the same sector sign s", ok_dj)
        check(f"N={N}: the A_j (t-dependent and density terms) are sector free",
              ok_free)


def check_spin_representation():
    section("Section 2.4 / Remark 6.5 -- the spin representation (exact 2^N matrices)")
    for N in (2, 3, 4):
        sx = [None] + [site_op(N, j, _SX) for j in range(1, N + 1)]
        sy = [None] + [site_op(N, j, _SY) for j in range(1, N + 1)]
        sz = [None] + [site_op(N, j, _SZ) for j in range(1, N + 1)]
        ID = Mat.eye(2 ** N)

        def string(j):                     # prod_{k<j} sigma^z_k
            out = ID
            for k in range(1, j):
                out = out * sz[k]
            return out

        gam = [None] * (2 * N + 1)
        for j in range(1, N + 1):          # Eq. (2.16)
            gam[j] = string(j) * sx[j]
            gam[N + j] = string(j) * sy[j]
        g0 = ID                            # Eq. (2.17): gamma_0 = prod sigma^z_k
        for k in range(1, N + 1):
            g0 = g0 * sz[k]

        gens = [gam[i] for i in range(1, 2 * N + 1)] + [g0]
        ok = all((gens[i] * gens[j] + gens[j] * gens[i]
                  - (ID.scale(2) if i == j else Mat(2 ** N))).is_zero
                 for i in range(len(gens)) for j in range(len(gens)))
        check(f"N={N}: Eqs. (2.15)-(2.17) the 2N+1 spin gammas obey the Clifford relations", ok)
        check(f"N={N}: Eq. (2.17) gamma_0^2 = 1 and {{gamma_0, gamma_i}} = 0",
              (g0 * g0 - ID).is_zero and
              all((g0 * gam[i] + gam[i] * g0).is_zero for i in range(1, 2 * N + 1)))
        ok = all(((gam[N + j] * gam[j]).scale((Fraction(0), Fraction(1))) - sz[j]).is_zero
                 for j in range(1, N + 1))
        check(f"N={N}: Eq. (2.16) i gamma_(N+j) gamma_j == sigma^z_j (strings cancel)", ok)

        # Remark 6.5 / Eq. (6.6): A_1...A_N = (prod beta_j) gamma_0 in the spin rep
        prod_sz = ID
        for k in range(1, N + 1):
            prod_sz = prod_sz * sz[k]
        check(f"N={N}: Remark 6.5 A_1...A_N == (prod beta_j) gamma_0 (constant exactly 1)",
              (prod_sz - g0).is_zero)

    # Eq. (2.18): the spin form of H_1 as the image of the fermionic H_1
    for N in (2, 3):
        alg, M, xs = paper_table(N, range(1, N + 1))
        sx = [None] + [site_op(N, j, _SX) for j in range(1, N + 1)]
        sy = [None] + [site_op(N, j, _SY) for j in range(1, N + 1)]
        sz = [None] + [site_op(N, j, _SZ) for j in range(1, N + 1)]
        ID = Mat.eye(2 ** N)

        def string(j):
            out = ID
            for k in range(1, j):
                out = out * sz[k]
            return out

        spin_gen = {}
        for j in range(1, N + 1):
            spin_gen[j] = string(j) * sx[j]
            spin_gen[N + j] = string(j) * sy[j]
        g0 = ID
        for k in range(1, N + 1):
            g0 = g0 * sz[k]
        spin_gen[0] = g0

        # random-but-exact rational values for the parameters
        vals = {}
        for j in range(1, N + 1):
            vals[f"beta{j}"] = Fraction(2 * j + 1, 3)
            vals[f"g{j}"] = Fraction(j + 2, 5)
        vals[xs[0]] = Fraction(3, 7)      # t
        vals[xs[1]] = Fraction(-5, 4)     # eps
        if N >= 3:
            vals[xs[2]] = Fraction(2, 9)  # eta

        def rho(expr):
            """Inverse fermionization: i c_0 M -> M, then the spin rep of the gammas."""
            out = Mat(2 ** N)
            c0_bit = 1 << alg.label_to_bit[-1]
            for mask, poly in expr.terms.items():
                coeff = _evaluate(alg, poly, vals)
                if coeff == (0, 0):
                    continue
                labels = [lab for bit, lab in enumerate(alg.labels) if mask & (1 << bit)]
                if mask & c0_bit:
                    # move c_0 (label -1, bit 0) to the front: it is already first
                    labels = labels[1:]
                    coeff = (coeff[1], -coeff[0])     # multiply by -i
                m = ID
                for lab in labels:
                    m = m * spin_gen[lab]
                out = out + m.scale(coeff)
            return out

        b = lambda j: (vals[f"beta{j}"], Fraction(0))
        gg = lambda j: (vals[f"g{j}"], Fraction(0))
        t, eps = (vals[xs[0]], Fraction(0)), (vals[xs[1]], Fraction(0))
        H1_spin = g0.scale(eps)
        for j in range(1, N + 1):
            H1_spin = H1_spin + sz[j].scale((b(j)[0] * t[0], Fraction(0))) + spin_gen[j].scale(gg(j))
        if N >= 3:
            eta = vals[xs[2]]
            for i, j in itertools.combinations(range(1, N + 1), 2):
                H1_spin = H1_spin - (sz[i] * sz[j]).scale((b(i)[0] * b(j)[0] * eta, Fraction(0)))
        check(f"N={N}: Eq. (2.18) H_1^spin == the spin image of the fermionic H_1",
              (rho(H_of(alg, M, 1, xs)) - H1_spin).is_zero)


def _evaluate(alg, poly, vals):
    """Evaluate a LaurentPoly at exact rational values."""
    total = (Fraction(0), Fraction(0))
    for exp, c in poly.terms.items():
        v = Fraction(1)
        for name, p in zip(alg.variables, exp):
            if p:
                v *= vals[name] ** p
        total = (total[0] + c[0] * v, total[1] + c[1] * v)
    return total


def check_section8_gaudin():
    section("Section 7 -- Gaudin comparison (Eqs. (7.1)-(7.5))")
    for N in (3, 4):
        n = 2 ** N
        sx = [None] + [site_op(N, j, _SX) for j in range(1, N + 1)]
        sy = [None] + [site_op(N, j, _SY) for j in range(1, N + 1)]
        sz = [None] + [site_op(N, j, _SZ) for j in range(1, N + 1)]
        dot = lambda i, j: sx[i] * sx[j] + sy[i] * sy[j] + sz[i] * sz[j]
        taus = [None] + [Fraction(v) for v in (1, 5, -2, 7, 11)[:N]]

        H = [None] + [
            _msum(n, [dot(k, j).scale((Fraction(1) / (taus[k] - taus[j]), Fraction(0)))
                      for j in range(1, N + 1) if j != k])
            for k in range(1, N + 1)
        ]
        check(f"N={N}: Eq. (7.1) Gaudin family commutes, [H^G_k, H^G_l] = 0",
              all(H[k].comm(H[l]).is_zero for k in range(1, N + 1) for l in range(1, N + 1)))

        # Eq. (7.3): the su(2) commutator identity, on distinct sites a,b,c
        ok = True
        for a, b, c in itertools.permutations(range(1, N + 1), 3):
            lhs = dot(a, b).comm(dot(b, c))
            cross = _msum(n, [
                (sy[a] * sz[c] - sz[a] * sy[c]) * sx[b],
                (sz[a] * sx[c] - sx[a] * sz[c]) * sy[b],
                (sx[a] * sy[c] - sy[a] * sx[c]) * sz[b],
            ])
            ok &= (lhs - cross.scale((Fraction(0), Fraction(2)))).is_zero
        check(f"N={N}: Eq. (7.3) [s_a.s_b, s_b.s_c] == 2i (s_a x s_c).s_b", ok)

    # Eq. (7.4): the three-point partial-fraction identity
    ok = True
    for tk, tl, tj in itertools.permutations([Fraction(x) for x in (1, 5, -2, 7, 13, -3)], 3):
        f = lambda a, b: Fraction(1) / (a - b)
        ok &= f(tk, tj) * f(tl, tj) + f(tk, tj) * f(tk, tl) - f(tk, tl) * f(tl, tj) == 0
    check("Eq. (7.4) f_kj f_lj + f_kj f_kl - f_kl f_lj == 0", ok)

    # Eq. (7.2) and Eq. (7.5): gradient condition and the logarithmic potential.
    # d/d(tau_k) of sum_{i<j} ln(tau_i - tau_j) S_ij, collected pairwise.
    ok_grad, ok_pot, ok_quot = True, True, True
    for N in (3, 4, 5):
        for k in range(1, N + 1):
            # potential derivative: pair (i,j) contributes only for k in {i,j}
            got = {}
            for i, j in itertools.combinations(range(1, N + 1), 2):
                if k == i:
                    got[(min(i, j), max(i, j))] = got.get((min(i, j), max(i, j)), 0) + 1  # +1/(t_i-t_j)
                elif k == j:
                    got[(min(i, j), max(i, j))] = got.get((min(i, j), max(i, j)), 0) - 1  # -1/(t_i-t_j)
            # H^G_k = sum_{j != k} S_kj / (tau_k - tau_j): pair (k,j) with sign +1 if k<j else -1
            want = {}
            for j in range(1, N + 1):
                if j == k:
                    continue
                want[(min(k, j), max(k, j))] = 1 if k < j else -1
            ok_pot &= got == want
    # The gradient condition is verified by DERIVING each partial derivative
    # from the Gaudin matrices, not by asserting the closed form. For
    # f(x)=1/(y-x) the exact difference quotient is 1/((y-x-h)(y-x)), so the
    # quotient matrix times (y-x-h)(y-x) must reproduce dot(l,k) exactly for
    # every nonzero rational h. That pins d_k H^G_l = dot(l,k)/(tau_l-tau_k)^2
    # without assuming it; only then are the two index orders compared.
    for N in (3, 4):
        n = 2 ** N
        sx = [None] + [site_op(N, j, _SX) for j in range(1, N + 1)]
        sy = [None] + [site_op(N, j, _SY) for j in range(1, N + 1)]
        sz = [None] + [site_op(N, j, _SZ) for j in range(1, N + 1)]
        dot = lambda i, j: sx[i] * sx[j] + sy[i] * sy[j] + sz[i] * sz[j]
        base = [None] + [Fraction(v) for v in (1, 5, -2, 7, 11)[:N]]

        def HG(tt, m):
            return _msum(n, [dot(m, j).scale((Fraction(1) / (tt[m] - tt[j]),
                                              Fraction(0)))
                             for j in range(1, N + 1) if j != m])

        for k in range(1, N + 1):
            for l in range(1, N + 1):
                if l == k:
                    continue
                d = base[l] - base[k]
                for h in (Fraction(1, 3), Fraction(-1, 7), Fraction(2, 5)):
                    tt = list(base)
                    tt[k] = base[k] + h
                    quot = (HG(tt, l) - HG(base, l)).scale(
                        (Fraction(1) / h, Fraction(0)))
                    resid = quot.scale(((d - h) * d, Fraction(0))) - dot(l, k)
                    ok_quot &= resid.is_zero
                # derivative pinned above, now compare the two index orders
                dkl = dot(l, k).scale((Fraction(1) / (d * d), Fraction(0)))
                e = base[k] - base[l]
                dlk = dot(k, l).scale((Fraction(1) / (e * e), Fraction(0)))
                ok_grad &= (dkl - dlk).is_zero
    check("Eq. (7.5) Phi^G = sum_{i<j} ln(tau_i-tau_j) s_i.s_j has grad = H^G_k", ok_pot)
    check("Eq. (7.2) difference quotient of H^G_l pins d_k H^G_l = s_l.s_k/(tau_l-tau_k)^2", ok_quot)
    check("Eq. (7.2) d_k H^G_l == d_l H^G_k (gradient/closedness condition)", ok_grad)


def _msum(n, mats):
    out = Mat(n)
    for m in mats:
        out = out + m
    return out


def main():
    check_table_vs_code()
    check_section2_four_lines()
    check_section2_fermions()
    check_section4()
    check_section5()
    check_section6()
    check_section7()
    check_sector_signs()
    check_spin_representation()
    check_section8_gaudin()

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)} check(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All consistency checks PASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
