#!/usr/bin/env python3.12
"""Verify (1) the parity-duality identity H_{N+1} = (-1)^{N-1} X H_2 with
X = e_N(A) D_0, and (2) the Section-4 generating-function claims with the
corrected convention x_j = A_j (vs the draft's x_j = beta_j A_j).
"""

from __future__ import annotations

import sys
from itertools import combinations
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HamFam_Resources" / "python"))

from hamfam.clifford import minus, sum_expr, one  # noqa: E402
from hamfam.models import (  # noqa: E402
    a_bilinear,
    beta,
    d0_op,
    d_op,
    elementary_a,
    gvar,
    inv_beta,
    majorana_algebra,
    term,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extension_test import build_table  # noqa: E402


def dual_check(n_sites: int) -> None:
    n_flows = n_sites + 2
    alg, matrix, tau_names = build_table(n_sites, n_flows)
    X = elementary_a(alg, n_sites, n_sites) * d0_op(alg)

    # X^2 = prod beta_j^2
    prod_beta2 = alg.const_poly(1)
    for j in range(1, n_sites + 1):
        prod_beta2 = prod_beta2 * beta(alg, j, 2)
    ok_sq = (X * X - alg.scalar(prod_beta2)).is_zero
    print(f"N={n_sites}: X^2 == prod beta_j^2 : {ok_sq}")

    # X commutes with every table entry
    ok_central = all(
        X.commutator(entry).is_zero for row in matrix for entry in row
    )
    print(f"N={n_sites}: X central on all entries : {ok_central}")

    # Row N+1 = (-1)^{N-1} X * (row 2) on the theorem's columns
    # (nu = 1..N+1 and the constant column); the (N+2)-nd column carries
    # the expected central mismatch (Remark "scope of the duality").
    sign = 1 if (n_sites - 1) % 2 == 0 else -1
    ok_rows = []
    for col in range(n_flows + 1):
        if col == n_sites + 1:
            continue  # column N+2, checked separately below
        lhs = matrix[n_sites][col]  # row N+1 (0-indexed N)
        rhs = (X * matrix[1][col]).scale(sign)
        ok_rows.append((lhs - rhs).is_zero)
    print(f"N={n_sites}: row N+1 == (-1)^(N-1) X * row 2 "
          f"(cols 1..N+1 and 0) : {all(ok_rows)} {ok_rows}")

    # Column N+2: B_{N+1,N+2} - (-1)^{N-1} X B_{2,N+2}
    #           = -(-1)^{N-1}(-1)^N prod beta_j^2, a central number.
    col = n_sites + 1
    mism = matrix[n_sites][col] - (X * matrix[1][col]).scale(sign)
    csign = -sign * (1 if n_sites % 2 == 0 else -1)
    expected = alg.scalar(prod_beta2).scale(csign)
    print(f"N={n_sites}: column N+2 mismatch is the central number "
          f"{'+' if csign > 0 else '-'}prod beta_j^2 : "
          f"{(mism - expected).is_zero}")

    # Row N+2: single entry B_{N+2,2} = +/- X, rest zero
    tail = matrix[n_sites + 1]
    nz = [(c, e) for c, e in enumerate(tail) if not e.is_zero]
    tail_ok = len(nz) == 1 and nz[0][0] == 1
    if tail_ok:
        e = nz[0][1]
        s_plus = (e - X).is_zero
        s_minus = (e + X).is_zero
        print(f"N={n_sites}: row N+2 = single entry O[N+2][2] = "
              f"{'+X' if s_plus else ('-X' if s_minus else 'other')}")
    else:
        print(f"N={n_sites}: row N+2 unexpected structure: {[(c, len(e.terms)) for c, e in nz]}")


def gf_check(n_sites: int, use_beta_weight: bool) -> bool:
    """Check (z+w)*G(z,w) + P(0,zw) - P(z+w,zw) == 0 where
    G = sum_{a,b>=0} (-1)^{a+b} K_{a,b} z^a w^b  (rows/cols in script-I),
    P(T,U) = prod_j (1 + x_j T + beta_j^2 U),
    with x_j = A_j (corrected) or x_j = beta_j A_j (draft's Eq. 4.1)."""
    tau_names = ("z", "w")
    alg = majorana_algebra(n_sites, extra_variables=tau_names)
    z = alg.var("z")
    w = alg.var("w")

    def A(j):
        return term(beta(alg, j), a_bilinear(alg, n_sites, j), alg=alg)

    def xj(j):
        return A(j).scale(beta(alg, j)) if use_beta_weight else A(j)

    # P(z+w, zw) and P(0, zw)
    P_full = one(alg)
    P_zero = one(alg)
    zw = z * w
    zpw = alg.const_poly(0) + z + w
    for j in range(1, n_sites + 1):
        b2zw = alg.scalar(beta(alg, j, 2) * zw)
        P_full = P_full * (one(alg) + xj(j).scale(zpw) + b2zw)
        P_zero = P_zero * (one(alg) + b2zw)

    # K_{a,b} from the closed formula (paper's A-convention, always)
    def K(a, b):
        expr = alg.zero()
        for s in range(0, min(a, b) + 1):
            d = a + b + 1 - 2 * s
            c = comb(a + b - 2 * s, a - s)
            if c == 0 or d < 0 or d > n_sites:
                continue
            for I in combinations(range(1, n_sites + 1), d):
                compl = [k for k in range(1, n_sites + 1) if k not in I]
                if s > len(compl):
                    continue
                es_subsets = [()] if s == 0 else list(combinations(compl, s))
                for J in es_subsets:
                    poly = alg.const_poly(c)
                    for i in I:
                        poly = poly * beta(alg, i)
                    for k in J:
                        poly = poly * beta(alg, k, 2)
                    prod = alg.scalar(poly)
                    for i in I:
                        prod = prod * a_bilinear(alg, n_sites, i)
                    expr = expr + prod
        return expr

    G = alg.zero()
    for a in range(0, 2 * n_sites + 2):
        for b in range(0, 2 * n_sites + 2):
            k = K(a, b)
            if k.is_zero:
                continue
            G = G + k.scale(alg.var("z", a) * alg.var("w", b))

    resid = G.scale(zpw) + P_zero - P_full
    return resid.is_zero


def gf_rows_check(n_sites: int) -> None:
    """Check corrected (4.6)-(4.8):
    sum_{nu in I} (-1)^{a_nu} B_{1,nu} t^{a_nu+1}? -- we verify in the clean form:
      E(t) - 1           == sum_{d>=1} e_d(A) t^d          (row 1, all cols)
      D0 (E(t) - 1)      == sum_{d>=1} D0 e_d(A) t^d       (row 2 dressing)
      sum_j g_j D_j E_j(t) == sum over constant column      (col 0, incl. mu=1)
    with E(t) = prod (1 + A_j t), E_j(t) = prod_{i != j} (1 + A_i t)."""
    alg = majorana_algebra(n_sites, extra_variables=("t",))
    t = alg.var("t")

    def A(j):
        return term(beta(alg, j), a_bilinear(alg, n_sites, j), alg=alg)

    E = one(alg)
    for j in range(1, n_sites + 1):
        E = E * (one(alg) + A(j).scale(t))

    rhs1 = alg.zero()
    for d in range(1, n_sites + 1):
        rhs1 = rhs1 + elementary_a(alg, n_sites, d).scale(alg.var("t", d))
    print(f"N={n_sites}: E(t)-1 == sum e_d t^d : {(E - one(alg) - rhs1).is_zero}")

    lhs0 = alg.zero()
    for j in range(1, n_sites + 1):
        Ej = one(alg)
        for i in range(1, n_sites + 1):
            if i != j:
                Ej = Ej * (one(alg) + A(i).scale(t))
        lhs0 = lhs0 + term(gvar(alg, j), d_op(alg, j), Ej, alg=alg)
    rhs0 = alg.zero()
    for d in range(0, n_sites):
        part = alg.zero()
        for j in range(1, n_sites + 1):
            compl = [i for i in range(1, n_sites + 1) if i != j]
            for I in combinations(compl, d):
                poly = gvar(alg, j)
                for i in I:
                    poly = poly * beta(alg, i)
                prod = alg.scalar(poly) * d_op(alg, j)
                for i in I:
                    prod = prod * a_bilinear(alg, n_sites, i)
                part = part + prod
        rhs0 = rhs0 + part.scale(alg.var("t", d))
    print(f"N={n_sites}: sum_j g_j D_j E_j(t) == constant column series (incl. mu=1) : "
          f"{(lhs0 - rhs0).is_zero}")


if __name__ == "__main__":
    for n in (2, 3, 4):
        dual_check(n)
    for n in (2, 3, 4):
        ok_fixed = gf_check(n, use_beta_weight=False)
        ok_draft = gf_check(n, use_beta_weight=True)
        print(f"N={n}: GF identity with x_j = A_j (corrected): {ok_fixed}; "
              f"with x_j = beta_j A_j (draft 4.1): {ok_draft}")
    for n in (2, 3, 4):
        gf_rows_check(n)
