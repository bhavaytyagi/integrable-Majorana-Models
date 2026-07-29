#!/usr/bin/env python3
"""Numerically verify the published N=3 spin gamma-magnet family.

This checks Eqs. (13)-(15) in 1905.05287v4 with plain Python matrices. It is
not a proof, but it is a useful guardrail: the original three-qubit spin family
commutes, while the current even-Majorana extension transcription fails the
full affine check in verify_n3_majorana.py.
"""

from __future__ import annotations


Matrix = list[list[complex]]


def eye(n: int) -> Matrix:
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def mat_add(a: Matrix, b: Matrix, scale: complex = 1) -> Matrix:
    return [[a[i][j] + scale * b[i][j] for j in range(len(a))] for i in range(len(a))]


def mat_scale(scale: complex, a: Matrix) -> Matrix:
    return [[scale * value for value in row] for row in a]


def mat_mul(a: Matrix, b: Matrix) -> Matrix:
    n = len(a)
    return [[sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


def kron(a: Matrix, b: Matrix) -> Matrix:
    out = []
    for row_a in a:
        for row_b in b:
            out.append([value_a * value_b for value_a in row_a for value_b in row_b])
    return out


def op(a: Matrix, b: Matrix, c: Matrix) -> Matrix:
    return kron(kron(a, b), c)


def comm(a: Matrix, b: Matrix) -> Matrix:
    return mat_add(mat_mul(a, b), mat_mul(b, a), -1)


def norm_max(a: Matrix) -> float:
    return max(abs(value) for row in a for value in row)


def plus(parts: list[Matrix]) -> Matrix:
    out = [[0 for _ in range(len(parts[0]))] for _ in range(len(parts[0]))]
    for part in parts:
        out = mat_add(out, part)
    return out


def main() -> int:
    ident = [[1, 0], [0, 1]]
    x = [[0, 1], [1, 0]]
    y = [[0, -1j], [1j, 0]]
    z = [[1, 0], [0, -1]]

    tau_x, tau_z = op(x, ident, ident), op(z, ident, ident)
    s_x, s_y, s_z = op(ident, x, ident), op(ident, y, ident), op(ident, z, ident)
    sig_x, sig_z = op(ident, ident, x), op(ident, ident, z)

    z1, z2, z3 = tau_z, s_z, sig_z
    gamma0 = mat_mul(mat_mul(z1, z2), z3)

    beta = {1: 2.0, 2: 3.0, 3: 5.0}
    g = {1: 0.7, 2: 1.1, 3: 1.3}
    t = 0.2
    eps = -0.4
    eta = 0.6

    h1 = plus(
        [
            mat_scale(eps, gamma0),
            mat_scale(t, plus([mat_scale(beta[1], z1), mat_scale(beta[2], z2), mat_scale(beta[3], z3)])),
            mat_scale(
                eta,
                plus(
                    [
                        mat_scale(beta[1] * beta[2], mat_mul(z1, z2)),
                        mat_scale(beta[1] * beta[3], mat_mul(z1, z3)),
                        mat_scale(beta[2] * beta[3], mat_mul(z2, z3)),
                    ]
                ),
            ),
            mat_scale(g[1], tau_x),
            mat_scale(g[2], mat_mul(tau_z, s_x)),
            mat_scale(g[3], mat_mul(mat_mul(tau_z, s_z), sig_x)),
        ]
    )

    h2 = plus(
        [
            mat_scale(t, gamma0),
            mat_scale(eps, plus([mat_scale(1 / beta[1], z1), mat_scale(1 / beta[2], z2), mat_scale(1 / beta[3], z3)])),
            mat_scale(
                eta,
                plus(
                    [
                        mat_scale(beta[3], mat_mul(z1, z2)),
                        mat_scale(beta[2], mat_mul(z1, z3)),
                        mat_scale(beta[1], mat_mul(z2, z3)),
                    ]
                ),
            ),
            mat_scale(g[1] / beta[1], mat_mul(mat_mul(tau_x, s_z), sig_z)),
            mat_scale(g[2] / beta[2], mat_mul(s_x, sig_z)),
            mat_scale(g[3] / beta[3], sig_x),
        ]
    )

    h3 = plus(
        [
            mat_scale(
                eps,
                plus(
                    [
                        mat_scale(beta[3], mat_mul(z1, z2)),
                        mat_scale(beta[2], mat_mul(z1, z3)),
                        mat_scale(beta[1], mat_mul(z2, z3)),
                    ]
                ),
            ),
            mat_scale(
                t,
                plus(
                    [
                        mat_scale(beta[1] * beta[2], mat_mul(z1, z2)),
                        mat_scale(beta[1] * beta[3], mat_mul(z1, z3)),
                        mat_scale(beta[2] * beta[3], mat_mul(z2, z3)),
                    ]
                ),
            ),
            mat_scale(
                eta,
                plus(
                    [
                        mat_scale(2 * beta[1] * beta[2] * beta[3], gamma0),
                        mat_scale(beta[1] * (beta[2] ** 2 + beta[3] ** 2), z1),
                        mat_scale(beta[2] * (beta[1] ** 2 + beta[3] ** 2), z2),
                        mat_scale(beta[3] * (beta[1] ** 2 + beta[2] ** 2), z3),
                    ]
                ),
            ),
            mat_scale(g[1], plus([mat_scale(beta[3], mat_mul(tau_x, sig_z)), mat_scale(beta[2], mat_mul(tau_x, s_z))])),
            mat_scale(
                g[2],
                plus([mat_scale(beta[3], mat_mul(mat_mul(tau_z, s_x), sig_z)), mat_scale(beta[1], s_x)]),
            ),
            mat_scale(g[3], plus([mat_scale(beta[2], mat_mul(tau_z, sig_x)), mat_scale(beta[1], mat_mul(s_z, sig_x))])),
        ]
    )

    norms = {
        "[H1,H2]": norm_max(comm(h1, h2)),
        "[H1,H3]": norm_max(comm(h1, h3)),
        "[H2,H3]": norm_max(comm(h2, h3)),
    }
    print("Published N=3 spin gamma-magnet numerical check")
    for name, value in norms.items():
        print(f"{name:8s}: {value:.3e}")
    return 0 if max(norms.values()) < 1e-10 else 1


if __name__ == "__main__":
    raise SystemExit(main())

