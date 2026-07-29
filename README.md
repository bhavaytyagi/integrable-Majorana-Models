# Integrable Majorana Models

Exact computer-algebra verification for the paper **"A Search for a Hamiltonian's
Family"** (BT, NS).

The paper constructs a family of `N` mutually commuting, explicitly
time-dependent Hamiltonians, affine in `N` times,

```
H_mu = O_mu + sum_nu tau_nu B_{mu nu},     mu = 1..N,
```

built on `2N+2` Majorana generators out of four species of Hermitian bilinears
`A_j`, `V_j`, `D_0`, `D_j` that square to constants. This repository contains the
code that verifies every algebraic claim made in the paper.

All checks are **exact**. Operators live in the abstract Clifford algebra with
Laurent-polynomial coefficients in `(beta_j, g_j, tau_nu)` over the rationals; no
floating-point arithmetic is used and no matrix representation is chosen. A check
either compares against a closed form, an independently derived reference, or an
invariant that must hold identically — "it ran without an exception" is never
treated as a pass.

## Requirements

Python 3.9+. **No third-party dependencies** — the Clifford arithmetic is
implemented from scratch in `HamFam_Resources/python/hamfam/`.

## Layout

```
HamFam_Resources/python/
  hamfam/              exact Clifford-algebra package (dependency-free)
    clifford.py          Majorana monomials, products, commutators
    exact.py             exact rational / Laurent-polynomial coefficients
    linear.py            exact linear solves (rank, nullity, consistency)
    models.py            the bilinears and the operator tables
  verify_n{3,4,5}_majorana.py       stored N=3,4,5 tables satisfy all conditions
  verify_n3_spin_reference.py       numerical cross-check in the spin rep
  solve_n3_constant_terms.py        exact solve for the N=3 constant column
  solve_n4_fourth_flow.py           exact solve for the N=4 fourth flow
  solve_n5_structured_ansatz.py     exact solve for the N=5 structured ansatz
  solve_n4_compact_ansatz.py        negative control (see below)

paper_checks/
  check_paper_consistency.py        the paper-facing check (see below)
  extension_test.py                 the extended (N+k)-flow hierarchy
  duality_and_gf_test.py            parity duality and generating functions

algebra_checks/
  commutator_tests.py               N=3,4,5 tables + closed-form projection
  uniqueness_checks.py              full-rank / zero-nullity uniqueness solves
```

## How to run

Each script is standalone and must be run from its own directory (the import
bootstrap resolves the package path relative to the script location):

```sh
cd paper_checks    && python3 check_paper_consistency.py   # ~90 s
cd paper_checks    && python3 extension_test.py
cd paper_checks    && python3 duality_and_gf_test.py
cd algebra_checks  && python3 commutator_tests.py          # ~13 s
cd algebra_checks  && python3 uniqueness_checks.py         # ~38 s
cd HamFam_Resources/python && python3 verify_n3_majorana.py
```

Every script exits nonzero on failure, with one deliberate exception noted below.

## Which script establishes which claim

| Claim in the paper | Verified by |
|---|---|
| The four-line statement of the family (the `B_{mu nu}`, `B_{2 nu}`, `O_mu` table), including the coefficient extraction from the kernel `(E(z)E(w) - prod_j(1+beta_j^2 zw))/(z+w)` | `paper_checks/check_paper_consistency.py` |
| Rewriting as `N` standard fermions coupled to one Majorana; the Jordan–Wigner spin representation (with exact `2^N` Pauli matrices) | `paper_checks/check_paper_consistency.py` |
| The local commutator identities among `A_j`, `V_j`, `D_0`, `D_j`, and the fermionization rule `M -> i c_0 M` | `paper_checks/check_paper_consistency.py` |
| The explicit `N=3` and `N=4` members, the `N=3` operator table, and the generating potential `Phi_3` | `paper_checks/check_paper_consistency.py` |
| Generating function for `B_{mu nu}`; the factorization `P(z+w, zw) = E(z)E(w)`; the row and constant-column series over their full extended index ranges | `paper_checks/check_paper_consistency.py`, `paper_checks/duality_and_gf_test.py` |
| Linearization of `e_a(A) e_b(A)` in `C[A]/(A_j^2 - beta_j^2)`; the three-term recursion; the kernel identity; the reflection identity `E(z)E(-z) = prod_j(1 - beta_j^2 z^2)`; the Krawtchouk evaluation | `paper_checks/check_paper_consistency.py` |
| All integrability conditions for the stored `N=3,4,5` families, and the projection of the closed form onto them | `algebra_checks/commutator_tests.py`, `HamFam_Resources/python/verify_n{3,4,5}_majorana.py` |
| Existence of the `(N+1)`-st commuting flow: `(N, flows) = (2,3), (3,4), (4,5)` all pass | `paper_checks/extension_test.py` |
| Termination of the hierarchy: at `(2,4)` and `(3,5)`, row `N+2` reduces to the single entry `B_{N+2,2} = (-1)^N X` and row `N+3` vanishes identically | `paper_checks/extension_test.py` |
| `X = A_1...A_N D_0` is central with `X^2 = prod_j beta_j^2`; the column-by-column parity duality `B_{N+1,nu} = (-1)^{N-1} X B_{2 nu}` for `N=2,3,4` | `paper_checks/duality_and_gf_test.py`, `paper_checks/check_paper_consistency.py` |
| The chirality identity `X = kappa_N (prod_j beta_j) Gamma` with `kappa_N = 1` for odd `N` and `i` for even `N`, and `Gamma^2 = (-1)^{N+1}` (measured directly for `N = 1..6`) | `paper_checks/check_paper_consistency.py` |
| Uniqueness evidence: the structured linear solves are full rank with zero nullity and no inconsistency (`N=3`: rank 13/13; `N=4` fourth flow: rank 448/448; `N=5`: rank 10/10) | `algebra_checks/uniqueness_checks.py`, `HamFam_Resources/python/solve_n{3,4,5}_*.py` |
| The Gaudin comparison: gradient condition, commutativity via the three-point partial-fraction identity, and the logarithmic potential | `paper_checks/check_paper_consistency.py` |

## Notes on how the checks are built

**`check_paper_consistency.py` is transcribed from the printed paper.** Every
displayed equation that makes an algebraic claim is typed in *literally, from the
printed equation*, and then compared against the closed form. The closed form
itself is **re-implemented from the printed table** rather than imported from
`extension_test.py`, so the check is independent of the builder; the two are
nevertheless cross-checked against each other entry by entry. Preserving that
independence is the point — a check that imports the object it is testing only
tests the object against itself.

**Negative controls are load-bearing.** Several checks assert that something
*fails*, and those assertions are part of the evidence, not bugs:

- `duality_and_gf_test.py` verifies that the generating identity holds with the
  substitution `x_j = A_j` and **fails** with `x_j = beta_j A_j`. The normalization
  matters because the collapse relation is `A_j^2 = beta_j^2`, whereas
  `(beta_j A_j)^2 = beta_j^4`.
- `duality_and_gf_test.py` also reports one expected mismatch on the `(N+2)`-nd
  column of the duality test. That is not a failure: it is the central constant
  `H_{N+1} = (-1)^{N-1} X (H_2 - (-1)^N tau_{N+2} X)`. The test asserts the
  mismatch equals exactly `±prod_j beta_j^2` and nothing else.
- `check_paper_consistency.py` asserts that truncating the row-1 series at
  `nu = N` does **not** close, and that the odd-`N` form of the chirality identity
  fails at even `N`.
- **`solve_n4_compact_ansatz.py` exits 1 by design.** It demonstrates that the
  compact `N=4` ansatz is inconsistent; the consistent solve is
  `solve_n4_fourth_flow.py`.

**One script is not exact.** `verify_n3_spin_reference.py` is float-based
(residuals `~1e-14`) and is a useful independent cross-check in an explicit spin
representation, but it is not one of the exact tests and should not be counted as
one.

## Notation: paper vs. code

The code uses the same names as the paper (`A_j`, `V_j`, `tau_mu`, `O_mu`,
`B_{mu nu}`) with one deliberate exception:

| Code | Paper |
|---|---|
| `a_bilinear` | the **unweighted** `i gamma_{N+j} gamma_j = A_j / beta_j` — the weight `beta_j` is supplied at each call site, so the symbol `A_j` never denotes two different objects |
| `v_bilinear` | `V_j` |
| `tau1`, `tau2`, ... | `tau_1 = t`, `tau_2 = epsilon`, `tau_3 = eta`, ... |
