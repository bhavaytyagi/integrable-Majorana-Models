"""Small exact linear solver over complex rational numbers."""

from __future__ import annotations

from dataclasses import dataclass

from .exact import ComplexRat, ONE_C, ZERO_C, cadd, cdiv, cmul, cneg, czero


@dataclass(frozen=True)
class LinearSolveResult:
    rank: int
    inconsistent_rows: list[int]
    solution: list[ComplexRat]
    pivot_columns: list[int]


def solve_linear(rows: list[list[ComplexRat]], rhs: list[ComplexRat], n_unknowns: int) -> LinearSolveResult:
    matrix = [row[:] for row in rows]
    values = rhs[:]
    n_rows = len(matrix)
    rank = 0
    pivots: list[int] = []

    for col in range(n_unknowns):
        pivot = next((r for r in range(rank, n_rows) if not czero(matrix[r][col])), None)
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        values[rank], values[pivot] = values[pivot], values[rank]

        inv = cdiv(ONE_C, matrix[rank][col])
        matrix[rank] = [cmul(inv, value) for value in matrix[rank]]
        values[rank] = cmul(inv, values[rank])

        for row in range(n_rows):
            if row == rank or czero(matrix[row][col]):
                continue
            factor = matrix[row][col]
            matrix[row] = [cadd(matrix[row][c], cneg(cmul(factor, matrix[rank][c]))) for c in range(n_unknowns)]
            values[row] = cadd(values[row], cneg(cmul(factor, values[rank])))

        pivots.append(col)
        rank += 1

    inconsistent = [
        row
        for row in range(n_rows)
        if all(czero(value) for value in matrix[row]) and not czero(values[row])
    ]

    solution = [ZERO_C for _ in range(n_unknowns)]
    if not inconsistent:
        for pivot_row, pivot_col in enumerate(pivots):
            solution[pivot_col] = values[pivot_row]

    return LinearSolveResult(rank=rank, inconsistent_rows=inconsistent, solution=solution, pivot_columns=pivots)

