"""Canonical Clifford algebra expressions over exact Laurent polynomials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .exact import I_C, LaurentPoly, MINUS_ONE_C, ONE_C, ComplexRat


def _multiply_masks(left: int, right: int) -> tuple[int, int]:
    """Return the Clifford sign and xor mask for two canonical monomials."""

    inversions = 0
    bits = left
    while bits:
        lsb = bits & -bits
        index = lsb.bit_length() - 1
        inversions += (right & ((1 << index) - 1)).bit_count()
        bits ^= lsb
    return (-1 if inversions % 2 else 1), left ^ right


@dataclass(frozen=True)
class CliffordAlgebra:
    labels: tuple[int, ...]
    variables: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("duplicate Clifford generator labels")

    @property
    def label_to_bit(self) -> dict[int, int]:
        return {label: i for i, label in enumerate(self.labels)}

    def zero_poly(self) -> LaurentPoly:
        return LaurentPoly.zero(self.variables)

    def const_poly(self, coeff: ComplexRat | int) -> LaurentPoly:
        return LaurentPoly.const(self.variables, coeff)

    def var(self, name: str, power: int = 1) -> LaurentPoly:
        return LaurentPoly.var(self.variables, name, power)

    def zero(self) -> "CliffordExpr":
        return CliffordExpr(self, {})

    def scalar(self, value: LaurentPoly | ComplexRat | int) -> "CliffordExpr":
        if isinstance(value, LaurentPoly):
            poly = value
        else:
            poly = self.const_poly(value)
        return CliffordExpr(self, {} if poly.is_zero else {0: poly})

    def gen(self, label: int) -> "CliffordExpr":
        return CliffordExpr(self, {1 << self.label_to_bit[label]: self.const_poly(1)})

    def product(self, labels: Iterable[int]) -> "CliffordExpr":
        out = self.scalar(1)
        for label in labels:
            out = out * self.gen(label)
        return out


@dataclass(frozen=True)
class CliffordExpr:
    algebra: CliffordAlgebra
    terms: Mapping[int, LaurentPoly]

    def __post_init__(self) -> None:
        clean = {mask: poly for mask, poly in self.terms.items() if not poly.is_zero}
        object.__setattr__(self, "terms", clean)

    @property
    def is_zero(self) -> bool:
        return not self.terms

    def _check(self, other: "CliffordExpr") -> None:
        if self.algebra != other.algebra:
            raise ValueError("Clifford algebra mismatch")

    def __add__(self, other: "CliffordExpr") -> "CliffordExpr":
        self._check(other)
        out: dict[int, LaurentPoly] = dict(self.terms)
        for mask, poly in other.terms.items():
            out[mask] = out.get(mask, self.algebra.zero_poly()) + poly
        return CliffordExpr(self.algebra, out)

    def __neg__(self) -> "CliffordExpr":
        return CliffordExpr(self.algebra, {mask: -poly for mask, poly in self.terms.items()})

    def __sub__(self, other: "CliffordExpr") -> "CliffordExpr":
        return self + (-other)

    def __mul__(self, other: "CliffordExpr") -> "CliffordExpr":
        self._check(other)
        out: dict[int, LaurentPoly] = {}
        for mask_a, poly_a in self.terms.items():
            for mask_b, poly_b in other.terms.items():
                sign, mask = _multiply_masks(mask_a, mask_b)
                poly = poly_a * poly_b
                if sign < 0:
                    poly = -poly
                out[mask] = out.get(mask, self.algebra.zero_poly()) + poly
        return CliffordExpr(self.algebra, out)

    def scale(self, scalar: LaurentPoly | ComplexRat | int) -> "CliffordExpr":
        if isinstance(scalar, LaurentPoly):
            poly = scalar
        else:
            poly = self.algebra.const_poly(scalar)
        return CliffordExpr(self.algebra, {mask: poly * term for mask, term in self.terms.items()})

    def commutator(self, other: "CliffordExpr") -> "CliffordExpr":
        return self * other - other * self

    def format_mask(self, mask: int) -> str:
        if mask == 0:
            return "1"
        return " ".join(
            f"m{label}"
            for bit, label in enumerate(self.algebra.labels)
            if mask & (1 << bit)
        )

    def format_terms(self, limit: int | None = None) -> str:
        if not self.terms:
            return "0"
        rows = []
        for index, (mask, poly) in enumerate(sorted(self.terms.items())):
            if limit is not None and index >= limit:
                rows.append(f"... ({len(self.terms) - limit} more terms)")
                break
            rows.append(f"{self.format_mask(mask)} : {poly.format()}")
        return "\n".join(rows)


def i_times(expr: CliffordExpr) -> CliffordExpr:
    return expr.scale(I_C)


def minus(expr: CliffordExpr) -> CliffordExpr:
    return expr.scale(MINUS_ONE_C)


def sum_expr(parts: Iterable[CliffordExpr], algebra: CliffordAlgebra) -> CliffordExpr:
    out = algebra.zero()
    for part in parts:
        out = out + part
    return out


def one(algebra: CliffordAlgebra) -> CliffordExpr:
    return algebra.scalar(ONE_C)

