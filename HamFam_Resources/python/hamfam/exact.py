"""Exact scalar arithmetic for Hamiltonian-family Clifford checks.

The coefficients are Laurent polynomials with complex rational coefficients.
This is deliberately small and dependency-free; SymPy is not available in the
current environment and the checks here only need polynomial collection.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping

ComplexRat = tuple[Fraction, Fraction]

ZERO_C: ComplexRat = (Fraction(0), Fraction(0))
ONE_C: ComplexRat = (Fraction(1), Fraction(0))
MINUS_ONE_C: ComplexRat = (Fraction(-1), Fraction(0))
I_C: ComplexRat = (Fraction(0), Fraction(1))


def cadd(a: ComplexRat, b: ComplexRat) -> ComplexRat:
    return (a[0] + b[0], a[1] + b[1])


def cneg(a: ComplexRat) -> ComplexRat:
    return (-a[0], -a[1])


def cmul(a: ComplexRat, b: ComplexRat) -> ComplexRat:
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def cdiv(a: ComplexRat, b: ComplexRat) -> ComplexRat:
    den = b[0] * b[0] + b[1] * b[1]
    if den == 0:
        raise ZeroDivisionError("complex rational division by zero")
    return ((a[0] * b[0] + a[1] * b[1]) / den, (a[1] * b[0] - a[0] * b[1]) / den)


def czero(a: ComplexRat) -> bool:
    return a[0] == 0 and a[1] == 0


def cint(n: int) -> ComplexRat:
    return (Fraction(n), Fraction(0))


def _fmt_fraction(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def fmt_complex(c: ComplexRat) -> str:
    real, imag = c
    if imag == 0:
        return _fmt_fraction(real)
    if real == 0:
        if imag == 1:
            return "i"
        if imag == -1:
            return "-i"
        return f"{_fmt_fraction(imag)}i"
    sign = "+" if imag > 0 else "-"
    imag_abs = abs(imag)
    imag_s = "i" if imag_abs == 1 else f"{_fmt_fraction(imag_abs)}i"
    return f"{_fmt_fraction(real)}{sign}{imag_s}"


@dataclass(frozen=True)
class LaurentPoly:
    """Laurent polynomial in a fixed ordered variable tuple."""

    variables: tuple[str, ...]
    terms: Mapping[tuple[int, ...], ComplexRat]

    def __post_init__(self) -> None:
        clean = {
            tuple(exp): coeff
            for exp, coeff in self.terms.items()
            if not czero(coeff)
        }
        for exp in clean:
            if len(exp) != len(self.variables):
                raise ValueError(f"exponent length {len(exp)} does not match variables")
        object.__setattr__(self, "terms", clean)

    @classmethod
    def zero(cls, variables: Iterable[str]) -> "LaurentPoly":
        return cls(tuple(variables), {})

    @classmethod
    def const(cls, variables: Iterable[str], coeff: ComplexRat | int) -> "LaurentPoly":
        variables = tuple(variables)
        if isinstance(coeff, int):
            coeff = cint(coeff)
        if czero(coeff):
            return cls.zero(variables)
        return cls(variables, {tuple(0 for _ in variables): coeff})

    @classmethod
    def var(cls, variables: Iterable[str], name: str, power: int = 1) -> "LaurentPoly":
        variables = tuple(variables)
        exp = [0 for _ in variables]
        exp[variables.index(name)] = power
        return cls(variables, {tuple(exp): ONE_C})

    @property
    def is_zero(self) -> bool:
        return not self.terms

    def _check(self, other: "LaurentPoly") -> None:
        if self.variables != other.variables:
            raise ValueError("Laurent polynomial variable mismatch")

    def __add__(self, other: "LaurentPoly") -> "LaurentPoly":
        self._check(other)
        out: dict[tuple[int, ...], ComplexRat] = dict(self.terms)
        for exp, coeff in other.terms.items():
            out[exp] = cadd(out.get(exp, ZERO_C), coeff)
        return LaurentPoly(self.variables, out)

    def __neg__(self) -> "LaurentPoly":
        return LaurentPoly(self.variables, {exp: cneg(coeff) for exp, coeff in self.terms.items()})

    def __sub__(self, other: "LaurentPoly") -> "LaurentPoly":
        return self + (-other)

    def __mul__(self, other: "LaurentPoly") -> "LaurentPoly":
        self._check(other)
        out: dict[tuple[int, ...], ComplexRat] = {}
        for exp_a, coeff_a in self.terms.items():
            for exp_b, coeff_b in other.terms.items():
                exp = tuple(a + b for a, b in zip(exp_a, exp_b))
                out[exp] = cadd(out.get(exp, ZERO_C), cmul(coeff_a, coeff_b))
        return LaurentPoly(self.variables, out)

    def scale(self, coeff: ComplexRat | int) -> "LaurentPoly":
        if isinstance(coeff, int):
            coeff = cint(coeff)
        return LaurentPoly(self.variables, {exp: cmul(coeff, value) for exp, value in self.terms.items()})

    def coefficient(self, exp: tuple[int, ...]) -> ComplexRat:
        return self.terms.get(exp, ZERO_C)

    def format(self) -> str:
        if not self.terms:
            return "0"
        parts: list[str] = []
        for exp, coeff in sorted(self.terms.items()):
            monomial = "*".join(
                name if power == 1 else f"{name}^{power}"
                for name, power in zip(self.variables, exp)
                if power
            )
            coeff_s = fmt_complex(coeff)
            if monomial:
                if coeff_s == "1":
                    parts.append(monomial)
                elif coeff_s == "-1":
                    parts.append(f"-{monomial}")
                else:
                    parts.append(f"{coeff_s}*{monomial}")
            else:
                parts.append(coeff_s)
        return " + ".join(parts).replace("+ -", "- ")

