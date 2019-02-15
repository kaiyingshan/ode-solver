import sympy
from sympy import *
import cmath
from typing import Union


def find_root(a: Union[float, int], b: Union[float, int], c: Union[float, int]):
    disc = sqrt(b**2 - 4*a*c)
    return (-b + disc) / (2*a), (-b - disc) / (2*a)


def sec_order_const_coeff(a: Union[float, int], b: Union[float, int], c: Union[float, int]) -> [Symbol, Symbol]:
    r1, r2 = find_root(a, b, c)
    real1, imag1 = r1.as_real_imag()
    real2, imag2 = r2.as_real_imag()

    t = Symbol("t")

    if imag1 == 0:
        y1 = E ** (real1 * t)
    else:
        y1 = (E ** (real1 * t)) * cos(imag1 * t)
    if imag2 == 0:
        if real1 == real2:
            y2 = t * E ** (real2 * t)
        else:
            y2 = E ** (real2 * t)
    else:
        y2 = (E ** (real2 * t)) * sin(imag2 * t)

    return y1, y2


def sec_order_euler(a: Union[float, int], b: Union[float, int], c: Union[float, int]) -> [Symbol, Symbol]:
    # t^r is a solution given that
    r1, r2 = find_root(a, b - a, c)

    real1, imag1 = r1.as_real_imag()
    real2, imag2 = r2.as_real_imag()

    t = Symbol("t")

    if imag1 == 0:
        y1 = t ** real1
    else:
        y1 = (t ** (real1)) * cos(imag1 * ln(t))
    if imag2 == 0:
        if real1 == real2:
            y2 = ln(t) * t ** real2
        else:
            y2 = t ** real2
    else:
        y2 = (t ** (real2)) * sin(imag2 * ln(t))

    return y1, y2


def solve_ivp(y1: Symbol, y2: Symbol, t1, yp1, t2, yp2):
    C1 = Symbol("C1")
    C2 = Symbol("C2")

    y = C1 * y1 + C2 * y2
    yp = diff(y, "t")

    eq1 = y.subs("t", t1) - yp1
    eq2 = yp.subs("t", t2) - yp2

    sol = solve([eq1, eq2], C1, C2)

    print(sol)


def main():
    y1, y2 = sec_order_const_coeff(1, 3, 2)
    print(y1)
    print(y2)

    y1, y2 = sec_order_euler(1, 11, 25)
    print(y1)
    print(y2)

    solve_ivp(y1, y2, 1, 8, 1, 5)


if __name__ == "__main__":
    main()
