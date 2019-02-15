import sympy
from sympy import Symbol, E, cos, sin, solve, exp, diff, integrate, sqrt, ln, Matrix
from sympy import *
import cmath
from typing import Union, List

number = Union[int, float]


def find_root(a: number, b: number, c: number):
    disc = sqrt(b**2 - 4*a*c)
    return (-b + disc) / (2*a), (-b - disc) / (2*a)


def sec_order_const_coeff(a: number, b: number, c: number) -> [Symbol, Symbol]:
    r1, r2 = find_root(a, b, c)
    real1, imag1 = r1.as_real_imag()
    real2, imag2 = r2.as_real_imag()

    t = Symbol("t")

    if imag1 == 0:
        y1 = exp(real1 * t)
    else:
        y1 = exp(real1 * t) * cos(imag1 * t)
    if imag2 == 0:
        if real1 == real2:
            y2 = t * exp(real2 * t)
        else:
            y2 = exp(real2 * t)
    else:
        y2 = exp(real2 * t) * sin(imag2 * t)

    return y1, y2


def sec_order_euler(a: number, b: number, c: number) -> [Symbol, Symbol]:
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


def solve_ivp(y1: Symbol, y2: Symbol, t1: number, yp1: number, t2: number, yp2: number, t: Symbol = Symbol("t")):
    C1 = Symbol("C1")
    C2 = Symbol("C2")

    y = C1 * y1 + C2 * y2
    yp = diff(y, "t")

    eq1 = y.subs("t", t1) - yp1
    eq2 = yp.subs("t", t2) - yp2

    sol = solve([eq1, eq2], C1, C2)

    print(sol)


def red_order(y1: Symbol, pt: Symbol, qt: Symbol, gt: Symbol, t: Symbol = Symbol("t")) -> Symbol:
    y1p = diff(y1, "t")
    fac = exp(integrate(pt, t))
    mu_t = (y1**2) * fac  # integrating factor
    C1 = Symbol("C1")
    if gt == 0:
        vp = C1 / mu_t
    else:
        vp = integrate(y1 * gt * fac, t) / mu_t

    v = integrate(vp, t) + Symbol("C2")
    return v * y1


def Wronskian(*args: List[Symbol], t: Symbol = Symbol("t")) -> Symbol:
    size = len(args)
    w: Matrix = Matrix([
        [diff(args[x], t, i) for i in range(size)] for x in range(size)
    ]).transpose()

    return w.det()


def var_parameters(y1: Symbol, y2: Symbol, gt: Symbol, t: Symbol = Symbol("t")):
    pass


def main():
    y1, y2 = sec_order_const_coeff(1, 3, 2)
    print(y1)
    print(y2)

    y1, y2 = sec_order_euler(1, 11, 25)
    print(y1)
    print(y2)

    solve_ivp(y1, y2, 1, 8, 1, 5)

    t = Symbol("t")
    print(red_order(4 / t, 3 / t, t**(-2), 0, t))

    print(Wronskian(t**2, t**3))


if __name__ == "__main__":
    main()
