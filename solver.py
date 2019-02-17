import sympy
from sympy import Symbol, E, cos, sin, solve, exp, diff, integrate, sqrt, ln, Matrix, Function, Eq, Integral, Determinant
from sympy.solvers.ode import constantsimp, constant_renumber
import sympy.solvers.ode
from typing import Union, List, Tuple, Dict

number = Union[int, float]
procedure = List[Tuple[str, List[Symbol]]]

__all__ = [
    "find_root", "sec_order_const_coeff", "sec_order_euler", "solve_ivp", "red_order", "Wronskian", "var_parameters", "to_std", "to_general"
]


def first_order_separable():
    pass


def find_root(a: number, b: number, c: number) -> Tuple[Symbol, Symbol]:
    """
    Return the root of the characteristic equation ar^2 + br + c = 0
    """
    disc = sqrt(b**2 - 4*a*c)
    return (-b + disc) / (2*a), (-b - disc) / (2*a)


def sec_order_const_coeff(a: number, b: number, c: number, t: Symbol = Symbol("t")) -> Tuple[Symbol, Symbol, List[Tuple[str, List[Symbol]]]]:
    """
    Solve the second order homogeneous differential equation with constant coefficients a, b and c
    Return the pair of complementary solution.
    """

    r1, r2 = find_root(a, b, c)

    real1, imag1 = r1.as_real_imag()
    real2, imag2 = r2.as_real_imag()

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

    r = Symbol("r")

    procedure = [
        ("\\text{Characteristic equation: }", [Eq(a*r**2 + b*r + c, 0)]),
        ("\\text{Roots: }", [Eq(Symbol("r1"), r1), Eq(Symbol("r2"), r2)]),
        ("\\text{Solutions: }", [y1, y2])
    ]

    return y1, y2, procedure


def sec_order_euler(a: number, b: number, c: number) -> Tuple[Symbol, Symbol, List[Tuple[str, List[Symbol]]]]:
    """
    Solve the second order homogeneous Euler's equation at^2 y'' + bty' + cy = 0
    Return the pair of solutions
    """

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

    r = Symbol("r")
    procedure = [
        ("\\text{Characteristic equation: }",
         [Eq(a*r**2 + (b - a)*r + c, 0)]),
        ("\\text{Roots: }", [Eq(Symbol("r1"), r1), Eq(Symbol("r2"), r2)]),
        ("\\text{Solutions: }", [y1, y2])
    ]

    return y1, y2, procedure


def solve_ivp(y: Symbol, v: List[Tuple[number, number]], t: Symbol = Symbol("t")) -> Tuple[Symbol, Dict[Symbol, number], List[Eq]]:
    """
    Solve the initial value problem given the general solution y

    :param y: the general solution with all arbitrary constants
    :param v: the list of initial conditions [(y(0), t0), (y'(t1), t1), (y''(t2), t2)...]

    :returns: [y with arbitrary constants solved, values of the arbitrary constants, list of equations]
    """
    equations = []

    for i, (t1, y1) in enumerate(v):
        eq = Eq(diff(y, t, i).subs(t, t1), y1)
        equations.append(eq)

    sol = solve(equations)
    for k in sol:
        y = y.subs(k, sol[k])

    return y, sol, equations


def red_order(y1: Symbol, pt: Symbol, qt: Symbol, gt: Symbol, t: Symbol = Symbol("t")) -> Symbol:
    """
    Get the other solution of a second order linear differential equation y'' + p(t)y' + q(t)y = g(t) given a solution y1 by reduction of order.
    :returns: the other solution
    """
    y1p = diff(y1, "t")
    fac = exp(integrate(pt, t))
    mu_t = (y1**2) * fac

    C1, C2 = Symbol("C1"), Symbol("C2")

    if gt == 0:
        vp = C1 / mu_t
    else:
        vp = integrate(y1 * gt * fac, t) / mu_t + C1

    v = integrate(vp, t) + C2
    return constantsimp(v * y1, {C1, C2})


def Wronskian(args: List[Symbol], t: Symbol = Symbol("t")) -> Tuple[Matrix, List[List[Symbol]]]:
    """
    :param args: List of solutions [y1, y2, y3, y4]
    """
    size = len(args)
    m = [
        [diff(args[x], t, i) for i in range(size)] for x in range(size)
    ]
    w = Matrix(m).transpose()

    return w, m


def var_parameters(y: List[Symbol], gt: Symbol, t: Symbol = Symbol("t")) -> Tuple[Symbol, procedure]:
    """
    Solve the particular solution of a nonhomogeneous second order differential equation given its two complementary solutions

    The equation must be in its standard form: y'' + p(t)y' + q(t)y = g(t)

    Return the particular solution
    """
    w, m = Wronskian(y, t)
    W = sympy.simplify(w.det())
    goW = sympy.simplify(gt / W)

    yp = 0

    Wdets = []
    integrals = []

    col = [0] * len(y)
    col[-1] = 1
    for i in range(len(y)):
        Wi = w.copy()
        Wi[:, i] = col.copy()
        Wi_det = Wi.det()

        integrand = Wi_det * goW
        integral = integrate(integrand, t)
        yp += y[i] * integral

        integrals.append(Eq(Integral(integrand, t), integral))
        Wdets.append(Eq(Determinant(Wi), Wi_det))

    yps = sympy.simplify(yp)
    procedure = [
        ('\\text{Compute Wronskian}', [Eq(Determinant(w), W, evaluate=False)]),
        ('\\text{Compute } W_i', Wdets),
        ('\\text{Compute } \\frac{g(t)}{W(t)}',
         [Eq(gt / W, goW, evaluate=False)]),
        ('\\text{Compute the integral of } \\frac{g(t)W_i(t)}{W(t)}', integrals),
        ('\\text{Sum together and simplify }', [Eq(yp, yps, evaluate=False)])
    ]

    return yps, procedure


def to_std(*args: List[Symbol]) -> List[Symbol]:
    """
    Convert a linear ordinary differential equation p(t)yn + q(t)yn-1 + ... + r(t)y = f(t) to standard form

    yn + q(t)/p(t)yn-1 + ... + r(t)/p(t)y = f(t)/p(t)
    """
    pt = args[0]
    assert pt != 0, "the leading coefficient cannot be zero!"
    return [(q / pt) for q in args[1:]]


def to_general(y: List[Symbol], yp: Symbol = 0, consts: List[Symbol] = []) -> Tuple[Symbol, List[Symbol]]:
    """
    Given a list of complementary solutions and a particular solution, give the general solution by

    y(t) = C1*y1(t) + C2*y2(t) + ... + Cn*yn(t) + yp
    """
    num = len(y)

    if len(consts) != num:
        consts = [Symbol('C' + str(i + 1)) for i in range(num)]

    general = yp
    for (y_, C) in zip(y, consts):
        general += C * y_

    return constantsimp(general, set(consts)), consts


def main():
    y1, y2, _ = sec_order_const_coeff(1, 3, 2)
    print(y1)
    print(y2)

    y1, y2, _ = sec_order_euler(1, 11, 25)
    print(y1)
    print(y2)

    y, consts = to_general([y1, y2])

    print(solve_ivp(y, [(1, 8), (1, 5)]))

    t = Symbol("t")
    print(red_order(4 / t, 3 / t, t**(-2), 0, t))

    print(red_order(4 / t, *to_std(t**2, 3*t, 1), 0, t))

    print(Wronskian([t**2, t**3]))

    y1, y2, _ = sec_order_const_coeff(1, -12, 36)
    print(y1, y2)

    yp, _ = var_parameters([y1, y2],
                           7 / t * exp(6 * t)
                           )
    print(to_general([y1, y2], yp))

    y = Function('y')
    print(sympy.dsolve(y(t).diff(t, 2) - 12 *
                       y(t).diff(t) + 36 * y(t) - 7 / t * exp(6*t), y(t)))


if __name__ == "__main__":
    main()
