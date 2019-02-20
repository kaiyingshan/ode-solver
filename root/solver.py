import sympy
from sympy import Symbol, E, cos, sin, solve, exp, diff, integrate, sqrt, ln, Matrix, Function, Eq, Integral, Determinant, collect, simplify, latex, trigsimp, expand, Derivative
from sympy.solvers.ode import constantsimp, constant_renumber
import sympy.solvers.ode
from typing import Union, List, Tuple, Dict

Number = Union[int, float]
Procedure = List[Tuple[str, List[Symbol]]]

__all__ = [
    "find_root", "sec_order_const_coeff", "sec_order_euler", "solve_ivp", "red_order", "Wronskian", "var_parameters", "to_std", "to_general", "display_procedure"
]


def display_procedure(procedure: Procedure) -> None:
    """
    Pretty print the procedure in IPython (Jupyter) notebook
    """
    from IPython.display import display, Math

    for desc, p in procedure:
        display(Math(desc))
        for x in p:
            display(Math(latex(x, ln_notation=True)))


def first_order_separable():
    pass


def first_order_linear():
    pass


def first_order_homogeneous():
    pass


def first_order_autonomous():
    pass


def first_order_exact():
    pass


def first_order_bernoulli():
    pass


def first_order_riccati():
    pass


def _derivative_repr(y: Function = Function('y', real=True), t: Symbol = "t", order: int = 0):
    return y(t).diff(t, order)


def find_root(a: Number, b: Number, c: Number) -> Tuple[Symbol, Symbol]:
    """
    Return the root of the characteristic equation ar^2 + br + c = 0
    """
    disc = sqrt(b**2 - 4*a*c)
    return (-b + disc) / (2*a), (-b - disc) / (2*a)


def sec_order_const_coeff(a: Number, b: Number, c: Number, t: Symbol = Symbol("t")) -> Tuple[Symbol, Symbol, Procedure]:
    """
    Solve the second order homogeneous differential equation with constant coefficients a, b and c
    Return the pair of complementary solution.
    """

    r1, r2 = find_root(a, b, c)

    real1, imag1 = r1.as_real_imag()
    real2, imag2 = r2.as_real_imag()

    imag2 = abs(imag2)

    if imag1 == 0 and imag2 == 0:  # two real roots
        y1 = exp(real1 * t)
        if real1 == real2:  # repeated roots
            y2 = t * exp(real2 * t)
        else:
            y2 = exp(real2 * t)
    else:  # imaginary/complex roots
        y1 = exp(real1 * t) * cos(imag1 * t)
        y2 = exp(real2 * t) * sin(imag2 * t)

    r = Symbol("r")

    procedure = [
        ("\\text{Characteristic equation: }", [Eq(a*r**2 + b*r + c, 0)]),
        ("\\text{Roots: }", [Eq(Symbol("r1"), r1), Eq(Symbol("r2"), r2)]),
        ("\\text{Solutions: }", [Eq(Symbol('y1'), y1), Eq(Symbol('y2'), y2)])
    ]

    return y1, y2, procedure


def sec_order_euler(a: Number, b: Number, c: Number) -> Tuple[Symbol, Symbol, Procedure]:
    """
    Solve the second order homogeneous Euler's equation at^2 y'' + bty' + cy = 0
    Return the pair of solutions
    """

    r1, r2 = find_root(a, b - a, c)

    real1, imag1 = r1.as_real_imag()
    real2, imag2 = r2.as_real_imag()

    imag2 = abs(imag2)

    t = Symbol("t")

    if imag1 == 0 and imag2 == 0:  # two real roots
        y1 = t ** real1
        if real1 == real2:  # repeated roots
            y2 = ln(t) * t ** real2
        else:  # distinct real roots
            y2 = t ** real2
    else:  # imaginary/complex roots
        y1 = (t ** (real1)) * cos(imag1 * ln(t))
        y2 = (t ** (real2)) * sin(imag2 * ln(t))

    r = Symbol("r")
    procedure = [
        ("\\text{Characteristic equation: }",
         [Eq(a*r**2 + (b - a)*r + c, 0)]),
        ("\\text{Roots: }", [Eq(Symbol("r1"), r1), Eq(Symbol("r2"), r2)]),
        ("\\text{Solutions: }", [Eq(Symbol('y1'), y1), Eq(Symbol('y2'), y2)])
    ]

    return y1, y2, procedure


def solve_ivp(y: Symbol, v: List[Tuple[Number, Number]], t: Symbol = Symbol("t")) -> Tuple[Symbol, Procedure]:
    """
    Solve the initial value problem given the general solution y

    :param y: the general solution with all arbitrary constants
    :param v: the list of initial conditions [(y(0), t0), (y'(t1), t1), (y''(t2), t2)...]

    :returns: [y with solved arbitrary constants substituted, procedure]
    """
    equations = []
    derivatives = []

    for i, (t1, y1) in enumerate(v):
        derivative = diff(y, t, i)

        d_simp = simplify(derivative)

        eq = Eq(d_simp.subs(t, t1), y1)
        # derivatives.append(Eq(_derivative_repr(order=i),
        #                       Eq(derivative, d_simp, evaluate=False), evaluate=False))

        derivatives.append(
            Eq(_derivative_repr(order=i), d_simp, evaluate=False))

        equations.append(eq)

    sol = solve(equations)
    for k in sol:
        y = y.subs(k, sol[k])

    procedure = [
        ('\\text{Find successive derivatives of } y(t)', derivatives),
        ('\\text{Substitute the initial conditions}', equations),
        ('\\text{Solve for the arbitrary constants}', [sol]),
        ('\\text{Substitute the solved constants into } y(t)', [y])
    ]

    return y, procedure


def red_order(y1: Symbol, pt: Symbol, qt: Symbol, gt: Symbol, t: Symbol = Symbol("t")) -> Tuple[Symbol, Procedure]:
    """
    Get the other solution of a second order linear differential equation y'' + p(t)y' + q(t)y = g(t) given a solution y1 by reduction of order.
    :returns: the other solution
    """

    v = Function('v')(t)
    y2 = v * y1

    # get the derivatives
    y2pp = simplify(diff(y2, t, 2)).expand()
    y2p = simplify(diff(y2, t, 1)).expand()

    # plug in derivatives and simplify
    expr = y2pp + pt * y2p + y2 * qt
    simp_expr = simplify(expr).expand()

    # now we should have an equation with only v'' and v'
    # use w = v'
    w = Function('w')(t)
    w_expr = simp_expr.subs(v.diff(t, 1), w).expand()

    # convert to the standard form of a first order linear diff eq.
    wp = w.diff(t, 1)
    wp_coeff = w_expr.collect(wp).coeff(wp)

    p = simplify(w_expr / wp_coeff).expand()
    q = simplify(gt / wp_coeff).expand()

    u = exp(integrate(p.coeff(w), t))

    C1 = Symbol('C1')
    C2 = Symbol('C2')

    w_sol = simplify((integrate(q * u, t) + C1) / u)

    # integrate w to find v
    v_sol = integrate(w_sol, t) + C2

    # y2 = v y1
    sol = simplify(y1 * v_sol).expand()
    sol_simp = constantsimp(sol, {C1, C2})

    procedure = [
        ('\\text{Solution }y_2 \\text{ takes the form } y_2 = v(t)y_1', [
            Eq(Symbol('y2'), y2, evaluate=False)
        ]),
        ('\\text{Calculate the derivatives}', [
            Eq(Derivative(y2, t, 1), y2p, evaluate=False),
            Eq(Derivative(y2, t, 2), y2pp, evaluate=False)
        ]),
        ("\\text{Plug in the derivatives and simplify LHS of } y'' + p(t)y' + q(t)y = g(t) \\\\v''(t) \\text{ terms should cancel out}", [
            Eq(Eq(expr, simp_expr, evaluate=False), gt, evaluate=False)
        ]),
        ("\\text{Let } w(t) = v'(t) \\text{ and convert to standard form}", [
            Eq(w_expr, gt, evaluate=False),  Eq(p, q, evaluate=False)
        ]),
        ('\\text{Solve the first order linear differential equation in } w(t)', [
            Eq(w, w_sol, evaluate=False)
        ]),
        ('\\text{Integrate } w(t) \\text{ to solve for} v(t)', [
            Eq(Integral(w_sol, t), v_sol, evaluate=False)
        ]),
        ('\\text{Solve for y2: } y_2 = v(t) y_1(t)', [
            Eq(Symbol('y2'), Eq(sol, sol_simp, evaluate=False), evaluate=False)
        ])
    ]

    return sol_simp, procedure
    # y1p = diff(y1, "t")
    # fac = exp(integrate(pt, t))
    # mu_t = (y1**2) * fac

    # C1, C2 = Symbol("C1"), Symbol("C2")

    # if gt == 0:
    #     vp = C1 / mu_t
    # else:
    #     vp = integrate(y1 * gt * fac, t) / mu_t + C1

    # v = integrate(vp, t) + C2
    # return constantsimp(v * y1, {C1, C2})


def Wronskian(args: List[Symbol], t: Symbol = Symbol("t")) -> Tuple[Determinant, Matrix]:
    """
    :param args: List of complementary solutions [y1, y2, ..., yn]

    :returns: [Wronskian determinant, Wronskian matrix]
    """
    size = len(args)
    w = Matrix([
        [diff(args[x], t, i) for i in range(size)] for x in range(size)
    ]).transpose()

    return trigsimp(simplify(w.det()), deep=True, recursive=True), w


def undetermined_coeffs(y: List[Symbol], gt: Symbol, t: Symbol = Symbol("t")) -> Tuple[Symbol, Procedure]:
    pass


def var_parameters(y: List[Symbol], gt: Symbol, t: Symbol = Symbol("t")) -> Tuple[Symbol, Procedure]:
    """
    Solve the particular solution of a nonhomogeneous second order differential equation given its two complementary solutions using variation of parameters

    The equation must be in its standard form: y'' + p(t)y' + q(t)y = g(t)

    Return the particular solution
    """
    W, w = Wronskian(y, t)
    goW = simplify(gt / W)

    yp = 0

    Wdets = []
    integrals = []

    col = [0] * len(y)
    col[-1] = 1
    for i in range(len(y)):
        Wi = w.copy()
        Wi[:, i] = col.copy()
        Wi_det = simplify(Wi.det())

        integrand = Wi_det * goW
        integral = integrate(integrand, t)
        yp += y[i] * integral

        integrals.append(Eq(Integral(integrand, t), integral))
        Wdets.append(
            Eq(Symbol('W{}'.format(i+1)), Eq(Determinant(Wi), Wi_det), evaluate=False))

    yps = simplify(yp)

    procedure = [
        ('\\text{Compute Wronskian}', [
         Eq(Symbol('W'), Eq(Determinant(w), W, evaluate=False), evaluate=False)]),
        ('\\text{Compute } W_i', Wdets),
        ('\\text{Compute } \\frac{g(t)}{W(t)}',
         [Eq(sympy.Mul(gt, sympy.Pow(W, -1, evaluate=False), evaluate=False), goW, evaluate=False)]),
        ('\\text{Compute } \\int \\frac{g(t)W_i(t)}{W(t)} dt', integrals),
        ('\\text{Compute the sum } \\sum_{i=1}^{k} y_i \\int \\frac{g(t)W_i(t)}{W(t)} dt', [
         Eq(yp, yps, evaluate=False)])
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


def to_general(y: List[Symbol], yp: Symbol = 0, t: Symbol = Symbol("t")) -> Tuple[Symbol, List[Symbol]]:
    """
    Given a list of complementary solutions and a particular solution, give the general solution by

    y(t) = C1*y1(t) + C2*y2(t) + ... + Cn*yn(t) + yp
    """
    num = len(y)

    # if len(consts) != num:
    consts = [Symbol('C' + str(i + 1)) for i in range(num)]

    general = yp
    for (y_, C) in zip(y, consts):
        general += C * y_

    general = sympy.collect(general, t)

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
