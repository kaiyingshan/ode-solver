import sympy
from sympy import Symbol, E, cos, sin, solve, exp, diff, integrate, sqrt, ln, Matrix, Function, Eq, Integral, Determinant, collect, simplify, latex, trigsimp, expand, Derivative, solveset, symbols, FiniteSet, logcombine, separatevars, numbered_symbols, Dummy, Add, roots, Poly, re, im, conjugate, atan2, pprint, pretty, RootOf, Rational
from collections import defaultdict, OrderedDict
from sympy.solvers.ode import constantsimp, constant_renumber, _mexpand, _undetermined_coefficients_match
import sympy.solvers.ode
from typing import Union, List, Tuple, Dict, Any
from sympy.abc import mu

Number = Union[int, float]
t = Symbol('t', real=True)

__all__ = [
   "Procedure", "find_root", "sec_order_euler", "solve_ivp", "red_order", "Wronskian", "variation_of_parameters", "to_std", "to_general", "first_order_separable", "first_order_linear", "first_order_homogeneous", "first_order_autonomous", "first_order_exact", "first_order_bernoulli", "undetermined_coefficients", "nth_order_const_coeff", "t"
]


class Content:
    def __init__(self, content: Any, *args, **kwargs):
        self.content = content
        self.options = kwargs

    def latex(self):
        raise NotImplementedError()

    def plain(self):
        raise NotImplementedError()

    def latex_nl(self, nl=True):
        return ("\\\\[{}pt]".format(self.options.get('vspace', 8)) if nl else "")


class Procedure:

    @staticmethod
    def is_notebook():
        try:
            from IPython import get_ipython
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except:
            return False      # Probably standard Python interpreter

    class Text(Content):
        def __init__(self, content: str, *args, **kwargs):
            options = {
                'nl': kwargs.get('nl', False)
            }
            return super().__init__(content, *args, **options)

        def latex(self):
            # + self.latex_nl(self.options.get('nl', False))
            return "\\text{" + self.content + "}"

        def plain(self):
            return self.content

    class Equation(Content):
        def __init__(self, content: Symbol, *args, **kwargs):
            options = {
                'nl': kwargs.get('nl', True),
                'ln_notation': kwargs.get('ln_notation', True),
                'long_frac_ratio': kwargs.get('long_frac_ratio', 5),
            }
            return super().__init__(content, *args, **options)

        def latex(self):
            # + self.latex_nl(self.options.get('nl', True))
            return latex(self.content, ln_notation=self.options['ln_notation'],
                         long_frac_ratio=self.options['long_frac_ratio'])

        def plain(self):
            return pretty(self.content)

    class EquArray(Content):
        def __init__(self, content: List[Symbol], *args, **kwargs):
            options = {
                'nl': kwargs.get('nl', True),
                'ln_notation': kwargs.get('ln_notation', True),
                'long_frac_ratio': kwargs.get('long_frac_ratio', 5),
            }
            return super().__init__(content, *args, **options)

        def latex(self):
            tex = "\\left\{ \\begin{array}{ll}"
            for item in self.content:
                tex += latex(item, ln_notation=self.options['ln_notation'],
                             long_frac_ratio=self.options['long_frac_ratio']) + "\\\\"
            # + self.latex_nl(self.options.get('nl', True))
            tex += "\\end{array} \\right."
            return tex

        def plain(self):
            return pretty(self.content)

    class EquList(Content):
        def __init__(self, content: List[Symbol], *args, **kwargs):
            options = {
                'nl': kwargs.get('nl', True),
                'ln_notation': kwargs.get('ln_notation', True),
                'long_frac_ratio': kwargs.get('long_frac_ratio', 5),
            }
            return super().__init__(content, *args, **options)

        def latex(self):
            tex = "\\begin{align*}"
            for item in self.content:
                tex += "&" + latex(item, ln_notation=self.options['ln_notation'],
                                   long_frac_ratio=self.options['long_frac_ratio']) + "\\\\"
            return tex + "\\end{align*}"

        def plain(self):
            return pretty(self.content)

    class LaTeX(Content):
        def __init__(self, content: str, *args, **kwargs):
            options = {
                'nl': kwargs.get('nl', False)
            }
            return super().__init__(content, *args, **options)

        def latex(self):
            # + self.latex_nl(self.options.get('nl', False))
            return self.content

        def plain(self):
            # + ("\n" if self.options.get('nl', False) else "")
            return self.content

    def __init__(self):
        self.content = []

    def text(self, text: str, **kwargs):
        self.content.append(self.Text(text, **kwargs))
        return self

    def latex(self, tex: str, **kwargs):
        self.content.append(self.LaTeX(tex, **kwargs))
        return self

    def eq(self, eq: Symbol, **kwargs):
        self.content.append(self.Equation(eq, **kwargs))
        return self

    def equarr(self, eqs: List[Symbol], **kwargs):
        self.content.append(self.EquArray(eqs, **kwargs))
        return self

    def equlist(self, eqs: List[Symbol], **kwargs):
        self.content.append(self.EquList(eqs, **kwargs))
        return self

    def extend(self, other):
        self.content.extend(other.content)
        return self

    def display_ipython(self):
        from IPython.display import display, Math
        tex = ""
        for item in self.content:
            tex += item.latex()
            if item.options['nl']:
                display(Math(tex))
                tex = ""
        if tex != "":
            display(Math(tex))

    def display_terminal(self):
        for item in self.content:
            if type(item) == self.Text or type(item) == self.LaTeX:
                print(item.content, end="")
            elif type(item) == self.EquList or type(item) == self.EquArray:
                for eq in item.content:
                    pprint(eq)
                    print()
            elif type(item) == self.Equation:
                pprint(item.content)
            if item.options['nl']:
                print("\n")

    def display(self):
        if self.is_notebook():
            self.display_ipython()
        else:
            self.display_terminal()


def display(sym):
    from IPython.display import display, Math
    display(Math(latex(sym, ln_notation=True, long_frac_ratio=5)))


def first_order_separable(Y: Symbol, T: Symbol, implicit=True, y: Symbol = Symbol('y'), t: Symbol = t) -> Tuple[Symbol, Procedure]:
    # c = Symbol('c')

    # y_int = integrate(Y, y)
    # t_int = integrate(T, t)
    # procedure = [("\\text{Integrate y and t,}", [Eq(Integral(Y, y), Integral(
    #     T, t), evaluate=False), Eq(y_int, t_int + c, evaluate=False)])]
    # # procedure = [("\\text{Integrate y}", [y_int]), ('\\text{Integrate t}', [t_int])]
    # if implicit:
    #     result = Eq(y_int, t_int + c, evaluate=False)
    #     return result, procedure
    # else:
    #     result = solveset(Eq(y_int, t_int), y)
    #     result_set = FiniteSet()
    #     for i in result:
    #         result_set = result_set + FiniteSet(simplify(i))
    #     return result_set, procedure
    pass


def first_order_linear(pt: Symbol, qt: Symbol, t: Symbol = t, y : Function = Function('y', real=True)(t)) -> Tuple[Symbol, Procedure]:
    mu = exp(integrate(pt, t))

    r = qt * mu

    rhs = integrate(qt * mu, t) + Symbol('C')
    result = simplify(rhs / mu).expand()

    procedure = Procedure()
    procedure.text('Calculate the integrating factor ').latex('\\mu', nl=True)\
            .eq(Eq(Dummy('mu'), Eq(exp(Integral(pt, t)), mu, evaluate=False), evaluate=False))\
            .text('Multiply both sides of the equation by ').latex('\\mu', nl=True)\
            .eq(Eq(((y.diff(t) + y * pt) * mu).expand(), r, evaluate=False))\
            .eq(Eq(Derivative(y * mu, t), r, evaluate=False))\
            .eq(Eq(y * mu, rhs, evaluate=False))\
            .eq(Eq(y, result, evaluate=False))

    return result, procedure


def first_order_homogeneous(F: Symbol, y: Symbol = Symbol('y'), t: Symbol = t):
    # from IPython.display import display
    # v, c = symbols('v c')

    # f_subs = F.subs(y / t, v)

    # f_subs = f_subs.subs(t / y, 1 / v)

    # logcombine(f_subs, force=True)

    # result_separable, p_separable = first_order_separable(
    #     1 / (f_subs - v), 1 / t, y=v)
    # result = result_separable.subs(v, y / t)

    # procedure = Procedure()

    # procedure.text('Substitute')\
    #     .latex('\\frac{y}{t}', nl = False)\
    #     .text('with')\
    #     .latex()

    # procedure = [('\\text{Substitute $\\frac{y}{t}$ with $v$}', [Eq(Derivative(y, t), F, evaluate=False), Eq(v + t * Derivative(v, t), f_subs, evaluate=False)]),
    #              ('\\text{Simplify,}', [Eq(Derivative(v, t), (f_subs - v) / t, evaluate=False)]), ('\\text{Solve the separable differential equation}', [])]
    # procedure.extend(p_separable)
    # procedure.append(('\\text{Replace v with $\\frac{y}{t}$}', [result]))
    # return result, procedure
    pass

def first_order_autonomous(F: Symbol, implicit=True, y: Symbol = Symbol('y'), t: Symbol = t):
    # c = Symbol('c')

    # f_int = integrate(1 / F, y)

    # result = Eq(f_int, t + c, evaluate=False)

    # procedure = [('\\text{Separate variables,}', [
    #               Eq(Integral(1 / F, y), Integral(1, t) + c, evaluate=False)])]

    # if implicit:
    #     return result, procedure
    # else:
    #     result = solveset(Eq(f_int, t + c), y)

    #     for i in result:
    #         result_solved = i
    #     return Eq(y, result_solved, evaluate=False), procedure
    pass


def first_order_exact(M: Symbol, N: Symbol, implicit=True, y: Symbol = Symbol('y'), x: Symbol = Symbol('x')):
    # c = Symbol('c')
    # m, n, my, nx, h_symbol = symbols('M N My Nx h')
    # My = diff(M, y)
    # Nx = diff(N, x)
    # exact = My.equals(Nx)
    # procedure = []

    # if not exact:
    #     # integrating factor
    #     miu_diff_x = (My - Nx) / N
    #     miu_diff_x = simplify(miu_diff_x)

    #     if y in miu_diff_x.free_symbols:

    #         miu_diff_y = (Nx - My) / M
    #         miu_diff_x = simplify(miu_diff_y)

    #         if x in miu_diff_y.free_symbols:
    #             return None, [('\\text{Could not solve}', [])]

    #         else:

    #             rhs = integrate(miu_diff_y, y)
    #             miu = Eq(ln(mu), rhs)

    #             miu_result = exp(rhs)

    #             procedure.extend([
    #                 ('Calculate the integrating factor', [
    #                     Eq(Derivative(mu, y), (Nx - My) * mu / M, evaluate=False)
    #                 ]),
    #                 ('\\text{The integrating factor is}', [miu_result])
    #             ])
    #             pass

    #     else:

    #         rhs = integrate(miu_diff_x, x)
    #         miu = Eq(ln(mu), rhs + c)

    #         miu_result = exp(rhs)

    #         procedure.extend([
    #             ('\\text{Calculate the integrating factor}', [
    #                 Eq(Derivative(mu, x), (My - Nx) * mu / N, evaluate=False)
    #             ]),
    #             ('\\text{The integrating factor is}', [miu_result])
    #         ])

    #     M = M * miu_result
    #     N = N * miu_result

    #     My = diff(M, y)
    #     Nx = diff(N, x)

    #     procedure.append(('\\text{Multiply both sides with the integrating factor,}', [
    #                      Eq(M + N * Derivative(y, x), 0, evaluate=False)]))

    # m_int_x = integrate(M, x)
    # h_diff = N - integrate(My, x)
    # h = integrate(h_diff, y)
    # result = Eq(m_int_x + h, c, evaluate=False)
    # result_simplified = Eq(simplify(m_int_x + h), c, evaluate=False)
    # procedure.extend([
    #     ('\\text{Determine if the equation is exact,}', [
    #         Eq(m, M, evaluate=False),
    #         Eq(n, N, evaluate=False),
    #         Eq(my, My, evaluate=False),
    #         Eq(nx, Nx, evaluate=False)
    #     ]),

    #     ('\\text{The equation is exact,}',
    #      [Eq(My, Nx, evaluate=False)]),

    #     ('\\text{Integrate $M$ with respect to $x$,}', [m_int_x]),

    #     ('\\text{Derive $h(y)$,}', [
    #         Eq(Derivative(h_symbol, y), h_diff, evaluate=False), Eq(h_symbol, h)
    #     ]),

    #     ('\\text{The solution is,}', [result])

    # ])

    # return result_simplified, procedure
    pass


def first_order_bernoulli(pt: Symbol, qt: Symbol, n: Number, t : Symbol = t):
    procedure = Procedure()
    y = Function('y', real=True)(t)
    v = Function('v', real=True)(t)

    vp = v.diff(t)
    yp = y ** n / (1 - n) * vp

    procedure.text('Use the substitution ').latex('v = y^{1-n}').text(', ').latex('y = v y^n', nl=True)\
    .eq(Eq(Function('v')(t).diff(t), (y ** (1 - n)).diff(t), evaluate=False), nl=False).latex('\\Rightarrow')\
    .eq(Eq(y.diff(t), yp, evaluate=False))\
    .text('Substitute ').latex("y'(t)").text(' into the original equation', nl=True)\
    .eq(Eq(yp + pt * y, qt * y ** n, evaluate=False))\
    .text('Divide both sides by ').latex('\\frac{y^{n}}{n - 1}', nl=True)\
    .eq(Eq(vp + (1 - n) * pt * y ** (1 - n), (1 - n) * qt, evaluate=False))\
    .text('Note that ').latex('v = y^{1-n}', nl=True)\
    .eq(Eq(vp + (1 - n) * pt * v, (1 - n) * qt, evaluate=False))\
    .text('Now we need to solve the linear ODE in ').latex('v', nl=True)

    res, _ = first_order_linear((1 - n) * pt, (1 - n) * qt, t, v)
    result = simplify(res ** (Rational(1, 1 - n)))

    procedure.extend(_)\
        .text('Use ').latex('y = v^{\\frac{1}{1 - n}}').text(' to solve for ').latex('y', nl=True)\
        .eq(Eq(y, result, evaluate=False))

    return result, procedure

def first_order_riccati():
    pass


def find_root(a: Number, b: Number, c: Number) -> Tuple[Symbol, Symbol]:
    """
    Return the root of the characteristic equation ar^2 + br + c = 0
    """
    disc = sqrt(b**2 - 4*a*c)
    return (-b + disc) / (2*a), (-b - disc) / (2*a)


def nth_order_const_coeff(*coeffs: List[Symbol], t: Symbol = t) -> Tuple[List[Symbol], Procedure]:
    """
    Solve a nth order homogeneous linear diff eq with constant coefficients by solving the characteristic equation
    Modified from sympy's source code
    https://github.com/sympy/sympy/blob/master/sympy/solvers/ode.py

    :param coeffs: the list of constant coefficients
    :returns: [list of (linearly independent) solutions, procedure]
    """

    # First, set up characteristic equation.
    char_eq_r, r = sympy.S.Zero, Dummy('r')

    for order, coeff in enumerate(coeffs[::-1]):
        char_eq_r += coeff * r ** order

    char_eq = Poly(char_eq_r, r)

    # Can't just call roots because it doesn't return rootof for unsolveable
    # polynomials.
    char_eq_roots = roots(char_eq, multiple=True)

    root_dict = defaultdict(int) # type: Dict[int, int]

    conjugate_roots = []
    for root in char_eq_roots:
        root_dict[root] += 1

    sols = []
    for root, mult in root_dict.items():
        for i in range(mult):
            if isinstance(root, RootOf):
                sols.append(t**i * exp(root*t))
            elif root.is_real:
                sols.append(t**i*exp(root*t))
            else:
                if root in conjugate_roots:
                    continue
                reroot = re(root)
                imroot = im(root)
                conjugate_roots.append(conjugate(root))
                sols.append(t**i*exp(reroot*t) * sin(abs(imroot) * t))
                sols.append(t**i*exp(reroot*t) * cos(imroot * t))

    # collect roots for display
    p_roots = []
    count = 1
    for root, mult in root_dict.items():
        p_roots.append(Eq(Dummy('r_{}'.format(
            ",".join([str(i) for i in range(count, count + mult)]))), root, evaluate=False))
        count += mult

    procedure = Procedure()
    procedure\
        .text('Characteristic equation: ', nl=True)\
        .eq(Eq(char_eq_r, 0, evaluate=False))\
        .text('Roots: ')\
        .equarr(p_roots)\
        .text('General Solution: ', nl=True)\
        .eq(Eq(Dummy('y'), to_general(sols)[0], evaluate=False))

    return sols, procedure


def sec_order_euler(a: Number, b: Number, c: Number, t: Symbol = t) -> Tuple[List[Symbol], Procedure]:
    """
    Solve the second order homogeneous Euler's equation at^2 y'' + bty' + cy = 0
    Return the pair of solutions
    """

    r1, r2 = find_root(a, b - a, c)

    real1, imag1 = r1.as_real_imag()
    real2, imag2 = r2.as_real_imag()

    imag2 = abs(imag2)

    if imag1 == 0 and imag2 == 0:  # two real roots
        y1 = t ** real1
        if real1 == real2:  # repeated roots
            y2 = ln(t) * t ** real2
        else:  # distinct real roots
            y2 = t ** real2
    else:  # imaginary/complex roots
        y1 = (t ** (real1)) * cos(imag1 * ln(t))
        y2 = (t ** (real2)) * sin(imag2 * ln(t))

    r = Dummy("r")

    procedure = Procedure()
    procedure\
        .text('Characteristic equation: ')\
        .eq(Eq(a*r**2 + (b - a)*r + c, 0))\
        .text('Roots: ')\
        .equarr([Eq(Dummy("r1"), r1), Eq(Dummy("r2"), r2)])\
        .text('General solution: ')\
        .eq(Eq(Dummy('y'), to_general([y1, y2])[0], evaluate=False))

    return [y1, y2], procedure


def solve_ivp(y: Symbol, v: List[Tuple[Number, Number]], t: Symbol = t, func: Function = Function('y', real=True)(t)) -> Tuple[Symbol, Procedure]:
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
        derivatives.append(Eq(func.diff(t, i), d_simp, evaluate=False))
        equations.append(eq)

    sol = solve(equations)
    for k in sol:
        y = y.subs(k, sol[k])

    procedure = Procedure()
    procedure\
        .text('Find successive derivatives of ').latex('y(t)', nl=True)\
        .equlist(derivatives)\
        .text('Substitute the initial conditions', nl=True)\
        .equarr(equations)\
        .text('Solve for the arbitrary constants', nl=True)\
        .equarr([Eq(k, v, evaluate=False) for k, v in sol.items()])\
        .text('Substitute the solved constants into ').latex('y(t)', nl=True)\
        .eq(Eq(Dummy('y'), y, evaluate=False))

    return y, procedure


def undetermined_coefficients(gensols: List[Symbol], func_coeffs: List[Symbol], gt: Symbol, t: Symbol = t) -> Tuple[Symbol, Procedure]:
    """
    Solve a linear diff eq with const coefficients using the method of undetermined coefficients
    Modified from sympy's source code to print out procedure

    :param gensols: a list of general solutions
    :param func_coeffs: a list of constant coefficients (a1 yn + a2 yn-1 + ... + an-1 y' + an y = g(t)
    :param gt: The RHS of the diff eq.
    """

    Y = Function('Y', real=True)(t)

    coeffs = numbered_symbols('A', cls=Dummy)
    coefflist = []

    trialset = _undetermined_coefficients_match(gt, t)['trialset']

    notneedset = set()

    mult = 0
    for i, sol in enumerate(gensols):
        check = sol
        if check in trialset:
            # If an element of the trial function is already part of the
            # homogeneous solution, we need to multiply by sufficient x to
            # make it linearly independent.  We also don't need to bother
            # checking for the coefficients on those elements, since we
            # already know it will be 0.
            while True:
                if check*t**mult in trialset:
                    mult += 1
                else:
                    break
            trialset.add(check*t**mult)
            notneedset.add(check)

    newtrialset = trialset - notneedset

    # while True:
    #     dependent = False
    #     for trial in newtrialset:
    #         if trial in gensols:
    #             dependent = True
    #             break
    #     if not dependent:
    #         break
    #     newtrialset = set([t*trial for trial in trialset])

    # trialset = trialset.union(newtrialset)

    trialfunc = sympy.Number(0)
    for i in newtrialset:
        c = next(coeffs)
        coefflist.append(c)
        trialfunc += c*i

    derivatives = []

    eqs = 0
    for order, coeff in enumerate(func_coeffs[::-1]):
        deriv = simplify(trialfunc.diff(t, order))
        derivatives.append(
            Eq(Derivative(Y, t, order), deriv, evaluate=False))
        eqs += coeff * deriv

    coeffsdict = dict(list(zip(trialset, [0]*(len(trialset) + 1))))

    eqs_lhs = eqs

    eqs = _mexpand(simplify(eqs - gt).expand())

    for i in Add.make_args(eqs):
        s = separatevars(i, dict=True, symbols=[t])
        coeffsdict[s[t]] += s['coeff']

    coeffvals = solve(list(coeffsdict.values()), coefflist)

    if not coeffvals:
        print(
            "Could not solve `%s` using the "
            "method of undetermined coefficients "
            "(unable to solve for coefficients)." % eqs)

    psol = trialfunc.subs(coeffvals)

    procedure = Procedure()
    procedure\
        .text('Find ').latex('Y(t)').text(' that mimics the form of ').latex('g(t)', nl=True)\
        .eq(Eq(Y, trialfunc, evaluate=False))\
        .text('Compute successive derivatives of ').latex('Y(t)', nl=True)\
        .equlist(derivatives)\
        .text('Plug the derivatives into the LHS and equate coefficients', nl=True)\
        .equlist([Eq(eqs_lhs, gt, evaluate=False),
                  Eq(simplify(eqs_lhs).expand().collect(t), gt, evaluate=False)])\
        .equarr([Eq(a, 0, evaluate=False) for a in coeffsdict.values()])\
        .text('Solve for the undetermined coefficients', nl=True)\
        .equarr([Eq(k, v, evaluate=False)
                 for k, v in coeffvals.items() if k != 0] if len(coeffvals) > 0 else [])\
        .text('Substitute the coefficients to get the particular solution', nl=True)\
        .eq(Eq(Dummy('y_p'), psol, evaluate=False))

    return psol, procedure


def red_order(y1: Symbol, pt: Symbol, qt: Symbol, gt: Symbol, t: Symbol = t) -> Tuple[Symbol, Procedure]:
    """
    Get the other solution of a second order linear differential equation y'' + p(t)y' + q(t)y = g(t) given a solution y1 that solves the homogeneous case by reduction of order.

    :returns: the other solution
    """

    y1p = diff(y1, t, 1)
    y1pp = diff(y1, t, 2)

    v = Function('v')(t)
    y2 = v * y1

    # get the derivatives
    y2pp = diff(y2, t, 2).expand()
    y2p = diff(y2, t, 1).expand()

    # plug in derivatives and simplify
    expr = (y2pp + pt * y2p + y2 * qt).expand().collect(v)

    # note that y1 should solve the homogeneous case
    simp_expr = expr.subs(y1pp + pt*y1p + qt*y1,
                          0).replace(y1pp + pt*y1p + qt*y1, 0)

    # now we should have an equation with only v'' and v'
    # use w = v'
    w = Function('w')(t)
    wp = w.diff(t, 1)
    w_expr = simp_expr.subs(v.diff(t, 1), w).expand().collect(w)

    # convert to the standard form of a first order linear diff eq.
    wp_coeff = w_expr.collect(wp).coeff(wp)

    p = (w_expr / wp_coeff).expand().collect(w)
    q = (gt / wp_coeff).expand()

    u = exp(integrate(p.coeff(w), t))

    C1 = Symbol('C1')
    C2 = Symbol('C2')

    w_sol = simplify((integrate(simplify(q * u).expand(), t) + C1) / u)

    # integrate w to find v
    v_sol = integrate(w_sol, t) + C2

    # y2 = v y1
    sol = (y1 * v_sol).expand()
    sol_simp = constantsimp(sol, {C1, C2})

    procedure = Procedure()
    procedure\
        .text('Solution ').latex('y_2').text(' takes the form ').latex('y_2 = v(t)y_1', nl=True)\
        .eq(Eq(Symbol('y2'), y2, evaluate=False))\
        .text('Calculate the derivatives', nl=True)\
        .equlist([
            Eq(Derivative(y2, t, 1), y2p, evaluate=False),
            Eq(Derivative(y2, t, 2), y2pp, evaluate=False)
        ])\
        .text('Plug in the derivatives and simplify LHS of ').latex("y'' + p(t)y' + q(t)y = g(t)", nl=True)\
        .equlist([Eq(y2pp + pt * y2p + y2 * qt, gt, evaluate=False),
                  Eq(expr, gt, evaluate=False)])\
        .text('Given that ').latex('y_1(t)').text(' satisfies the homogeneous equation ')\
        .latex(" y_1''(t) + p(t)y_1' + q(t)y_1 = 0", nl=True)\
        .eq(Eq(simp_expr, gt, evaluate=False))\
        .text('Let ').latex("w(t) = v'(t)").text(' and convert to standard form', nl=True)\
        .equlist([Eq(w_expr, gt, evaluate=False),  Eq(p, q, evaluate=False)])\
        .text('Solve the first order linear differential equation in ').latex('w(t)', nl=True)\
        .eq(Eq(w, w_sol, evaluate=False))\
        .eq(Eq(Eq(Dummy('v(t)'), Integral(w_sol, t), evaluate=False), v_sol, evaluate=False))\
        .eq(Eq(Symbol('y2'), Eq(sol, sol_simp, evaluate=False), evaluate=False))

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


def Wronskian(args: List[Symbol], t: Symbol = t) -> Tuple[Determinant, Matrix]:
    """
    :param args: List of complementary solutions [y1, y2, ..., yn]

    :returns: [Wronskian determinant, Wronskian matrix]
    """
    size = len(args)
    w = Matrix([
        [diff(args[x], t, i) for i in range(size)] for x in range(size)
    ]).transpose()

    return trigsimp(simplify(w.det()), deep=True, recursive=True), w


def variation_of_parameters(y: List[Symbol], gt: Symbol, t: Symbol = t, do_integral=True) -> Tuple[Symbol, Procedure]:
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

        # reduce cos^2 t + sin^2 t to 1
        Wi_det = trigsimp(simplify(Wi.det()), deep=True, recursive=True)

        integrand = (Wi_det * goW).expand()
        integral = integrate(
            integrand, t) if do_integral else Integral(integrand, t)
        yp += y[i] * integral

        if do_integral:
            integrals.append(
                Eq(Dummy('mu_{}'.format(i + 1)),
                   Eq(Integral(integrand, t), integral, evaluate=False), evaluate=False)
            )
        else:
            integrals.append(Eq(Dummy('mu_{}'.format(i)),
                                Integral(integrand, t), evaluate=False))

        Wdets.append(
            Eq(Symbol('W{}'.format(i+1)), Eq(Determinant(Wi), Wi_det, evaluate=False), evaluate=False))

    yps = logcombine(simplify(yp))

    procedure = Procedure()
    procedure\
        .text('Compute the Wronskian determinant', nl=True)\
        .eq(Eq(Dummy('W'), Eq(Determinant(w), W, evaluate=False), evaluate=False))\
        .text('Compute ').latex('W_i', nl=True)\
        .equlist(Wdets)\
        .text('Calculate and simplify ').latex('\\frac{g(t)}{W(t)}', nl=True)\
        .eq(Eq(sympy.Mul(gt, sympy.Pow(W, -1, evaluate=False), evaluate=False), goW, evaluate=False))\
        .text('Compute ').latex('\\mu_i = \\int \\frac{g(t)W_i(t)}{W(t)} dt', nl=True)\
        .equlist(integrals)\
        .text('Compute the sum ').latex('\\sum_{i=1}^{k} y_i \\int \\frac{g(t)W_i(t)}{W(t)} dt', nl=True)\
        .equlist([
            Eq(Dummy('y_p'), yp, evaluate=False),
            Eq(Dummy('y_p'), yps, evaluate=False)
        ])\
        .text('Complementray + particular = general', nl=True)\
        .eq(Eq(Dummy('y'), to_general(y, yps)[0], evaluate=False))

    return yps, procedure


def to_std(*args: List[Symbol]) -> List[Symbol]:
    """
    Convert a linear ordinary differential equation p(t)yn + q(t)yn-1 + ... + r(t)y = f(t) to standard form

    yn + q(t)/p(t)yn-1 + ... + r(t)/p(t)y = f(t)/p(t)
    """
    pt = args[0]
    assert pt != 0, "the leading coefficient cannot be zero!"
    return [(q / pt) for q in args[1:]]


def to_general(y: List[Symbol], yp: Symbol = 0, t: Symbol = t, constant_prefix: str = "C") -> Tuple[Symbol, List[Symbol]]:
    """
    Given a list of complementary solutions and a particular solution, give the general solution by

    y(t) = C1*y1(t) + C2*y2(t) + ... + Cn*yn(t) + yp
    """

    const_iter = numbered_symbols(prefix=constant_prefix, start=1)
    consts = []

    general = yp
    for y_ in y:
        const = next(const_iter)
        consts.append(const)
        general += const * y_
        general = constantsimp(general.collect(y_), consts)

    return general, consts


def main():
    from sympy import init_printing, pprint
    init_printing(use_latex=True)

    a, b, c, d = symbols('a b c d', real=True)
    # nth_order_const_coeff(a, b, c, d)
    sols, _ = nth_order_const_coeff(1, 4, 20, 64, 64, 0, 0)

    sols, _ = sec_order_euler(1, 11, 25)

    y, consts = to_general(sols)

    sol, p = solve_ivp(y, [(1, 8), (1, 5)])
    p.display()

    sol, p = red_order(4 / t, *to_std(t**2, 3*t, 1), 0)
    p.display()

    yp, _ = variation_of_parameters(sols, ln(t))
    _.display()



if __name__ == "__main__":
    pass