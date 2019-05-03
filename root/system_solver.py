import sympy
from sympy import Symbol, E, cos, sin, solve, exp, diff, integrate, sqrt, ln, Matrix, Function, Eq, Integral, Determinant, collect, simplify, latex, trigsimp, expand, Derivative, solveset, symbols, FiniteSet, logcombine, separatevars, numbered_symbols, Dummy, Add, roots, Poly, re, im, conjugate, atan2, pprint, pretty, RootOf, Rational, eye, I
from collections import defaultdict, OrderedDict
from sympy.solvers.ode import constantsimp, constant_renumber, _mexpand, _undetermined_coefficients_match
import sympy.solvers.ode
from typing import Union, List, Tuple, Dict, Any
from sympy.abc import mu
from .solver import *


def system(coeffs: List[List[Union[Symbol, int]]], t: Symbol = Symbol('t', real=True)):
    matrix = Matrix(coeffs)
    procedure = Procedure()
    ident = eye(matrix.rows)
    lam = Symbol('lambda')
    char_eq = simplify((matrix - lam * ident).det())

    procedure\
        .text('characteristic equation: ', nl=True)\
        .eq(Eq(char_eq, 0, evaluate=False))

    rts = roots(char_eq, lam)

    procedure.text('Eigenvalues and eigenvectors', nl=True)

    eigenvects = matrix.eigenvects()
    count = 1
    consts = numbered_symbols('C', Dummy, 1)
    sols = []
    conj_roots = []
    for eigenval, mult, eigenvec in eigenvects:

        if mult != len(eigenvec):  # repeated eigenvectors
            procedure.latex('\\lambda_{%s} = %s\\,\\,' %
                            (",".join(map(lambda x: str(x), range(count, mult + count))), eigenval), nl=True)
            procedure.eq(Eq(Dummy('v'), eigenvec[0], evaluate=False))

            vec_syms = symbols('a0:{}'.format(matrix.rows))
            generalized_eigenvec = Matrix(vec_syms)

            result = solve((matrix - eigenval * ident) *
                           Matrix(generalized_eigenvec) - eigenvec[0], generalized_eigenvec)

            free_vars = list(vec_syms)

            for var in result:
                if var in free_vars:
                    free_vars.remove(var)
                generalized_eigenvec = generalized_eigenvec.subs(
                    var, result[var])
            for i, var in enumerate(free_vars):  # use 0, 1... for free variables
                generalized_eigenvec = generalized_eigenvec.subs(var, i)

            procedure.eq(
                Eq(Dummy('w'), generalized_eigenvec, evaluate=False))

            sols.append(
                ['gen', exp(eigenval * t), eigenvec[0], generalized_eigenvec])

        else:
            procedure.latex('\\lambda_{} = {}'.format(
                count, eigenval), nl=True)
            for i in range(mult):
                procedure.eq(Eq(Dummy('v'), eigenvec[i], evaluate=False))
                if not eigenval.is_real:
                    if eigenval in conj_roots:
                        continue
                    real, imag = eigenval.as_real_imag()
                    real_vec, imag_vec = (
                        eigenvec[i] * expand(exp(imag*I*t), complex=True)).as_real_imag()
                    sols.append(['comp', exp(real * t), real_vec, imag_vec])

                    # we don't need the conjugate
                    conj_roots.append(conjugate(eigenval))
                else:
                    sols.append(['real', exp(eigenval * t), eigenvec[i]])

        count += mult

    procedure.text('General solution: ', nl=True)
    procedure.latex('\\vec{\\mathbf{x}} = ')
    for i in range(len(sols)):
        sol = sols[i]
        if sol[0] == 'real':
            procedure.eq(next(consts), nl=False).eq(
                sol[1], nl=False).eq(sol[2], nl=False)
        elif sol[0] == 'gen':
            procedure.eq(next(consts), nl=False).eq(sol[1], nl=False)\
                .latex('\\left(').eq(sol[2], nl=False).latex('t + ')\
                .eq(sol[3], nl=False).latex('\\right)')
        elif sol[0] == 'comp':
            procedure.eq(sol[1], nl=False)\
                .latex('\\left(').eq(next(consts), nl=False).eq(sol[2], nl=False).latex(' + ')\
                .eq(next(consts), nl=False).eq(sol[3], nl=False).latex('\\right)')

        if i != len(sols) - 1:
            procedure.latex('+')

    return procedure


if __name__ == "__main__":
    system([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])

    system([
        [1, 1, 1],
        [2, 1, -1],
        [0, -1, 1]
    ])
