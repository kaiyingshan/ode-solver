# ode-solver

An elementary ODE solver with detailed procedure, developed by Hanzhi Zhou and Kaiying Shana

## Supported ODEs

- first order linear
- first order bernoulli
- nth order homogeneous linear ode with constant coefficients
- second order Euler
- variation of parameters for nth order ode
- undetermined coefficients for nth order linear ode with constant coefficients
- 2x2 and 3x3 system of linear odes with constant coefficients
- variation of parameters for 2x2 and 3x3 system of odes
- phase portrait for 2x2 system of linear odes with constant coefficients

> Note 1: If I have time, I'll manage to complete the first order part.

> Note 2: The Laplace transform (and its inverse transform) with procedure is not easy to implement. I may or may not be able to do that.

## Getting started

> Note: Due to time constraints, the code is not well documented

### Requirements

- Python >= 3.5
- SymPy >= 1.3
- Jupyter
- NumPy (for phase portrait)
- Matplotlib (for phase portrait)

```bash
git clone https://github.com/kaiyingshan/ode-solver
pip install sympy jupyter numpy matplotlib
cd ode-solver
jupyter notebook # launch the jupyter notebook server
```

In order to use our solver, you need some basic knowledge of Python and Jupyter notebook.
There are a plenty of examples under the [notebooks](notebooks/) folder.
