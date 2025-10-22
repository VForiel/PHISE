"""Module generated docstring."""
from IPython.display import display
import sympy as sp

def show():
    """"show.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    I = sp.IndexedBase('I', real=True)
    E = sp.IndexedBase('E')
    A = sp.IndexedBase('A', real=True)
    P = sp.IndexedBase('phi', real=True)
    T = sp.IndexedBase('theta', real=True)
    a = sp.symbols('a', cls=sp.Idx)
    b = sp.symbols('b', cls=sp.Idx)
    s = sp.symbols('s', cls=sp.Idx)
    p = sp.symbols('p', cls=sp.Idx)
    Ia = I[a, s] + I[a, p]
    Ib = I[b, s] + I[b, p]
    print('Input intensities:')
    display(Ia, Ib)
    Ias = abs(E[1, s] + E[2, s] + E[3, s] + E[4, s]) ** 2
    Iap = abs(E[1, p] + E[2, p] + E[3, p] + E[4, p]) ** 2
    Ibs = abs(E[1, s] + E[2, s] + E[3, s] + E[4, s]) ** 2
    Ibp = abs(E[1, p] + E[2, p] + E[3, p] + E[4, p]) ** 2
    Ia = Ia.subs(I[a, s], Ias).subs(I[a, p], Iap)
    Ib = Ia.subs(I[b, s], Ibs).subs(I[b, p], Ibp)
    print('Fields contributions:')
    display(Ia, Ib)
    E1s = A[s]
    E2s = A[s] * (1 + sp.I * T[2])
    E3s = A[s] * (1 + sp.I * T[3])
    E4s = A[s] * (1 + sp.I * T[4])
    E1p = A[p] * sp.exp(sp.I * P[1])
    E2p = A[p] * sp.exp(sp.I * P[2]) * (1 + sp.I * T[2])
    E3p = A[p] * sp.exp(sp.I * P[3]) * (1 + sp.I * T[3])
    E4p = A[p] * sp.exp(sp.I * P[4]) * (1 + sp.I * T[4])
    Ia = Ia.subs(E[1, s], E1s).subs(E[2, s], -E2s).subs(E[3, s], sp.I * E3s).subs(E[4, s], -sp.I * E4s)
    Ia = Ia.subs(E[1, p], E1p).subs(E[2, p], -E2p).subs(E[3, p], sp.I * E3p).subs(E[4, p], -sp.I * E4p)
    Ib = Ib.subs(E[1, p], E1p).subs(E[2, p], -E2p).subs(E[3, p], -sp.I * E3p).subs(E[4, p], sp.I * E4p)
    Ib = Ib.subs(E[1, s], E1s).subs(E[2, s], -E2s).subs(E[3, s], -sp.I * E3s).subs(E[4, s], sp.I * E4s)
    print('Decomposition in amplitudes and phases:')
    display(Ia.expand().simplify(), Ib.expand().simplify())
    Ik = Ia - Ib
    print('Difference between the signals')
    display(Ik.expand().simplify())