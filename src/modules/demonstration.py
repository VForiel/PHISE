from IPython.display import display
import sympy as sp

def show():

    # Elements definition
    I = sp.IndexedBase("I", real=True)  # Kernel intensity
    E = sp.IndexedBase("E")  # Electric field
    A = sp.IndexedBase("A", real=True)  # Amplitude
    P = sp.IndexedBase("phi", real=True)  # Relative phase
    T = sp.IndexedBase("theta", real=True)  # Phase perturbation
    a = sp.symbols("a", cls=sp.Idx)  # First dark
    b = sp.symbols("b", cls=sp.Idx)  # Second dark
    s = sp.symbols("s", cls=sp.Idx)  # Star index
    p = sp.symbols("p", cls=sp.Idx)  # Planet index

    # Intensity in a dark output is the sum of the intensities coming from the star and from the planet
    Ia = I[a, s] + I[a, p]
    Ib = I[b, s] + I[b, p]
    print("Input intensities:")
    display(Ia, Ib)

    # Developping Intensities as interference of the electrical fields
    Ias = abs(E[1, s] + E[2, s] + E[3, s] + E[4, s]) ** 2
    Iap = abs(E[1, p] + E[2, p] + E[3, p] + E[4, p]) ** 2
    Ibs = abs(E[1, s] + E[2, s] + E[3, s] + E[4, s]) ** 2
    Ibp = abs(E[1, p] + E[2, p] + E[3, p] + E[4, p]) ** 2

    Ia = Ia.subs(I[a, s], Ias).subs(I[a, p], Iap)
    Ib = Ia.subs(I[b, s], Ibs).subs(I[b, p], Ibp)
    print("Fields contributions:")
    display(Ia, Ib)

    # Expressing the electric fields as a function of the amplitudes and the relative phases
    E1s = A[s]
    E2s = A[s] * (1 + sp.I * T[2])
    E3s = A[s] * (1 + sp.I * T[3])
    E4s = A[s] * (1 + sp.I * T[4])
    E1p = A[p] * sp.exp(sp.I * P[1])
    E2p = A[p] * sp.exp(sp.I * P[2]) * (1 + sp.I * T[2])
    E3p = A[p] * sp.exp(sp.I * P[3]) * (1 + sp.I * T[3])
    E4p = A[p] * sp.exp(sp.I * P[4]) * (1 + sp.I * T[4])

    # Relative phase : E1 -> 0, E2 -> pi, E3 -> pi/2, E4 -> -pi/2
    Ia = (
        Ia.subs(E[1, s], E1s)
        .subs(E[2, s], -E2s)
        .subs(E[3, s], sp.I * E3s)
        .subs(E[4, s], -sp.I * E4s)
    )
    Ia = (
        Ia.subs(E[1, p], E1p)
        .subs(E[2, p], -E2p)
        .subs(E[3, p], sp.I * E3p)
        .subs(E[4, p], -sp.I * E4p)
    )
    # Relative phase : E1 -> 0, E2 -> pi, E3 -> -pi/2, E4 -> pi/2
    Ib = (
        Ib.subs(E[1, p], E1p)
        .subs(E[2, p], -E2p)
        .subs(E[3, p], -sp.I * E3p)
        .subs(E[4, p], sp.I * E4p)
    )
    Ib = (
        Ib.subs(E[1, s], E1s)
        .subs(E[2, s], -E2s)
        .subs(E[3, s], -sp.I * E3s)
        .subs(E[4, s], sp.I * E4s)
    )
    print("Decomposition in amplitudes and phases:")
    display(Ia.expand().simplify(), Ib.expand().simplify())

    # Kernel intensity
    Ik = Ia - Ib
    print("Difference between the signals")
    display(Ik.expand().simplify())