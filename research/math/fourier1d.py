"""1D Fourier model of CORVID's p-two-level method: hierarchical P2 = P1 (vertex) + element bubbles,
Helmholtz -u'' - k^2 u on a uniform periodic mesh of size h. Symbols derived with sympy.

Per Fourier mode theta the operator is the 2x2 Hermitian symbol  K(theta) = [[a, b], [conj(b), c]]
(vertex row, bubble row). Coarse = P1 (vertex) rows.

  injection Galerkin coarse (CORVID 0.2.0):  I - Pi = [[0, -b/a], [0, 1]]
  ideal / Schur coarse (proposed):            I - Pi = [[0, 0], [conj(b)/c, 1]],  Kc = S = a - |b|^2 / c

The vertex error created from a bubble error is amplified by |b/a| (injection) vs |b/c| (Schur).
a vanishes at the P1 discrete wavenumber, S at the P2 one: the mismatch is the coarse dispersion error.
"""
import sympy as sy
import numpy as np

x, h, k, th = sy.symbols("x h k theta", positive=True)
t = x / h
phi0, phi1, beta = 1 - t, t, 4 * t * (1 - t)
el = lambda f, g: sy.integrate(sy.diff(f, x) * sy.diff(g, x) - k**2 * f * g, (x, 0, h))
A_hh, A_h0h1 = sy.simplify(el(phi0, phi0)), sy.simplify(el(phi0, phi1))
A_hb, A_bb = sy.simplify(el(phi0, beta)), sy.simplify(el(beta, beta))
print("element entries: hat-hat", A_hh, "| hat-hat'", A_h0h1, "| hat-bubble", A_hb, "| bubble-bubble", A_bb)
# Fourier symbols: vertex j sees elements j-1, j; bubble of element j touches vertices j, j+1
a = sy.simplify(2 * A_hh + 2 * A_h0h1 * sy.cos(th))
b = A_hb * (1 + sy.exp(sy.I * th))          # vertex <- bubble coupling (two elements)
c = A_bb
S = sy.simplify(a - sy.simplify(b * sy.conjugate(b)) / c)
print("a(theta) =", a)
print("c =", c)
print("S(theta) =", sy.simplify(S))

fa, fb, fc = (sy.lambdify((k, h, th), e, "numpy") for e in (a, b, c))
# discrete dispersion: theta at which a = 0 (P1) and S = 0 (P2); exact theta = k h
ths = np.linspace(1e-4, np.pi, 200001)
print("\n  k h  edges/lam  theta_P1/kh  theta_P2/kh  max|b/a|  max|b/c|  band(|b/a|>1)/pi")
for khv in (0.25, 0.5, 1.0, 1.5, 2.0, 2.5):
    A_, B_, C_ = fa(khv, 1.0, ths), fb(khv, 1.0, ths), fc(khv, 1.0, ths) * np.ones_like(ths)
    S_ = A_ - abs(B_) ** 2 / C_
    z1 = ths[np.argmin(abs(A_))]; z2 = ths[np.argmin(abs(S_))]
    ratio = abs(B_ / A_)
    print(f"  {khv:4.2f}  {2*np.pi/khv:8.1f}  {z1/khv:10.4f}  {z2/khv:10.5f}  {ratio.max():9.1e}  {np.max(abs(B_/C_)):7.3f}"
          f"  {np.mean(ratio > 1):.4f}")
