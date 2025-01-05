import numpy as np
import scipy.linalg as la

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import time

from kifmm2d.scalar.fmm import FMM as KI_FMM
import numba

## Set the parameters

n_boundary = 512
n_quadrature = int(n_boundary * 10)
n_angles = n_boundary
n_radii = n_boundary
a, b = 0.3, 3
plot_contours = False
n_levels = 128
colors = cm.prism

L = 2.0 * np.pi  # angles from 0 to L

# Boudary curve
def r(t):
    xi = (1 + a * np.cos(b * (t + a * np.sin(t)))) * np.cos(t)
    eta = (1 + a * np.cos(b * (t + a * np.sin(t)))) * np.sin(t)
    return xi, eta

# First derivative
def drdt(t):
    dxidt = -a * b * (a * np.cos(t) + 1) * np.sin(b * (a * np.sin(t) + t)) * np.cos(t) - (
                a * np.cos(b * (a * np.sin(t) + t)) + 1) * np.sin(t)
    detadt = -a * b * (a * np.cos(t) + 1) * np.sin(t) * np.sin(b * (a * np.sin(t) + t)) + (
                a * np.cos(b * (a * np.sin(t) + t)) + 1) * np.cos(t)
    return dxidt, detadt

# Second derivative
def d2rdt2(t):
    d2xidt2 = 2 * a * b * (a * np.cos(t) + 1) * np.sin(t) * np.sin(b * (a * np.sin(t) + t)) + a * b * (
                a * np.sin(t) * np.sin(b * (a * np.sin(t) + t)) - b * (a * np.cos(t) + 1) ** 2 * np.cos(
            b * (a * np.sin(t) + t))) * np.cos(t) - (a * np.cos(b * (a * np.sin(t) + t)) + 1) * np.cos(t)
    d2etadt2 = -2 * a * b * (a * np.cos(t) + 1) * np.sin(b * (a * np.sin(t) + t)) * np.cos(t) + a * b * (
                a * np.sin(t) * np.sin(b * (a * np.sin(t) + t)) - b * (a * np.cos(t) + 1) ** 2 * np.cos(
            b * (a * np.sin(t) + t))) * np.sin(t) - (a * np.cos(b * (a * np.sin(t) + t)) + 1) * np.sin(t)
    return d2xidt2, d2etadt2

# Kernel function
def k(t, s):
    if isinstance(s, float):
        if np.abs(t - s) > 1e-15: # When t≠s
            xi_t, eta_t = r(t)
            xi_s, eta_s = r(s)
            dxi_s, deta_s = drdt(s)
            return (deta_s * (xi_t - xi_s) - dxi_s * (eta_t - eta_s)) / (
                        2 * np.pi * ((xi_t - xi_s) ** 2 + (eta_t - eta_s) ** 2))
        else: # When t=s
            dxi_s, deta_s = drdt(s)
            dxi2_s, deta2_s = d2rdt2(s)
            return (dxi2_s * deta_s - deta2_s * dxi_s) / (4 * np.pi * (dxi_s ** 2 + deta_s ** 2))
    else:
        flag = (np.abs(t - s) > 1e-15)
        K = np.zeros(s.shape)
        s1 = s[flag]
        s2 = s[~flag]
        xi_t, eta_t = r(t)
        xi_s1, eta_s1 = r(s1)
        dxi_s1, deta_s1 = drdt(s1)
        K[flag] = (deta_s1 * (xi_t - xi_s1) - dxi_s1 * (eta_t - eta_s1)) / (
                        2 * np.pi * ((xi_t - xi_s1) ** 2 + (eta_t - eta_s1) ** 2))

        dxi_s2, deta_s2 = drdt(s2)
        dxi2_s2, deta2_s2 = d2rdt2(s2)
        K[~flag] = (dxi2_s2 * deta_s2 - deta2_s2 * dxi_s2) / (4 * np.pi * (dxi_s2 ** 2 + deta_s2 ** 2))
        return K


# BC
def f(t):
    xi, eta = r(t)
    return np.exp(eta) * np.cos(xi)

# Sample the angles for the boundary discretization
T1 = time.time()
t, h = np.linspace(0, L, n_boundary, endpoint=False, retstep=True)

# Assemble matrix (dense!)

A = np.zeros((n_boundary, n_boundary))

for i in range(n_boundary):
    for j in range(n_boundary):
        A[i, j] = k(t[i], t[j])

A = -0.5 * np.eye(n_boundary) + h * A

# Assemble right-hand side

f_n = f(t)

# Solve for the approximation of the kernel $\rho_n$.

rho_n = la.solve(A, f_n)


# We can approximate the kernel $\rho(t)$ at any t using
# interpolation.

def rho_int(s):
    # Nystrom interpolation to obtain value at arbitrary s
    K = h * np.dot(k(s, t), rho_n)
    return 2 * (-f(s) + K)

T, H = np.linspace(0, L, n_quadrature, endpoint=False, retstep=True)

if n_quadrature != n_boundary:
    rho = np.array([rho_int(tau) for tau in T]).flatten()
else:
    rho = rho_n

##########################################################################
@numba.njit(fastmath=True)
def Eval(sx, sy, tx, ty):
    dx = tx-sx
    dy = ty-sy
    d2 = dx*dx + dy*dy
    return -np.log(d2)/(4*np.pi)

@numba.njit(fastmath=True)
def Gradient_Eval1(sx, sy, tx, ty, tau, extras):
    dx = tx-sx
    dy = ty-sy
    d2 = dx*dx + dy*dy
    id2 = 1.0/d2
    scale = 1.0/(2*np.pi)
    ux = scale*dx*id2*tau
    uy = scale*dy*id2*tau
    return ux, uy

@numba.njit(fastmath=True)
def Gradient_Eval2(sx, sy, tx, ty, tau, extras):
    dx = tx-sx
    dy = ty-sy
    d2 = dx*dx + dy*dy
    id2 = 1.0/d2
    scale = 1.0/(2*np.pi)
    ux = scale*dx*id2*tau[0]
    uy = scale*dy*id2*tau[0]
    return ux, uy

angles = np.linspace(1e-10, L, n_angles, endpoint=True)

radii_min = 1e-10 * np.ones((n_angles, 1))
radii_max = (1 + a * np.cos(b * (angles + a * np.sin(angles)))).reshape(-1, 1)
radii = np.linspace(radii_min, radii_max, n_radii, endpoint=False).squeeze(2).T
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)

X = (radii * np.cos(angles)).flatten()
Y = (radii * np.sin(angles)).flatten()

Nequiv = 150
N_cutoff = 50

xi, eta = r(T)
dxi, deta = drdt(T)
tau = deta * rho

FMM = KI_FMM(xi, eta, Eval, N_cutoff, Nequiv)
FMM.build_expansions(tau)
FMM.register_evaluator(Gradient_Eval1, Gradient_Eval2, 'gradient', 2)
potential1 = FMM.evaluate_to_points(X, Y, 'gradient', True)[0, :] * H

tau = -dxi * rho
FMM.build_expansions(tau)
FMM.register_evaluator(Gradient_Eval1, Gradient_Eval2, 'gradient', 2)
potential2 = FMM.evaluate_to_points(X, Y, 'gradient', True)[1, :] * H

Z = potential1 + potential2
##########################################################################


## We now plot the solution

def u_exact(x, y):
    return np.exp(y) * np.cos(x)

Z_exact = []
for x, y in zip(X, Y):
    Z_exact.append(u_exact(x, y))

Z = np.array(Z)
Z_exact = np.array(Z_exact)
T2 = time.time()
print(f"The process takes {T2 - T1:.3f} seconds")

X = X.reshape(n_angles, n_radii)
Y = Y.reshape(n_angles, n_radii)
Z = Z.reshape(n_angles, n_radii)
Z_exact = Z_exact.reshape(n_angles, n_radii)
fig = plt.figure(figsize=(6, 5))
plt.contourf(X[::3, ::3], Y[::3, ::3], np.log(np.abs(Z_exact[::3, ::3] - Z[::3, ::3]) + 1e-16) / np.log(10), levels=np.linspace(-17, -1, 17))
plt.axis('equal')
plt.axis('off')
plt.colorbar()

plt.show()
# plt.savefig('./figures/untrivial_FMM.pdf', format = 'pdf', bbox_inches = 'tight')