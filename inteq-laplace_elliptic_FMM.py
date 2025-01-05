import numpy as np
import scipy.linalg as la

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import time

from kifmm2d.scalar.fmm import FMM as KI_FMM
import numba

# Set the parameters

n_boundary = 1024
n_quadrature = int(n_boundary * 10)
n_angles = n_boundary
n_radii = n_boundary
min_radius = 1e-10
max_radius = 0.99
n_levels = 128
colors = cm.prism


L = 2.0 * np.pi  # angles from 0 to L

# ellipse boundary

a = 2  # half major axis
b = 1  # half minor axis

def r(t):
    # ellipse boundary
    x = a * np.cos(t)
    y = b * np.sin(t)
    return x, y


def k(r, s):
    # kernel
    theta = (r+s)/2.0
    cost = np.cos(theta)
    sint = np.sin(theta)
    sig = np.sqrt(a**2.0 * sint**2.0 + b**2.0 * cost**2.0)
    return -1/(2*np.pi) * a * b / (2.0 * sig**2.0)


# Dirichlet BC
def f(r):
    return np.exp(b * np.sin(r)) * np.cos(a * np.cos(r))

T1 = time.time()
# Sample the angles for the boundary discretization

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
    K = h*np.dot(k(s, t), rho_n)
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

radii = np.linspace(min_radius, max_radius, n_radii)

angles = np.linspace(1e-15, L, n_angles, endpoint=True)
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)

# Target points
X = a * (radii * np.cos(angles)).flatten()
Y = b * (radii * np.sin(angles)).flatten()

Nequiv = 150
N_cutoff = 50

tau = b * np.cos(T) * rho
FMM = KI_FMM(a * np.cos(T), b * np.sin(T), Eval, N_cutoff, Nequiv)
FMM.build_expansions(tau)
FMM.register_evaluator(Gradient_Eval1, Gradient_Eval2, 'gradient', 2)
potential1 = FMM.evaluate_to_points(X, Y, 'gradient', True)[0, :] * H

tau = a * np.sin(T) * rho
FMM.build_expansions(tau)
FMM.register_evaluator(Gradient_Eval1, Gradient_Eval2, 'gradient', 2)
potential2 = FMM.evaluate_to_points(X, Y, 'gradient', True)[1, :] * H

Z = potential1 + potential2
##########################################################################

def u_exact(x, y):
    return np.exp(y) * np.cos(x)

Z_exact = []
for x, y in zip(X, Y):
    Z_exact.append(u_exact(x, y))
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
plt.colorbar(orientation='horizontal')

# plt.savefig('./figures/elliptic_bound_FMM.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()
