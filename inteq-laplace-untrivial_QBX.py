import numpy as np
import scipy.linalg as la

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import time

## Set the parameters

n_boundary = 256
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

##################################################################
def Z(s):
    xi, eta = r(s)
    return xi + eta * 1j


def dZ(s):
    dxi, deta = drdt(s)
    return dxi + deta * 1j

L = 2 * np.pi
t, h = np.linspace(0, L, n_boundary, endpoint=False, retstep=True)

N_box = int(n_boundary / 5)
box_size = L / N_box
box_center = [box_size / 2 + i * box_size + 2.5 * h * 1j for i in range(N_box)]

n_quadrature = int(n_boundary * 6)
T, H = np.linspace(0, L, n_quadrature, endpoint=False, retstep=True)

# We can approximate the kernel $\rho(t)$ at any t using
# interpolation.

def rho_int(s):
    # Nystrom interpolation to obtain value at arbitrary s
    K = h*np.dot(k(s, t), rho_n)
    return 2 * (-f(s) + K)

if n_quadrature != n_boundary:
    rho = np.array([rho_int(tau) for tau in T]).flatten()
else:
    rho = rho_n

p = 10
# Storing the coefficients of the Taylor expansion for each box
coeff = np.zeros((N_box, p)) + np.zeros((N_box, p)) * 1j

for i in range(N_box):
    for m in range(p):
        z0 = Z(box_center[i])
        coeff[i, m] = (1j / n_quadrature) * np.sum(rho * dZ(T) / (Z(T) - z0)**(m+1))

def u_near_bound(s, t):
    # Note that x + iy = Z(s + it)
    flag = np.mod(int(s / box_size), N_box)  # assign the correct box to this point
    z0 = Z(box_center[flag])
    z = Z(s + t * 1j)
    c = coeff[flag, :]
    v = np.polyval(c[::-1], z - z0)
    return np.real(v)

def u_exact(x, y):
    return np.exp(y) * np.cos(x)


S = np.linspace(0, L, int(n_boundary), endpoint=True)
T = np.linspace(0, 5 * h, int(n_boundary / 2))

S, T = np.meshgrid(S, T)
X_near, Y_near = np.real(Z(S + T * 1j)), np.imag(Z(S + T * 1j))
U_near = []
for s, t in zip(S.flatten(), T.flatten()):
    U_near.append(u_near_bound(s, t))

U_near = np.array(U_near)
U_near = U_near.reshape(S.shape)
U_exact = u_exact(X_near, Y_near)

fig = plt.figure(figsize=(6, 5))
plt.contourf(X_near, Y_near, np.log(np.abs(U_exact - U_near) + 1e-16) / np.log(10), levels=np.linspace(-17, -10, 8))
plt.axis('equal')
plt.axis('off')

plt.colorbar()
plt.show()
# plt.savefig('./figures/untrivial_QBX.pdf', format = 'pdf', bbox_inches = 'tight')
##################################################################