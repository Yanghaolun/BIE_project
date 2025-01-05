import numpy as np
import scipy.linalg as la

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import time

# Set the parameters
n_boundary = 256
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
    # ellipse boundary
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


##################################################################
def Z(s):
    xi = a * np.cos(s)
    eta = b * np.sin(s)
    return xi + eta * 1j


def dZ(s):
    dxi = -a * np.sin(s)
    deta = b * np.cos(s)
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
# Storing the coefficients
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
plt.colorbar(orientation='horizontal')

# plt.savefig('./figures/elliptic_bound_QBX.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()

##################################################################



