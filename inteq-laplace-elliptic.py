import numpy as np
import scipy.linalg as la

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import time

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


# Kernel in DLP
def M(x, y, s):
    coss = np.cos(s)
    sins = np.sin(s)
    numer = - b * coss * (a * coss - x) - a * sins * (b * sins - y)
    denom = (a * coss - x)**2.0 + (b * sins - y)**2.0
    return (1/(2*np.pi)) * numer/denom

# Sample the angles for the quadrature

T, H = np.linspace(0, L, n_quadrature, endpoint=False, retstep=True)

if n_quadrature != n_boundary:
    rho = np.array([rho_int(tau) for tau in T]).flatten()
else:
    rho = rho_n

def u(x, y):
    # solution given by trapezoidal rule
    return H * np.dot(M(x, y, T), rho)

# We now plot the solution

# First sample the x and y coordinates of the points we want
# to evaluate the solution u(x,y) at

radii = np.linspace(min_radius, max_radius, n_radii)

angles = np.linspace(1e-15, L, n_angles, endpoint=True)
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)

X = a * (radii * np.cos(angles)).flatten()
Y = b * (radii * np.sin(angles)).flatten()

def u_exact(x, y):
    return np.exp(y) * np.cos(x)

Z_exact = []
Z = []
for x, y in zip(X, Y):
    Z.append(u(x, y))
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
x1, x2 = -1.009, -0.438
y1, y2 = -0.942, -0.797
# plt.axis([x1, x2, y1, y2])
plt.axis('off')
# plt.plot(np.array([x1, x2]), np.array([y1, y1]), 'black')
# plt.plot(np.array([x1, x2]), np.array([y2, y2]), 'black')
# plt.plot(np.array([x1, x1]), np.array([y1, y2]), 'black')
# plt.plot(np.array([x2, x2]), np.array([y1, y2]), 'black')
plt.colorbar(orientation='horizontal')
# plt.savefig('./figures/elliptic_bound_无FMM对照组.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()
