import numpy as np
import triangle as tr
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

a, b = 2.0, 1.0  # Semi-major and semi-minor axes

# Generate boundary points
theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
x_boundary = a * np.cos(theta)
y_boundary = b * np.sin(theta)

# Create boundary points and segments
boundary_points = np.column_stack([x_boundary, y_boundary])
segments = np.array([[i, (i + 1) % len(boundary_points)] for i in range(len(boundary_points))])

# Constrained Delaunay triangulation
mesh_info = dict(vertices=boundary_points, segments=segments)
mesh = tr.triangulate(mesh_info, 'pq30a0.05')

vertices = mesh["vertices"]
triangles = mesh["triangles"]

# Visualize the triangulation
plt.figure(figsize=(6, 3))
plt.triplot(vertices[:, 0], vertices[:, 1], triangles, color="blue", linewidth=1)
# plt.scatter(vertices[:, 0], vertices[:, 1], color="red", s=5)
plt.axis("equal")
plt.axis('off')
plt.show()
# plt.savefig('./figures/elliptic_三角剖分.pdf', format = 'pdf', bbox_inches = 'tight')

# boundary condition
f = lambda x, y: np.exp(y) * np.cos(x)

# Initialize global stiffness matrix and RHS vector
num_nodes = len(vertices)
A = sp.lil_matrix((num_nodes, num_nodes))
b = np.zeros(num_nodes)

# FEM Assembly
for tri in triangles:
    p = vertices[tri]
    B = np.array([[1, p[0, 0], p[0, 1]],
                  [1, p[1, 0], p[1, 1]],
                  [1, p[2, 0], p[2, 1]]])
    C = np.linalg.inv(B)
    grad_phi = C[1:, :]  # Gradient of basis functions
    area = 0.5 * np.abs(np.linalg.det(B))
    K_local = area * grad_phi.T @ grad_phi
    # Assemble into global stiffness matrix
    for i in range(3):
        for j in range(3):
            A[tri[i], tri[j]] += K_local[i, j]

# Apply boundary conditions
boundary_indices = np.unique(segments.flatten())
for idx in boundary_indices:
    x, y = vertices[idx]
    A[idx, :] = 0
    A[idx, idx] = 1
    b[idx] = f(x, y)

A = A.tocsr()  # Convert to CSR format for efficient solving
u = spla.spsolve(A, b)
def u_exact(x, y):
    return np.exp(y) * np.cos(x)


U_exact = u_exact(vertices[:, 0], vertices[:, 1])

# Plot the solution
plt.figure(figsize=(6, 5))
plt.tricontourf(vertices[:, 0], vertices[:, 1], triangles, np.log(np.abs(U_exact - u) + 1e-16) / np.log(10), levels=np.linspace(-16, 0, 17))
plt.colorbar(orientation='horizontal')
plt.axis("equal")
plt.axis('off')
plt.show()
