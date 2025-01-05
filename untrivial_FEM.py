import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import triangle as tr
import matplotlib.pyplot as plt

# Define the polar boundary function
def r_theta(theta):
    return 1 + 0.3 * np.cos(3 * (theta + 0.3 * np.sin(theta)))

# Discretize the boundary
theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
r = r_theta(theta)
x_boundary = r * np.cos(theta)
y_boundary = r * np.sin(theta)

# Create boundary points and segments
points = np.column_stack([x_boundary, y_boundary])
segments = np.array([[i, (i + 1) % len(points)] for i in range(len(points))])

# Constrained Delaunay triangulation
mesh_info = dict(vertices=points, segments=segments)
mesh = tr.triangulate(mesh_info, 'pq30a0.001')

# Visualize the triangulation
plt.figure(figsize=(6, 5))
plt.triplot(mesh["vertices"][:, 0], mesh["vertices"][:, 1], mesh["triangles"], color="blue", linewidth=0.5)
# plt.scatter(mesh["vertices"][:, 0], mesh["vertices"][:, 1], color="red", s=5)
plt.axis("equal")
plt.axis('off')
plt.show()
# plt.savefig('./figures/Untrivial_三角剖分.pdf', format = 'pdf', bbox_inches = 'tight')

vertices = mesh["vertices"]
triangles = mesh["triangles"]

num_nodes = len(vertices)
num_elements = len(triangles)

# Define the boundary condition
f = lambda x, y: np.exp(y) * np.cos(x)

# Initialize global stiffness matrix and RHS vector
A = sp.lil_matrix((num_nodes, num_nodes))
b = np.zeros(num_nodes)

# FEM Assembly
for tri in triangles:
    p = vertices[tri]
    # Compute the local stiffness matrix
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
plt.colorbar()
plt.axis("equal")
plt.axis('off')
# plt.show()
plt.savefig('./figures/Untrivial_FEM.pdf', format = 'pdf', bbox_inches = 'tight')
