import numpy as np
import matplotlib.pyplot as plt
import GNR_inputs
from scipy.io import savemat

# Parameters
M = GNR_inputs.M
u0 = GNR_inputs.u0

# Lattice vectors
# a12 is a 2x2 matrix
a12 = np.array([[3, 0], [0, np.sqrt(3)]]) * u0

# Atomic positions in the unit cell
# r0 calculation (4x2 matrix)
r0 = np.array([[0, 1/2], [1/6, 0], [1/2, 0], [2/3, 1/2]])
r0 = r0 @ a12

# Generate lattice repetitions
a = np.arange(M).reshape(-1, 1)
# Create a 2D array where col1 is 0 and col2 is 0:M-1
a = np.hstack([np.zeros((M, 1)), a]) @ a12

# Concatenate the 4 basis atoms shifted by the repetitions
a = np.vstack([a + r0[0, :],
               a + r0[1, :],
               a + r0[2, :],
               a + r0[3, :]])

# Centering transformations
a[:, 1] = a[:, 1] - np.min(a[:, 1])
a[:, 1] = a[:, 1] - np.max(a[:, 1]) / 2

# Update a12 for the supercell
a12 = np.array([[1, 0], [0, M]]) @ a12

# Save to .mat file
savemat('GNR_unitcell.mat', {'u0': u0, 'a12': a12, 'a': a})

# Plotting
plt.figure(figsize=(10, 8))

# Loop for repetitions
for p in range(M):
    plt.plot(a[:, 0], a[:, 1], 'bo', alpha=0.5)
    a = a + np.array([1, 0]) @ a12

plt.plot(a[:, 0], a[:, 1], 'ro', alpha=0.5)
a = a - np.array([M+1, 0]) @ a12
plt.plot(a[:, 0], a[:, 1], 'ro', alpha=0.5)

plt.axis('equal')
plt.title('GNR Unit Cell Repetitions')
plt.show()
