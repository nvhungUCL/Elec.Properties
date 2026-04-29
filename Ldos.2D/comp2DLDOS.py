import numpy as np
import inputs
from scipy.io import loadmat, savemat
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# Constants and Parameters
CoffB = inputs.CoffB
B_tesla = inputs.B_tesla
nkx = inputs.nkx
Ecomp = inputs.Ecomp
eta = inputs.eta
del inputs

B = B_tesla * CoffB

# Symmetric directions
kvx_range = np.arange(0.5 - nkx, nkx + 0.5, 1)
kvx = np.pi * kvx_range / nkx
nkx = len(kvx)

# Load Hamiltonian data
# Note: Ensure Hamils.mat is in the same directory
data = loadmat('Hamils.mat')
t0 = data['t0']
H11 = data['H11']
H12 = data['H12']
del data

iH11 = H11[:, 0]
iH11 = iH11.astype(np.int64)
jH11 = H11[:, 1]
jH11 = jH11.astype(np.int64)
matH11 = H11[:, 2:4]

iH12 = H12[:, 0]
iH12 = iH12.astype(np.int64)
jH12 = H12[:, 1]
jH12 = jH12.astype(np.int64)
matH12 = H12[:, 2:4]

# Apply magnetic field and phase
# matH11[:, 0] is the value, matH11[:, 1] is the coefficient for B
val11 = matH11[:, 0] * np.exp(1j * (matH11[:, 1] * B))
val12 = matH12[:, 0] * np.exp(1j * (matH12[:, 1] * B))

# Create Sparse Matrices
H11 = csr_matrix((val11, (iH11, jH11)))
H12 = csr_matrix((val12, (iH12, jH12)))
M = H11.shape[0]

LDOS = 0

# Main Loop
for k in range(nkx):
    # Calculate Hk
    phase = np.exp(1j * kvx[k])
    Hk_sparse = H12 * phase

    # Construct total Hamiltonian (Hk + Hk_hermitian + H11)
    # Convert to dense for eigendecomposition
    H_total = (Hk_sparse + Hk_sparse.getH() + H11).toarray()

    # Compute the Green's function
    GF = np.linalg.inv((Ecomp + 1j * eta) * np.eye(M) - H_total)

    # Compute LDOS
    LDOS = LDOS - np.diag(GF.imag)

# Convert to arrays for saving/plotting

# Save results
LDOS = LDOS.T
savemat('LDOSdata.mat', {'Ene': Ecomp, 'B': B_tesla, 'LDOS': LDOS})

