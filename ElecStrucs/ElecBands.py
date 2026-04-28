import numpy as np
import inputs
from scipy.io import loadmat, savemat
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# Constants and Parameters
nkx = inputs.nkx
BZsize = inputs.BZsize
Emax = inputs.Emax
kmax = inputs.kmax
del inputs

# Symmetric directions
kvx_range = np.arange(0.5 - nkx, nkx + 0.5, 1)
kvx = np.pi * kvx_range * BZsize / nkx
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
val11 = H11[:, 2]

iH12 = H12[:, 0]
iH12 = iH12.astype(np.int64)
jH12 = H12[:, 1]
jH12 = jH12.astype(np.int64)
val12 = H12[:, 2]

# Create Sparse Matrices
H11 = csr_matrix((val11, (iH11, jH11)))
H12 = csr_matrix((val12, (iH12, jH12)))

Eb = []
KP = []

# Main Loop
for k in range(nkx):
    # Calculate Hk
    phase = np.exp(1j * kvx[k])
    Hk_sparse = H12 * phase

    # Construct total Hamiltonian (Hk + Hk_hermitian + H11)
    # Convert to dense for eigendecomposition
    H_total = (Hk_sparse + Hk_sparse.getH() + H11).toarray()

    # Eigenvalues (eigh is optimized for Hermitian/Symmetric matrices)
    evals = np.linalg.eigh(H_total)[0]
    Hk_res = np.real(evals * t0)

    # Store results
    KP.append((k + 1 - 0.5) / nkx - 0.5)
    Eb.append(Hk_res)

# Convert to arrays for saving/plotting
KP = np.array(KP)
Eb = np.array(Eb)
Eb = Eb.squeeze()
KP = KP.reshape(-1, 1)

# Save results
savemat('Bands.mat', {'KP': KP, 'Eb': Eb})

plt.figure(figsize=(10, 8))
plt.plot(KP, Eb)
plt.xlim(-kmax, kmax)
plt.ylim(-Emax, Emax)
plt.grid(True)
plt.show()

