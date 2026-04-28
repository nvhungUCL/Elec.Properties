import numpy as np
from scipy.io import loadmat, savemat

# Load the workspace
data = loadmat('GNR_unitcell.mat')
a = data['a']
a[:, 1] = a[:, 1] - np.min(a[:, 1])
a[:, 1] = a[:, 1] - np.max(a[:, 1])*0.5
a12 = data['a12']
u0 = data['u0']  # Ensure u0 is defined in your .mat file

# Normalization
a = a / u0
a12 = a12 / u0

M = len(a)

# --- Compute H11 ---
H11 = [[M-1, M-1, 0, 0]]
for p in range(M - 1):
    # ip is the range of indices after p
    ip = np.arange(p + 1, M)

    # Calculate squared distances: (dx^2 + dy^2)
    dist_sq = np.sum((a[ip, :] - a[p, :])**2, axis=1)
    drB = (a[ip, 1] + a[p, 1]) * (a[ip, 0] - a[p, 0]) * 0.5

    # Combine indices and distances, then sort by distance (column 1)
    inei = np.column_stack((ip, dist_sq, drB))
    inei = inei[inei[:, 1].argsort()]

    # Check the 3 nearest neighbors
    num_neighbors = min(3, inei.shape[0])
    for k in range(num_neighbors):
        if abs(inei[k, 1] - 1) < 0.2:
            iq = int(inei[k, 0])
            H11.append([p, iq, -1, inei[k, 2]])
            H11.append([iq, p, -1, -inei[k, 2]])

# --- Compute H12 ---
ip = np.arange(M)
vpq = np.array([1, 0]) @ a12

H12 = [[M-1, M-1, 0, 0]]
for p in range(M):
    # Calculate distances to atoms in the adjacent unit cell
    dist_sq = np.sum((a + vpq - a[p, :])**2, axis=1)
    drB = (a[:, 1] + vpq[1] + a[p, 1]) * (a[:, 0] + vpq[0] - a[p, 0]) * 0.5

    inei = np.column_stack((ip, dist_sq, drB))
    inei = inei[inei[:, 1].argsort()]

    for k in range(3):
        if abs(inei[k, 1] - 1) < 0.2:
            iq = int(inei[k, 0])
            H12.append([p, iq, -1, inei[k,2]])

# --- Save Results ---
t0 = 2.7
save_dict = {'t0': t0, 'H11': H11, 'H12': H12}
savemat('GNR_Hamils.mat', save_dict)
