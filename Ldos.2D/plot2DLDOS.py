import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# 1. Load the data
device_data = loadmat('Device.mat')
ldos_data = loadmat('LDOSdata.mat')

# Extract variables (handling potential 2D vs 1D array differences from MATLAB)
a = device_data['a']      # Coordinates
a12 = device_data['a12']  # Lattice vectors
ldos = ldos_data['LDOS']  # LDOS values
B = ldos_data['B']        # B field in Tesla
del ldos_data, device_data

# 2. Append LDOS as a new column
# If LDOS is (N,1), use hstack. If it's (1,N), transpose it.
a = np.hstack([a, ldos.reshape(-1, 1)])
a1 = np.copy(a)

# 3. Replicate the geometry for p = 1 to 180
a_list = [a]
for p in range(1, 181):
    vp = np.array([p, 0]) @ a12
    # Create new coordinates but keep the LDOS values from the original unit
    # New coordinates: old_coords + [vp_x, vp_y, 0]
    shift = np.append(vp, 0)
    a_list.append(a1 + shift)

# Combine all replicated segments
a = np.vstack(a_list)

# 4. Sort by the 3rd column (index 2)
# This mimics sortrows(a, 3)
a = a[a[:, 2].argsort()]
a[:,0:2] = a[:,0:2]/10

# 5. Visualization
plt.figure(figsize=(10, 8))
# scatter(x, y, size, color, ...)
# a[:, 0] is x, a[:, 1] is y, a[:, 3] is the LDOS value (color)
sc = plt.scatter(a[:, 0], a[:, 1], s=15, c=a[:, 2], cmap='viridis', edgecolors='none')
plt.axis('equal')
plt.colorbar(sc, label='LDOS (a.u.)')
plt.xlabel('x (nm)')
plt.ylabel('y (nm)')
plt.title(f'GNR Device LDOS Mapping: $B$ = {B[0, 0]} T')
plt.axis('equal') # axis equal
plt.autoscale(tight=True) # tight
#plt.axis('off') # axis off
plt.show()
