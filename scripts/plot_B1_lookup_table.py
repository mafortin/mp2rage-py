"""Plot the impact of B1+ inhomogeneities on the MP2RAGE UNI to T1 lookup table."""

import numpy as np
import matplotlib.pyplot as plt
from func.functions import compute_T1_lookup_table

# =============================================================================
# MP2RAGE parameters (requires user inputs based on their MP2RAGE protocol)

eff = 0.96 #0.96 or 1?
encoding = 'linear' # 'linear' (k-space center is reached after half the k-space is filled) or 'centric' (k-space center is reached at the beginning)
TR_MP2RAGE = 4.3
TI_1 = 0.840
TI_2 = 2.370
FA_1 = np.deg2rad(5.0)
FA_2 = np.deg2rad(6.0)
NR_RF = 224 # MAF: Nb. of small flip angles/shots in each RAGE block, Siemens' exam card parameter: Slices per slab. This is **excluding** the Slice partial Fourier Factor in Exam card.
PF_RF = 6/8 # MAF: Very important that this is the Partial Fourier Factor on the **INNER LOOP** (short TR), not the outer loop (long TR). This changes the moment the center of k-space is crossed, and not the total acquisition time [it's the Phase Partial Fourier Factor that reduces scan time]. Very confusing naming convention from Siemens/developer's team.
TR_GRE = 0.0072

# Range of B1+ to plot lookup table for
B1vec = np.arange(0.4, 1.21, 0.2)

# =============================================================================
# Miscellaneous values that can be modified by the user (no need to change them if unsure)

# WM/GM/CSF=1.05/1.85/3.35 s  (at 7T)
T1wm = 1.05
T1gm = 1.85
T1csf = 3.35

# Boundaries to calculate the T1 lookuptable
T1min = 0.001 #s
T1max = 5.001 #s

# Step for UNI-T1 lookuptable
nr_timepoints = 5001
T1s = np.linspace(T1min, T1max, nr_timepoints)

# =============================================================================
# Impact of B1+ inhomogeneities

cmap = plt.get_cmap('viridis')

# Find UNI to T1 mapping using lookup table method
for idx, b1 in enumerate(B1vec):

    arr_UNI, arr_T1 = compute_T1_lookup_table(
        T1s=T1s, TR_MP2RAGE=TR_MP2RAGE, TR_GRE=TR_GRE, NR_RF=NR_RF, PF_RF=PF_RF, encoding=encoding,
        TI_1=TI_1, TI_2=TI_2, FA_1=b1*FA_1, FA_2=b1*FA_2, M0=1.0, eff=eff, only_bijective_part=False)

    # Plot UNI vs T1
    if round(b1,1) == 1.0:
        fig = plt.plot(arr_UNI, arr_T1, linewidth=1.25, color='r', label='%s*B1' % (round(b1,1)))
    else:
        fig = plt.plot(arr_UNI, arr_T1, linewidth=1.25, color=cmap(idx / len(B1vec)), label='%s*B1' % (round(b1, 1)))

plt.axhline(y=T1wm, color='k', linestyle='--', linewidth=0.75, label='_nolegend_')
plt.axhline(y=T1gm, color='k', linestyle='--', linewidth=0.75, label='_nolegend_')
plt.axhline(y=T1csf, color='k', linestyle='--', linewidth=0.75, label='_nolegend_')

# Add annotation
plt.annotate('WM', fontweight='bold', xy=(-0.45, T1wm+0.05), xytext=(-0.45, T1wm+0.05))
plt.annotate('GM', fontweight='bold', xy=(-0.45, T1gm+0.05), xytext=(-0.45, T1gm+0.05))
plt.annotate('CSF', fontweight='bold', xy=(-0.45, T1csf+0.05), xytext=(-0.45, T1csf+0.05))

# Plot setup
mp2min = -0.51
mp2max = 0.51
stepsize = 0.1

plt.xlim(mp2min, mp2max)
plt.xticks(np.arange(mp2min+0.01, mp2max, step=stepsize))
plt.yticks(np.arange(0, T1max, step=1))
plt.ylim(0, T1max)
plt.xlabel("MP2RAGE UNI Signal", fontweight='bold')
plt.ylabel("T1 [s]", rotation='horizontal', labelpad=20, fontweight='bold')
plt.title("MP2RAGE UNI - T1 lookuptable\n Impact of B1+ inhomogeneities", fontweight='bold')
plt.legend(loc='lower center')
plt.show()

print("Done! You can now look at your MP2RAGE-T1 lookup table. :)")