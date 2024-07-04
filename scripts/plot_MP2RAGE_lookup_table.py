"""Plot MP2RAGE UNI to T1 lookup table."""

import numpy as np
import matplotlib.pyplot as plt
from mp2ragelib.core import compute_T1_lookup_table

# Required parameters from MP2RAGE protocol
TR_MP2RAGE = 5.0
INVERSION_TIME_1 = 0.8
INVERSION_TIME_2 = 2.7
FLIP_ANGLE_1 = np.deg2rad(4.)
FLIP_ANGLE_2 = np.deg2rad(5.)

# Needed to determine the duration of the readout blocks
NR_RF = 216
TR_GRE = 0.00291

# Efficiency of the initial pulse
EFFICIENCY = 0.96

# WM/GM/CSF=1.05/1.85/3.35 s  (at 7T)
nr_timepoints = 1001
T1s = np.linspace(0.5, 5.5, nr_timepoints)

# =============================================================================
# Find UNI to T1 mapping using lookup table method
arr_UNI, arr_T1 = compute_T1_lookup_table(
    T1s=T1s, TR_MP2RAGE=TR_MP2RAGE, TR_GRE=TR_GRE, NR_RF=NR_RF,
    TI_1=INVERSION_TIME_1, TI_2=INVERSION_TIME_2,
    FA_1=FLIP_ANGLE_1, FA_2=FLIP_ANGLE_2, M0=1., eff=EFFICIENCY,
    only_bijective_part=True)

# =============================================================================
# Plot UNI vs T1
fig = plt.plot(arr_UNI, arr_T1)
plt.xlabel("UNI")
plt.ylabel("T1")
plt.xlim((-0.5, 0.5))
plt.show()

print("Finished")
