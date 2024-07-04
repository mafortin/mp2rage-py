"""Plot MP2RAGE signal over time."""

import numpy as np
import matplotlib.pyplot as plt
from mp2ragelib.core import Mz_inv, Mz_0rf, Mz_nrf

# Parameters
eff = 1
TR_MP2RAGE = 6.0
TI_1 = 0.84
TI_2 = 2.37
FA_1 = np.deg2rad(5.)
FA_2 = np.deg2rad(6.)
NR_RF = 168
TR_GRE = 0.0042

nr_timepoints = 1001
time = np.linspace(0, TR_MP2RAGE, nr_timepoints)
signal = np.zeros(nr_timepoints)

# WM/GM/CSF=1.05/1.85/3.35 s
T1s = [1.05, 1.85, 3.35]
T1s_names = ['WM', 'GM', 'CSF']

T_GRE1_start = TI_1 - (NR_RF * TR_GRE / 2)
T_GRE1_end = TI_1 + (NR_RF * TR_GRE / 2)
T_GRE2_start = TI_2 - T_GRE1_end - (NR_RF * TR_GRE / 2)
T_GRE2_end = TI_2 + (NR_RF * TR_GRE / 2)

# Step 0: Inversion
M0 = 1
S0 = Mz_inv(eff=eff, mz0=M0)
Mz0 = 0

for idx, t1 in enumerate(T1s): #loop over the different T1 values provided

    for i, t in enumerate(time): #loop through all time values provided the time variable

        # Step 1: Period with no pulses
        if t < T_GRE1_start:
            signal[i] = Mz_0rf(mz0=S0, t1=t1, t=t, m0=M0)
            Mz0 = signal[i]

        # Step 2: First GRE block
        elif t < T_GRE1_end:
            signal[i] = Mz_nrf(mz0=Mz0, t1=t1, n_gre=NR_RF, tr_gre=TR_GRE,
                               alpha=FA_1, m0=M0)

        # Step 3: Prediod with no pulses
        elif t < T_GRE2_start:
            signal[i] = Mz_0rf(mz0=S0, t1=t1, t=t, m0=M0)
            Mz0 = signal[i]

        # Step 4: Second GRE block
        elif t < T_GRE2_end:
            signal[i] = Mz_nrf(mz0=Mz0, t1=t1, n_gre=NR_RF, tr_gre=TR_GRE,
                               alpha=FA_2, m0=M0)

        # Step 5: Final recovery with no pulses
        else:
            signal[i] = Mz_0rf(mz0=S0, t1=t1, t=t, m0=M0)

    fig = plt.plot(time, signal, label=T1s_names[idx])
    plt.xlabel("Time [s]", fontweight='bold')
    plt.ylabel("Mz(t)", rotation='horizontal', fontweight='bold')
    plt.ylim((-1, 1))
    plt.legend(loc='lower right')
    plt.title("Longitudinal magnetization during MP2RAGE acquisition")

plt.show()