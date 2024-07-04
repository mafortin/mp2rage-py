"""Map UNI values to T1 values."""

import os
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
from nilearn import plotting
from func.functions import compute_T1_lookup_table, map_UNI_to_T1

# =============================================================================
# User-defined parameters

# path to the MP2RAGE UNI
uni_filename = "/home/mafortin/OneDrive/PhD/Data/IMPARK-test-b1cor-mp2rage/sub-01/ses-01/anat/8_t1_mp2rage_sag_0.75-ISO_UNI_Images.nii.gz"

# MP2RAGE parameters
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


# =============================================================================
# Miscellaneous values that *can* be modified by the user (no need to change them tho)

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
# Find UNI to T1 mapping using lookup table method
arr_UNI, arr_T1 = compute_T1_lookup_table(
    T1s=T1s, TR_MP2RAGE=TR_MP2RAGE, TR_GRE=TR_GRE, NR_RF=NR_RF, PF_RF=PF_RF, encoding=encoding, TI_1=TI_1, TI_2=TI_2, FA_1=FA_1, FA_2=FA_2, M0=1.0, eff=eff, only_bijective_part=True)

# Same thing but only to show the full extent of the lookuptable in case of considerable overlapping
# Can be removed/commented once you know that your lookuptable is not overly overlapping at long T1 values.
arr_UNI_full, arr_T1_full = compute_T1_lookup_table(
    T1s=T1s, TR_MP2RAGE=TR_MP2RAGE, TR_GRE=TR_GRE, NR_RF=NR_RF, PF_RF=PF_RF, encoding=encoding, TI_1=TI_1, TI_2=TI_2, FA_1=FA_1, FA_2=FA_2, M0=1.0, eff=eff, only_bijective_part=False)

# Plot UNI vs T1
fig = plt.plot(arr_UNI_full, arr_T1_full, linewidth=1.5)
plt.axhline(y=T1wm, color='k', linestyle='--', linewidth=0.75, label="T1 WM")
plt.axhline(y=T1gm, color='k', linestyle='--', linewidth=0.75, label="T1 GM")
plt.axhline(y=T1csf, color='k', linestyle='--', linewidth=0.75, label="T1 CSF")

# Add annotation
plt.annotate('WM', fontweight='bold', xy=(-0.45, T1wm+0.05), xytext=(-0.45, T1wm+0.05))
plt.annotate('GM', fontweight='bold', xy=(-0.45, T1gm+0.05), xytext=(-0.45, T1gm+0.05))
plt.annotate('CSF', fontweight='bold', xy=(-0.45, T1csf+0.05), xytext=(-0.45, T1csf+0.05))

# Plot setup
plt.xlim(-0.51, 0.51)
plt.ylim(0, 5.01)
plt.xlabel("MP2RAGE UNI Signal", fontweight='bold')
plt.ylabel("T1 [s]", rotation='horizontal', labelpad=20, fontweight='bold')
plt.title("MP2RAGE UNI - T1 lookuptable", fontweight='bold')
plt.show()

# =============================================================================
# Load UNI data
nii = nb.load(uni_filename)
data = nii.get_fdata()
print('Sanity Check: Min & Max value in MP2RAGE UNI image:', round(np.min(data),0), round(np.max(data),0)) # MAF: Sanity check to check it is from 0 to 4095

# Scale range from [0, 4095]] to [-0.5, 0.5]
data /= 4095
data -= 0.5
print('Sanity Check: Min & Max value of rescaled MP2RAGE UNI image:', round(np.min(data),1), round(np.max(data),1)) # MAF: Sanity check

# Compute the T1 map from the lookuptable
T1map = map_UNI_to_T1(img_UNI=data, arr_UNI=arr_UNI, arr_T1=arr_T1)
T1map_ms = T1map*1000  # Seconds to milliseconds

# Save as T1 map as nifti image
img_out = nb.Nifti1Image(T1map_ms, affine=nii.affine)
basename = nii.get_filename().split(os.extsep, 1)[0]
out_name = "{}_T1.nii.gz".format(basename)
nb.save(img_out, out_name)
print("T1 map saved as: %s" % out_name)

# =============================================================================
# Show for the sake of showing

# Show rescaled MP2RAGE UNI image
fig = plt.figure(figsize=(12, 6))
plotting.plot_anat(nb.Nifti1Image(data,affine=nii.affine), figure=fig, cut_coords=(0,0,0), display_mode='ortho', draw_cross=False, vmin=-0.5, vmax=0.5, colorbar=True, annotate=True, title="MP2RAGE UNI image")
plt.show()

# Show T1 map
fig = plt.figure(figsize=(12, 6))
plotting.plot_anat(img_out, figure=fig, cut_coords=(0,0,0), display_mode='ortho', draw_cross=False, vmin=T1min*1000, vmax=T1max*1000, colorbar=True, annotate=True, title="T1 map [ms]")
plt.show()

print("Finished")
