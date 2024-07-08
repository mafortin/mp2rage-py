"""Get a B1 corrected T1 map."""
import SimpleITK
# Imports

import numpy as np
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt
from func.functions import coreg_and_resample_B1map, resample_B1map

# Get paths of both the B1 map (moving image) and MP2RAGE UNI (fixed image)
b1map_pathname = '/home/mafortin/OneDrive/PhD/Data/IMPARK-test-b1cor-mp2rage/sub-01/ses-01/anat/3_b1map_tra_p2_cp_adj_vol.nii'
mp2rage_pathname = '/home/mafortin/OneDrive/PhD/Data/IMPARK-test-b1cor-mp2rage/sub-01/ses-01/anat/8_t1_mp2rage_sag_0.75-ISO_UNI_Images.nii.gz'

b1map = nib.load(b1map_pathname)
mp2rage = nib.load(mp2rage_pathname)

# =============================================================================
# Show for the sake of showing

# Show B1 map
fig = plt.figure(figsize=(12, 6))
plotting.plot_anat(b1map, figure=fig, cut_coords=(0,0,0), display_mode='ortho', draw_cross=False, vmin=200, vmax=1200, colorbar=True, annotate=True, title="B1 map (moving image)")
#plt.show()

# Show MP2RAGE UNI image
fig = plt.figure(figsize=(12, 6))
plotting.plot_anat(mp2rage, figure=fig, cut_coords=(0,0,0), display_mode='ortho', draw_cross=False, vmin=0, vmax=4095, colorbar=True, annotate=True, title="MP2RAGE UNI (fixed image)")
#plt.show()

# =============================================================================
# Apply rigid registration and resampling of the B1 map to the MP2RAGE UNI image

b1_coreg_resamp = coreg_and_resample_B1map(b1map_pathname, mp2rage_pathname)

# Resample the moving image to match the fixed image
#b1_resamp_img = resample_B1map(b1map_pathname, mp2rage_pathname)

# =============================================================================
# Show for the sake of showing

# Show B1 map
fig = plt.figure(figsize=(12, 6))
plotting.plot_anat(b1_coreg_resamp, figure=fig, cut_coords=(0,0,0), display_mode='ortho', draw_cross=False, vmin=None, vmax=None, colorbar=True, annotate=True, title="B1 map coregistered and resampled")
plt.show()

# Show MP2RAGE UNI image
#fig = plt.figure(figsize=(12, 6))
#plotting.plot_anat(mp2rage, figure=fig, cut_coords=(0,0,0), display_mode='ortho', draw_cross=False, vmin=0, vmax=4095, colorbar=True, annotate=True, title="MP2RAGE UNI (fixed image)")
#plt.show()

