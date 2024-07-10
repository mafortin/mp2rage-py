"""Map UNI values to T1 values."""

import os
import argparse
import numpy as np
import nibabel as nb
from func.functions import compute_T1_lookup_table, map_UNI_to_T1


def main():
    parser = argparse.ArgumentParser(description="Map UNI values to T1 values.")

    # User-defined parameters
    parser.add_argument('--uni_pathname', type=str, required=True,
                        help='Path to the MP2RAGE UNI image')
    parser.add_argument('--t1map_pathname', type=str, required=False, default=os.getcwd(),
                        help='Path to the resulting T1 map (default: current working directory)')
    parser.add_argument('--eff', type=float, default=0.96,
                        help='MP2RAGE inversion efficiency (default: 0.96)')
    parser.add_argument('--encoding', type=str, choices=['linear', 'centric'], default='linear',
                        help='MP2RAGE k-space encoding type. If unsure, leave it to linear. (default: linear)')
    parser.add_argument('--TR_MP2RAGE', type=float,
                        help='MP2RAGE repetition time [Siemens exam card: TR]')
    parser.add_argument('--TI1', type=float,
                        help='First inversion time [Siemens exam card: TI 1]')
    parser.add_argument('--TI2', type=float,
                        help='Second inversion time [Siemens exam card: TI 2]')
    parser.add_argument('--FA1', type=float,
                        help='First flip angle in degrees [Siemens exam card: Flip angle 1]')
    parser.add_argument('--FA2', type=float,
                        help='Second flip angle in degrees [Siemens exam card: Flip angle 2]')
    parser.add_argument('--N_RF', type=int,
                        help='Number of small flip angles/shots in each RAGE block  [Siemens exam card: Slice per slab]')
    parser.add_argument('--PF_RF', type=float,
                        help='Partial Fourier factor on the inner loop (i.e., on the short TRs, not the TR_MP2RAGE). Please enter the value in decimals instead of the fraction (e.g., 0.75 instead of 6/8)  [Siemens exam card: Slice Partial Fourier]')
    parser.add_argument('--TR_GRE', type=float,
                        help='Repetition time [s] of each small flip angle excitation of the FLASH readout [Siemens exam card: Echo Spacing]')
    parser.add_argument('--only_bijective', type=bool, default=True,
                        help='Set this flag to True if you want to cut the UNI-T1 lookup table so that it becomes bijective and no overlap is present (default: True)')

    # Parse arguments
    args = parser.parse_args()

    # Convert flip angles from degrees to radians
    FA1 = np.deg2rad(args.FA1)
    FA2 = np.deg2rad(args.FA2)

    ### T1min, T1max and nr_timepoints can be modified by the user based on personal preferences if desired, but this is not required.
    # Boundaries to calculate the T1 lookuptable
    T1min = 0.001  # s
    T1max = 5.001  # s

    # Step for UNI-T1 lookuptable
    nr_timepoints = 5001
    T1s = np.linspace(T1min, T1max, nr_timepoints)
    ###

    # Find UNI to T1 mapping using lookup table method
    arr_UNI, arr_T1 = compute_T1_lookup_table(
        T1s=T1s, TR_MP2RAGE=args.TR_MP2RAGE, TR_GRE=args.TR_GRE, NR_RF=args.N_RF, PF_RF=args.PF_RF,
        encoding=args.encoding, TI_1=args.TI1, TI_2=args.TI2, FA_1=FA1, FA_2=FA2,
        M0=1.0, eff=args.eff, only_bijective_part=args.only_bijective)

    # Load UNI data
    nii = nb.load(args.uni_pathname)
    data = nii.get_fdata()
    print('Sanity Check: Min & Max value in MP2RAGE UNI image:', round(np.min(data), 0), round(np.max(data), 0))

    # Scale range from [0, 4095] to [-0.5, 0.5]
    data /= 4095
    data -= 0.5
    print('Sanity Check: Min & Max value of rescaled MP2RAGE UNI image:', round(np.min(data), 1),
          round(np.max(data), 1))

    # Compute the T1 map from the lookuptable
    T1map = map_UNI_to_T1(img_UNI=data, arr_UNI=arr_UNI, arr_T1=arr_T1)
    T1map *= 1000  # Seconds to milliseconds

    # Save as T1 map as nifti image
    img_out = nb.Nifti1Image(T1map, affine=nii.affine)
    basename = nii.get_filename().split(os.extsep, 1)[0]
    out_name = "{}_T1.nii.gz".format(basename)
    nb.save(img_out, out_name)
    print("T1 map saved as: %s" % out_name)


if __name__ == "__main__":
    main()
