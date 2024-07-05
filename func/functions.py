"""Core functions."""
import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib

def compute_UNI(inv1_re, inv1_im, inv2_re, inv2_im, scale=False):
    """Compute UNI image.

    Parameters
    ----------
    inv1_re: np.ndarray
        Real component of MP2RAGE first inversion complex image.
    inv1_im: np.ndarray
        Imaginary component of MP2RAGE first inversion complex image.
    inv2_re: np.ndarray
        Real component of MP2RAGE second inversion complex image.
    inv2_im: np.ndarray
        Imaginary component of MP2RAGE second inversion complex image.
    scale : bool
        Do not scale and clip the results when false.

    Returns
    -------
    uni: np.ndarray, 3D
        Unified image that looks similar to T1-weighted images.

    Reference
    ---------
    - Marques, J. P., Kober, T., Krueger, G., van der Zwaag, W.,
    Van de Moortele, P.-F., Gruetter, R. (2010). MP2RAGE, a self bias-field
    corrected sequence for improved segmentation and T1-mapping at high field.
    NeuroImage, 49(2), 1271–1281.
    <https://doi.org/10.1016/j.neuroimage.2009.10.002>

    """
    inv1 = inv1_re + inv1_im * 1j
    inv2 = inv2_re + inv2_im * 1j

    # Marques et al. (2010), equation 3.
    uni = inv1.conj() * inv2 / (np.abs(inv1)**2 + np.abs(inv2)**2)
    uni = np.real(uni)

    if scale:  # Scale to 0-4095 (uin12) range
        # NOTE(Faruk): Looks redundant but keeping for backwards compatibility.
        uni *= 4095
        uni += 2048
        uni = np.clip(uni, 0, 4095)

    return uni


def Mz_inv(mz0, eff):
    """Magnetization after adiabatic inversion pulse.

    Parameters
    ----------
    mz0 : float
        Longitudinal magnetization at time=0.
    eff : float
        Efficiency of the adiabatic inversion pulse. The default value of
        '0.96' is used by Marques et al. (2010)

    Returns
    -------
    signal : float
        Longitudinal magnetization.

    Notes
    -----
    Marques et al. (2010), Appendix 1, Item A):
    'Longitudinal magnetization is inverted by means of an adiabatic pulse of
    a given efficiency.'

    Reference
    ---------
    - Marques, J. P., Kober, T., Krueger, G., van der Zwaag, W.,
    Van de Moortele, P.-F., Gruetter, R. (2010). MP2RAGE, a self bias-field
    corrected sequence for improved segmentation and T1-mapping at high field.
    NeuroImage, 49(2), 1271–1281.
    <https://doi.org/10.1016/j.neuroimage.2009.10.002>

    """
    signal = -eff * mz0
    return signal


def Mz_nrf(mz0, t1, n_gre, tr_gre, alpha, m0):
    """Magnetization during the GRE block.

    Parameters
    ----------
    mz0 : float
        Longitudinal magnetization at time=0.
    t1 : float, np.ndarray
        T1 time in seconds.
    n_gre : int
        Number of radio frequency (RF) pulses in each GRE block.
    tr_gre : float
        Repetition time (TR) of gradient recalled echo (GRE) pulses in
        seconds.
    alpha : float
        Flip angle in radians.
    m0 : float
        Longitudinal magnetization at the start of the RF free periods.

    Returns
    -------
    signal : float
        Longitudinal magnetization.

    Notes
    -----
    - Marques et al. (2010), Appendix 1, Item B):
    'During the GRE blocks of n RF pulses with constant flip angles (alpha),
    separated by an interval TR, the longitudinal magnetization evolves in
    the following way (Deichmann and Haase, 1992).'

    Reference
    ---------
    - Marques, J. P., Kober, T., Krueger, G., van der Zwaag, W.,
    Van de Moortele, P.-F., Gruetter, R. (2010). MP2RAGE, a self bias-field
    corrected sequence for improved segmentation and T1-mapping at high field.
    NeuroImage, 49(2), 1271–1281.
    <https://doi.org/10.1016/j.neuroimage.2009.10.002>

    """
    signal = (mz0 * (np.cos(alpha) * np.exp(-tr_gre / t1)) ** n_gre
              + m0 * (1 - np.exp(-tr_gre / t1))
              * (1 - (np.cos(alpha) * np.exp(-tr_gre / t1)) ** n_gre)
              / (1 - np.cos(alpha) * np.exp(-tr_gre / t1))
              )
    return signal


# Appendix c)
def Mz_0rf(mz0, t1, t, m0):
    """Magnetization during the period with no pulses.

    Marques et al. (2010), Appendix 1, Item C):
    'During the periods with no RF pulses, the longitudinal magnetization
    relaxes freely towards equilibrium following the conventional T1
    relaxation expression.'

    Parameters
    ----------
    mz0 : float
        Longitudinal magnetization at time=0.
    t1 : float, np.ndarray
        T1 time in seconds.
    t : float
        Time in seconds.
    m0 : float
        Longitudinal magnetization at the start of the RF free periods.

    Returns
    -------
    signal : float
        Longitudinal magnetization.

    Reference
    ---------
    - Marques, J. P., Kober, T., Krueger, G., van der Zwaag, W.,
    Van de Moortele, P.-F., Gruetter, R. (2010). MP2RAGE, a self bias-field
    corrected sequence for improved segmentation and T1-mapping at high field.
    NeuroImage, 49(2), 1271–1281.
    <https://doi.org/10.1016/j.neuroimage.2009.10.002>

    """
    signal = mz0 * np.exp(-t / t1) + m0 * (1 - np.exp(-t / t1))
    return signal


def compute_T1_lookup_table(T1s, TR_MP2RAGE, TR_GRE, NR_RF, PF_RF, encoding, TI_1, TI_2, FA_1, FA_2, M0=1., eff=0.96, only_bijective_part=True):
    """Find T1 values from UNI image.

    Parameters
    ----------
    T1s : numpy.ndarray, 1D, float
        An array of T1 times in seconds. The range of T1 values in this array
        will be used to generate the corresponding UNI values.
    M0 : float
        Longitudinal signal at time=0.
    NR_RF : int
        Number of radio frequency pulses in one GRE readout.
    PF_RF : float
        Partial Fourier factor applied on the number of RF pulses/shots (inner loop [i.e., short TRs in each RAGE block])
        The Siemens acquisition parameter related to that value is 'Slice partial Fourier' (which can bring confusion)
    encoding : string
        K-space encoding scheme used. Siemens' acquisition parameter: 'Reordering'. Usually 'Linear', 'centric' or 'radial'.
        The provided script is designed for 'linear' only at the moment, but 'centric' encoding should be pretty straightforward
        to implement.
    TR_GRE : float
        Time between successive excitation pulses in the GRE kernel in
        seconds, which is composed of NR_RF pulses.
    TR_MP2RAGE : float
        Time between two successive inversion pulses in seconds.
    TI_1 : float
        First inversion time in seconds.
    TI_2 : float
        Second inversion time in seconds.
    FA_1 : float
        Flip angle 1 in radians.
    FA_2 : float
        Flip angle 2 in radians.
    eff: float
        Inversion efficiency of the adiabatic pulse. Default is 0.96 as used in
        Marques et al. (2010), mentioned below Equation 3.
    only_bijective_part: bool
        Only take the bijective part. If 'False', return all values.

    Returns
    -------
    arr_UNI: np.ndarray, 1D, float
        Array of UNI values (MP2RAGE signal)
    arr_T1: np.ndarray, 1D, float
        Array of T1 values that correspond to the UNI values.

    Notes
    -----
    - Marques et al. (2010), Appendix 1, Item C):
    'A full account of the signal resulting from the MP2RAGE sequence has to
    take into account the steady-state condition. This implies that the
    longitudinal magnetization before successive inversions, mz,ss, has to be
    the same. Between two successive inversions the mz,ss undergoes first an
    inversion (a), followed by recovery for a period TA (c), a first GRE
    block (b), a free recovery for a period TB (c), a second GRE block (b),
    and a final recovery for a period TC (c) by the end of which it should be
    back to its initial value.'

    Reference
    ---------
    - Marques, J. P., Kober, T., Krueger, G., van der Zwaag, W.,
    Van de Moortele, P.-F., Gruetter, R. (2010). MP2RAGE, a self bias-field
    corrected sequence for improved segmentation and T1-mapping at high field.
    NeuroImage, 49(2), 1271–1281.
    <https://doi.org/10.1016/j.neuroimage.2009.10.002>

    """

    # Calculating the proper timings based on Partial Fourier
    if PF_RF == 1:

        if encoding == 'linear':
            # Derived time parameters
            TA = TI_1 - (NR_RF * TR_GRE / 2)
            TB = TI_2 - (TI_1 + (NR_RF * TR_GRE / 2))
            TC = TR_MP2RAGE - (TI_2 + (NR_RF * TR_GRE / 2)) # MAF: Typo changed from TI_1 to TI_2 (correct value).

        elif encoding == 'centric':
            # Derived time parameters
            TA = TI_1
            TB = TI_2 - (TI_1 + (NR_RF * TR_GRE))
            TC = TR_MP2RAGE - (TI_2 + (NR_RF * TR_GRE))

    else: # If there is PF applied on the RAGE block/RF pulses

        NR_RFeff = PF_RF * NR_RF  #'Effective' nb. of RF pulses/shots used when Partial Fourier is applied.

        if encoding == 'linear':

            NR_RFaf = NR_RF/2 #Nb. of RF pulses **after** reaching k-space center, which should stay unchanged even with PF applied.
            NR_RFbef = NR_RFeff - NR_RFaf #Nb. of RF pulses **before** reaching k-space center. PF is applied at the beginning to reach the k-space center as soon as possible.

            # Derived time parameters
            TA = TI_1 - (NR_RFbef * TR_GRE)
            TB = TI_2 - (TI_1 + (NR_RFaf * TR_GRE))
            TC = TR_MP2RAGE - (TI_2 + (NR_RFaf * TR_GRE))  # MAF: Typo changed from TI_1 to TI_2 (correct value).

        elif encoding == 'centric':

            # Derived time parameters
            TA = TI_1
            TB = TI_2 - (TI_1 + (NR_RFeff * TR_GRE))
            TC = TR_MP2RAGE - (TI_2 + (NR_RFeff * TR_GRE))


    # NOTE(Faruk): Go through all stages for completeness.
    signal = np.zeros(T1s.shape + (6,))

    for i, t1 in enumerate(T1s):
        # Step 0: Inversion
        signal[i, 0] = Mz_inv(eff=eff, mz0=M0)

        # Step 1: Period with no pulses
        signal[i, 1] = Mz_0rf(mz0=signal[i, 0], t1=t1, t=TA, m0=M0)

        # Step 2: First GRE block
        signal[i, 2] = Mz_nrf(mz0=signal[i, 1], t1=t1, n_gre=NR_RF,
                              tr_gre=TR_GRE, alpha=FA_1, m0=M0)

        # Step 3: Prediod with no pulses
        signal[i, 3] = Mz_0rf(mz0=signal[i, 2], t1=t1, t=TB, m0=M0)

        # Step 4: Second GRE block
        signal[i, 4] = Mz_nrf(mz0=signal[i, 3], t1=t1, n_gre=NR_RF,
                              tr_gre=TR_GRE, alpha=FA_2, m0=M0)

        # Step 5: Final recovery with no pulses
        signal[i, 5] = Mz_0rf(mz0=signal[i, 4], t1=t1, t=TC, m0=M0)

    # Compute uni
    signal_gre1 = signal[:, 2]
    signal_gre2 = signal[:, 4]
    signal_uni = signal_gre1 * signal_gre2 / (signal_gre1**2 + signal_gre2**2)

    if only_bijective_part:
        idx_min = np.argmin(signal_uni)
        idx_max = np.argmax(signal_uni)
        arr_UNI = signal_uni[range(idx_min, idx_max, -1)]
        arr_T1 = T1s[range(idx_min, idx_max, -1)]
    else:
        arr_UNI = signal_uni
        arr_T1 = T1s

    return arr_UNI, arr_T1


def map_UNI_to_T1(img_UNI, arr_UNI, arr_T1):
    """Map from UNI values to T1 values.

    Parameters
    ----------
    img_UNI : numpy.ndarray, float
        Expects an array of measured UNI values. The value range should be in
        between -0.5 to 0.5.
    arr_UNI : np.ndarray, 1D, float
        Array of UNI values (MP2RAGE signal)
        **HAS TO BE IN ASCENDING ORDER, NOT AN ISSUE IF ONLY THE BIJECTIVE PART OF THE LOOKUP TABLE IS USED**
    arr_T1 : np.ndarray, 1D, float
        Array of T1 values that correspond to the UNI values.
        **HAS TO BE IN ASCENDING ORDER, NOT AN ISSUE IF ONLY THE BIJECTIVE PART OF THE LOOKUP TABLE IS USED**

    Returns
    -------
    img_T1 : numpy.ndarray, float
        Array of T1 values (in seconds) generated from the measured UNI values.

    """
    img_T1 = np.interp(img_UNI, xp=arr_UNI, fp=arr_T1)

    return img_T1

def coreg_and_resample_B1map(path2B1, path2mp2rage):

    print("Rigid registration and resampling of the B1 map to the MP2RAGE started.")
    # These are the images/objects, not the numpy 3D arrays
    fixed_image = sitk.ReadImage(path2mp2rage)
    fixed_image_float = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(path2B1) # You want to register and resample the B1 map to the MP2RAGE UNI image
    moving_image_float = sitk.Cast(moving_image, sitk.sitkFloat32)

    ## For later stage to output the results into a nibabel object get the affine.
    #moving_image_nib = nib.load(path2B1)

    # Rigid registration and Resampling (tusen takk til ChatGPT for denne)
    # Initial alignment
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # Setup registration method (can/could be optimized based on your requirements/data)
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetInterpolator(sitk.sitkBSpline) # MAF: Changed from linear to BSpline
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Execute registration
    final_transform = registration_method.Execute(fixed_image_float, moving_image_float)

    print(f"Optimizer's stopping condition: {registration_method.GetOptimizerStopConditionDescription()}")
    print(f"Final metric value: {registration_method.GetMetricValue()}")

    # Apply the final transform to the moving image
    moving_resampled = sitk.Resample(moving_image_float, fixed_image_float, final_transform, sitk.sitkBSpline, 0.0, moving_image_float.GetPixelID())

    # Save the resampled moving image
    dir2B1 = os.path.dirname(path2B1)
    B1filename_withnii = os.path.basename(path2B1)
    B1filename, extension = os.path.splitext(B1filename_withnii)
    new_B1filename = '%s_rigreg_resamp.nii' % (B1filename)
    new_b1pathname = '%s/%s' % (dir2B1,new_B1filename)
    sitk.WriteImage(moving_resampled, new_b1pathname)

    ### MAF: To be removed?
    # Change back to a nibabel image object (personal preference to work with nibabel images than sitk.Images)
    B1coreg_resamp_data = sitk.GetArrayFromImage(moving_resampled)

    # Get image origin and spacing from SimpleITK image
    origin = moving_resampled.GetOrigin()
    spacing = moving_resampled.GetSpacing()
    # Construct an affine transformation matrix
    affine_matrix = np.eye(4)
    affine_matrix[0, 0] = spacing[0]
    affine_matrix[1, 1] = spacing[1]
    affine_matrix[2, 2] = spacing[2]
    affine_matrix[0:3, 3] = origin

    B1coreg_resamp = nib.Nifti1Image(B1coreg_resamp_data, affine=affine_matrix)

    print("New registered and resampled B1 map saved to: %s" % new_b1pathname)
    print("Rigid registration and resampling of the B1 map to the MP2RAGE completed! :)")

    return B1coreg_resamp
