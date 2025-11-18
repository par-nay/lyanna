#! /usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from scipy.fft import fft, ifft, fftshift

N_params = 2  # (T0, gamma)

rel_T0_string    = r'$(T_0 - \hat{T}_0) / \hat{T}_0$'
rel_gamma_string = r'$(\gamma - \hat{\gamma}) / \hat{\gamma}$'

def rescale_T0(
        T0s, 
        mode = 'down', 
        alternative = False
    ):
    if mode == 'down':
        if alternative:
            return (T0s - 10500)/7500
        else:
            return (T0s - 10500)/5000
    elif mode == 'up':
        if alternative:
            return 7500*T0s + 10500
        else:
            return 5000*T0s + 10500
    else:
        raise ValueError('Invalid mode!')

        
def rescale_gamma(
        gammas, 
        mode = 'down', 
        alternative = False
    ):
    if mode == 'down':
        if alternative:
            return (gammas - 1.50)/0.3
        else:
            return (gammas - 1.48)/0.2
    elif mode == 'up':
        if alternative:
            return 0.3*gammas + 1.50
        else:
            return 0.2*gammas + 1.48
    else:
        raise ValueError('Invalid mode!')
        

def rescale_taueff_Planck18(
        taueffs, 
        mode = 'down'
    ):
    if mode == 'down':
        return (taueffs - 0.17)/0.05
    elif mode == 'up':
        return 0.05*taueffs + 0.17
    else:
        raise ValueError('Invalid mode!')

        
def rescale_chain(
        chain, 
        mode = 'down', 
        alternative = False
    ):
    if chain.shape[1] == N_params:
        chain = chain.T
    rescaled = np.array(
        [rescale_T0(
            chain[0, :], 
            mode = mode, 
            alternative = alternative
        ), 
        rescale_gamma(
            chain[1, :], 
            mode = mode, 
            alternative = alternative
        )]
    )
    if chain.shape[0] == N_params:
        rescaled = rescaled.T
    return rescaled

def rescale_noise_sigma(sigp): # the queried sigpq for nSansaQuery
    a = -7.5; b = -2.5
    return (2.*np.log(sigp) - (a+b))/(b-a)
    

v0 = np.array([-0.85902894, -0.51192703])  
v1 = np.array([-0.51192703, 0.85902894])
# V = np.array([v0, v1])   # Note that this is an involutory matrix, i.e., V^-1 = V
def change_basis(x, y):    # Because of the involutory nature of this operation, it can be used both ways without any changes
    a = v0[0]*x + v0[1]*y
    b = v1[0]*x + v1[1]*y
    return a, b


def cholesky_to_covariance(L):
    assert L.shape[1]==L.shape[2]==N_params
    if N_params == 2:
        c1s = L[:,0,0]
        c2s = L[:,1,1]
        c3s = L[:,1,0]
        C = np.zeros((len(c1s), N_params, N_params))
        C[:,0,0] = (c2s**2 + c3s**2)/(c1s*c2s)**2
        C[:,0,1] = -c3s/(c1s*c2s**2)
        C[:,1,0] = -c3s/(c1s*c2s**2)
        C[:,1,1] = 1/c2s**2
    else: 
        P = L @ L.transpose((0,2,1))
        C = np.linalg.inv(P)
    return C


def cholesky_to_precision(L):
    assert L.shape[1]==L.shape[2]==N_params
    if N_params == 2:
        c1s = L[:,0,0]
        c2s = L[:,1,1]
        c3s = L[:,1,0]
        P = np.zeros((len(c1s), 2, 2))
        P[:,0,0] = c1s**2
        P[:,0,1] = c1s*c3s
        P[:,1,0] = c1s*c3s
        P[:,1,1] = c2s**2 + c3s**2
    else:
        P = L @ L.transpose((0,2,1))
    return P



def wiener_deconvolve_flux_pdf_padded(
        pdf, 
        bin_edges_ext, 
        sigp, 
        gamma=1e-2
    ):
    """
    Perform Wiener deconvolution on a noisy flux PDF using FFT-based filtering with zero-padding.

    Parameters:
    - pdf (ndarray): The observed noisy PDF sampled on an extended bin grid.
    - bin_edges_ext (ndarray): The bin edges corresponding to the extended flux domain.
    - sigp (float): The standard deviation of the Gaussian noise kernel per pixel.
    - gamma (float): The Wiener filter regularization parameter (default: 1e-2).

    Returns:
    - pdf_deconv_crop (ndarray): The deconvolved and normalized PDF cropped to [0, 1].
    - bin_centers_crop (ndarray): The corresponding bin centers within [0, 1].
    """
    
    bin_centers_ext = bin_edges_ext[:-1] + 0.5 * np.diff(bin_edges_ext)
    dx = bin_centers_ext[1] - bin_centers_ext[0]
    N = len(bin_centers_ext)

    # Pad both PDF and kernel with zeros to double length
    pad_width = N
    pdf_padded = np.pad(pdf, (pad_width, pad_width), mode='constant')
    N_pad = len(pdf_padded)

    # Build matching padded Gaussian kernel
    x = (np.arange(N_pad) - N_pad // 2) * dx
    kernel = np.exp(-0.5 * (x / sigp)**2)
    kernel /= np.sum(kernel)

    # Perform Wiener deconvolution in Fourier space
    pdf_fft = fft(fftshift(pdf_padded))
    kernel_fft = fft(fftshift(kernel))
    denom = np.abs(kernel_fft)**2 + gamma
    pdf_deconv_fft = pdf_fft * np.conj(kernel_fft) / denom
    pdf_deconv_padded = np.real(fftshift(ifft(pdf_deconv_fft)))

    # Crop back to original domain (center slice)
    start = pad_width
    end = pad_width + N
    pdf_deconv = pdf_deconv_padded[start:end]

    # Normalize properly using fixed norm factor
    pdf_deconv *= 1.0 / (np.sum(pdf_deconv) * dx)

    # Crop to [0, 1] range
    mask = (bin_centers_ext >= 0.0) & (bin_centers_ext <= 1.0)
    pdf_deconv_crop  = pdf_deconv[mask]
    bin_centers_crop = bin_centers_ext[mask]

    return pdf_deconv_crop, bin_centers_crop




def bootstrap_subsample_means(
        summaries, 
        n_subsamples = 1000, 
        subset_size = 100, 
        seed = 42
    ):
    """
    Generate mean summary vectors from bootstrap subsampling.

    Parameters:
    - summaries (ndarray): Array of summary vectors with shape (n_samples, n_bins)
    - n_subsamples (int): Number of bootstrap subsamples to draw
    - subset_size (int): Number of samples per subsample
    - seed (int): Random seed for reproducibility

    Returns:
    - (ndarray): Array of mean summary vectors from each subsample
    """
    rng = np.random.default_rng(seed)
    mean_summaries = []

    for _ in range(n_subsamples):
        idx = rng.choice(
            len(summaries), 
            size = subset_size, 
            replace = False,
        )
        mean_summaries.append(np.mean(summaries[idx], axis=0))

    return np.array(mean_summaries)



def stratified_subsample_means(
        summaries, 
        subset_size  = 100, 
        n_iterations = 100, 
        seed = 42
    ):
    """
    Generate mean summary vectors using stratified disjoint subsampling.

    On each iteration, the full dataset is randomly shuffled and split into disjoint, non-overlapping
    subsets of size `subset_size`. The mean summary vector is computed for each subset.

    Parameters:
    - summaries (ndarray): Array of summary vectors with shape (n_samples, n_bins)
    - subset_size (int): Number of samples in each disjoint subset
    - n_iterations (int): Number of times to reshuffle and subsample the data
    - seed (int): Random seed for reproducibility

    Returns:
    - (ndarray): Array of mean summary vectors from all disjoint subsets across all iterations
    """
    rng       = np.random.default_rng(seed)
    n_total   = len(summaries)
    n_subsets = n_total // subset_size
    mean_summaries = []

    for _ in range(n_iterations):
        indices = rng.permutation(n_total)[:n_subsets * subset_size]
        reshaped = summaries[indices].reshape(n_subsets, subset_size, -1)
        mean_summaries.extend(np.mean(reshaped, axis=1))

    return np.array(mean_summaries)