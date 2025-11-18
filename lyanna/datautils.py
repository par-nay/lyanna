#! /usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import h5py
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from lyanna.utils import *
from numpy.fft import irfft, rfft


def discard_large_k_modes(
    spectra, 
    cutoff_k_idx = 122+1, #256+1 for Sansa, 122+1 for nSansa
):
    rfft_skewers = rfft(spectra)
    rfft_skewers[:, cutoff_k_idx:] = 0j
    irfft_skewers = irfft(rfft_skewers)
    return irfft_skewers


def discard_large_k_modes_and_smooth(
    spectra, 
    k, 
    cutoff_k_idx = 122+1, #256+1 for Sansa, 122+1 for nSansa
    R_FWHM = 6000, #10820.21 for Sansa, 6000 for nSansa
):
    rfft_skewers  = rfft(spectra)
    v_sigma       = 2.998E5 / R_FWHM / 2.35482
    kernel        = np.exp(-v_sigma**2 * k**2 / 2.)
    rfft_skewers *= kernel
    rfft_skewers[:, cutoff_k_idx:] = 0j
    irfft_skewers = irfft(rfft_skewers)
    return irfft_skewers


def extract_subset(
    full_dataset, 
    N_files, 
    full_size_each, 
    subset_size_each, 
    start_id_each = 0,
):
    subset = np.zeros(full_dataset.shape)[:subset_size_each*N_files, :]
    for i in range(N_files):
        subset[i*subset_size_each : (i+1)*subset_size_each, :] = full_dataset[start_id_each+i*full_size_each : start_id_each+i*full_size_each + subset_size_each, :]
    return subset


def roll_spectral_dataset(
    spectral_dataset, 
    N_roll = 256, 
    N_pix_spectrum = 512
): # spectral_dataset needs to be in lyanna dataset format
    new_set = np.copy(spectral_dataset)
    new_set[:, :N_pix_spectrum] = np.roll(spectral_dataset[:,:N_pix_spectrum], N_roll, axis = 1)
    return new_set


def roll_spectral_dataset_randomly(
    spectral_dataset, 
    N_pix_spectrum = 512, 
    seed = 100
): # spectral_dataset needs to be in lyanna dataset format
    new_set = np.copy(spectral_dataset)
    np.random.seed(seed)
    N_roll_arr = np.random.randint(0, N_pix_spectrum, spectral_dataset.shape[0])
    for i in range(spectral_dataset.shape[0]):
        new_set[i, :N_pix_spectrum] = np.roll(spectral_dataset[i,:N_pix_spectrum], N_roll_arr[i], axis = 0)
    return new_set


def flip_spectral_dataset(
    spectral_dataset, 
    N_pix_spectrum = 512, 
    randomly = False, 
    seed = 100, 
    p = 0.5
):
    new_set = np.copy(spectral_dataset)
    if randomly:
        np.random.seed(seed)
        # yesno = 2*np.random.randint(0, 2, spectral_dataset.shape[0]) - 1
        yesno = np.random.choice([-1, 1], size = spectral_dataset.shape[0], p = [p, 1-p])
        for i in range(spectral_dataset.shape[0]):
            new_set[i, :N_pix_spectrum] = spectral_dataset[i,:N_pix_spectrum][::yesno[i],:]
    else:
        new_set[:, :N_pix_spectrum] = spectral_dataset[:,:N_pix_spectrum][:,::-1,:]
    return new_set


class Noise:
    def __init__(
        self, 
        SNR, 
        v_h_skewer, 
        CNR = True, 
        N_pix_spectrum = 512
    ):
        self.SNR            = SNR
        self.v_h_skewer     = v_h_skewer
        # self.CNR            = CNR
        self.N_pix_spectrum = N_pix_spectrum
        if CNR:
            self.sigv = 1./SNR
        self.R        = (v_h_skewer[1] - v_h_skewer[0]) * (4096/N_pix_spectrum)*1e-5 /6.
        self.sigp     = tf.convert_to_tensor(np.sqrt(self.R)*self.sigv)
        self.seed     = 0
        
    def add_noise(
        self, 
        spectral_inputs
    ):
        # np.random.seed(self.seed)
        tf.random.set_seed(self.seed);
        noise         = tf.random.normal(tf.shape(spectral_inputs), 0, self.sigp, tf.float64)
        noisy_inputs  = tf.identity(spectral_inputs)
        # noise      = np.random.normal(0.0, self.sigp, spectral_dataset[:, :self.N_pix_spectrum, :].shape)
        noise         = tf.dtypes.cast(noise, tf.float32)
        noisy_inputs  = tf.math.add(noisy_inputs, noise)
        self.seed    += 1
        return noisy_inputs



class Smoothing:
    def __init__(self, delta_v_pixel, k_257, R_FWHM_min = 6000, R_FWHM_max = 11000, seed = 0):
        self.seed = seed
        self.R_FWHM_min = R_FWHM_min
        self.R_FWHM_max = R_FWHM_max
        self.delta_v_pixel = delta_v_pixel
        self.k = tf.constant(k_257, dtype = tf.complex64)
        self.R_FWHM_fid = 10820.21
        self.sigv_fid   = 2.998E5 / self.R_FWHM_fid / 2.35482

    def gaussian_kernel_tf(self, size, sigma):
        x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
        g = tf.exp(-tf.pow(x / sigma, 2) / 2)
        return g / tf.reduce_sum(g)
    
    def pad_1d_periodic_tf(self, tensor, pad_width_total):
        tensor_shape = tf.shape(tensor)
        pad_width_left  = pad_width_total // 2
        pad_width_right = pad_width_total - pad_width_left
        tensor_len = tensor_shape[0]
        
        left_pad_indices  = tensor_len + tf.range(pad_width_left) - pad_width_left 
        right_pad_indices = tf.range(pad_width_right) 

        left_pad_values = tf.gather(tensor, left_pad_indices)
        right_pad_values = tf.gather(tensor, right_pad_indices)

        padded_tensor = tf.concat([left_pad_values, tensor, right_pad_values], axis=0)
        return padded_tensor
    
    def smooth_1d_tensor(self, tensor, kernel_size, sigma):
        pad_width_total = kernel_size - 1
        padded_tensor   = self.pad_1d_periodic_tf(tensor, pad_width_total)
        
        kernel = self.gaussian_kernel_tf(kernel_size, sigma)
        kernel = tf.reshape(kernel, [kernel_size, 1, 1])
        padded_tensor = tf.expand_dims(tf.expand_dims(padded_tensor, axis=0), axis=-1)
        padded_tensor = tf.cast(padded_tensor, dtype=tf.float32)
        smoothed = tf.nn.conv1d(padded_tensor, kernel, stride=1, padding='VALID')
        return tf.squeeze(smoothed, axis=-1)
        
    def smooth(self, spectral_inputs):
        tf.random.set_seed(self.seed)
        R_FWHM = tf.random.uniform(shape=[], minval=self.R_FWHM_min, maxval=self.R_FWHM_max, dtype=tf.float32)
        sigv   = 2.998E5 / R_FWHM / 2.35482
        sigp   = sigv / self.delta_v_pixel
        kernel_size = tf.cast(8*sigp, dtype = tf.int32)
        smooth_spectra = tf.map_fn(
            lambda tensor: self.smooth_1d_tensor(tensor, kernel_size, sigp), 
            spectral_inputs[:,:,0], 
            dtype = tf.float32
        )
        smooth_spectral_inputs = tf.reshape(smooth_spectra, tf.shape(spectral_inputs))
        self.seed += 1
        return smooth_spectral_inputs
    
    def smooth_fourier_space(self, spectral_inputs):
        original_shape  = spectral_inputs.shape
        spectral_inputs = tf.reshape(spectral_inputs, [spectral_inputs.shape[0], spectral_inputs.shape[1]])
        fft_spectra = tf.signal.rfft(spectral_inputs)
        tf.random.set_seed(self.seed)
        RNG = np.random.default_rng(seed = self.seed)
        R_FWHM = RNG.uniform(low = self.R_FWHM_min, high = self.R_FWHM_max) #tf.random.uniform(shape=[], minval=self.R_FWHM_min, maxval=self.R_FWHM_max, dtype=tf.float32)
        sigv   = 2.998E5 / R_FWHM / 2.35482
        sigv   = np.sqrt(sigv**2 - self.sigv_fid**2)
        self.sigv = sigv
        # sigv   = tf.cast(sigv, dtype = tf.complex64)
        fft_kernel         = tf.exp(-tf.pow(self.k*sigv, 2) / 2)
        smooth_fft_spectra = tf.multiply(fft_spectra, fft_kernel)
        smooth_spectra     = tf.signal.irfft(smooth_fft_spectra)
        smooth_spectra     = tf.reshape(smooth_spectra, original_shape)
        self.seed += 1
        return smooth_spectra
    
    def smooth_validation_fourier_space(self, spectral_inputs):
        original_shape  = spectral_inputs.shape
        spectral_inputs = tf.reshape(spectral_inputs, [spectral_inputs.shape[0], spectral_inputs.shape[1]])
        fft_spectra = tf.signal.rfft(spectral_inputs)
        fft_kernel         = tf.exp(-tf.pow(self.k*self.sigv, 2) / 2)
        smooth_fft_spectra = tf.multiply(fft_spectra, fft_kernel)
        smooth_spectra     = tf.signal.irfft(smooth_fft_spectra)
        smooth_spectra     = tf.reshape(smooth_spectra, original_shape)
        self.seed += 1
        return smooth_spectra
    



class Augmentation:
    def __init__(
        self, 
        p_flip, 
        seed = 0
    ):
        self.p_flip = p_flip
        self.seed   = seed
    

    def init_noise(
        self, 
        SNR, 
        v_h_skewer, 
        CNR = True, 
        N_pix_spectrum = 512, 
        SNR_max = None, 
        mode = 'p0s0', 
        batch_size = None, 
        sigma_spread_factor = None,
    ):
        self.SNR            = SNR
        self.v_h_skewer     = v_h_skewer
        # self.CNR            = CNR
        self.mode           = mode
        self.N_pix_spectrum = N_pix_spectrum
        self.R        = (v_h_skewer[1] - v_h_skewer[0]) * (4096/N_pix_spectrum)*1e-5 /6.
        if CNR:
            self.sigv = 1./SNR
            if mode == 'p0s1':
                self.sigv_min = 1./SNR_max
                self.sigp_min = tf.convert_to_tensor(self.sigv_min / np.sqrt(self.R))
                self.means_noise = tf.zeros((batch_size, 1), dtype = tf.float64)
            elif mode == 'p1s1':
                self.sigma_spread_factor = tf.convert_to_tensor(sigma_spread_factor, dtype = tf.float64)
                self.means_noise = tf.zeros((batch_size, N_pix_spectrum, 1), dtype = tf.float64)
        self.sigp     = tf.convert_to_tensor(self.sigv / np.sqrt(self.R))
        # self.seed     = self.seed


    def roll(
        self, 
        spectral_inputs, 
        random_indices
    ):
        rows, cols, _ = spectral_inputs.shape
        shifts = tf.repeat(tf.expand_dims(random_indices, axis = 1), cols, axis = 1)
        row_indices = tf.tile(tf.expand_dims(tf.range(rows), axis = 1), [1, cols])
        indices = tf.stack([row_indices, (tf.range(cols) - shifts)%cols], axis = -1)
        return tf.gather_nd(spectral_inputs, indices)
    

    def flip(
        self, 
        spectral_inputs, 
        random_indices
    ):
        return tf.tensor_scatter_nd_update(spectral_inputs, tf.expand_dims(random_indices, axis = 1), tf.reverse(tf.gather(spectral_inputs, random_indices), axis = [1]))
    

    def augment_without_noise(
        self, 
        spectral_inputs
    ):
        tf.random.set_seed(self.seed)
        I = tf.range(spectral_inputs.shape[0])
        N = int(spectral_inputs.shape[0]*self.p_flip)
        indices_roll = tf.random.uniform(shape=(spectral_inputs.shape[0],), minval=1, maxval=spectral_inputs.shape[1], dtype=tf.int32)
        indices_flip = tf.random.shuffle(I)[:N] #self.RNG_f.choice(a, size = int(spectral_inputs.shape[0]*self.p_flip), replace = False)
        spectra          = self.flip(spectral_inputs, indices_flip)
        spectral_outputs = self.roll(spectra, indices_roll)
        self.seed += 1
        return spectral_outputs
    

    def augment_with_noise(
        self, 
        spectral_inputs
    ):
        tf.random.set_seed(self.seed)
        # print(spectral_inputs.shape)
        I = tf.range(spectral_inputs.shape[0])
        N = int(spectral_inputs.shape[0]*self.p_flip)
        indices_roll = tf.random.uniform(shape=(spectral_inputs.shape[0],), minval=1, maxval=spectral_inputs.shape[1], dtype=tf.int32)
        indices_flip = tf.random.shuffle(I)[:N] #self.RNG_f.choice(a, size = int(spectral_inputs.shape[0]*self.p_flip), replace = False)
        spectra          = self.flip(spectral_inputs, indices_flip)
        spectral_outputs = self.roll(spectra, indices_roll)
        try:
            if self.mode == 'p0s0':
                noise     = tf.random.normal(tf.shape(spectral_inputs), 0, self.sigp, tf.float64)
            elif self.mode == 'p0s1':
                sigps     = tf.random.uniform((tf.shape(spectral_inputs)[0], tf.shape(spectral_inputs)[-1]), minval = self.sigp_min, maxval = self.sigp, dtype =  tf.float64)
                noise     = tf.random.normal((tf.shape(spectral_inputs)[1], tf.shape(spectral_inputs)[0],tf.shape(spectral_inputs)[-1]), self.means_noise, sigps, tf.float64)
                noise = tf.transpose(noise, perm = [1,0,2])
            elif self.mode == 'p1s1':
                sigps     = tf.random.normal((tf.shape(spectral_inputs)[-3], tf.shape(spectral_inputs)[-2], tf.shape(spectral_inputs)[-1]), self.sigp, self.sigp*self.sigma_spread_factor, dtype =  tf.float64)
                noise     = tf.random.normal((tf.shape(spectral_inputs)[-3], tf.shape(spectral_inputs)[-2],tf.shape(spectral_inputs)[-1]), self.means_noise, sigps, tf.float64)
                # noise = tf.transpose(noise, perm = [1,0,2])
        except NameError:
            raise NotImplementedError("Noise not initialized! Run ClassInstance.init_noise(__) properly before trying to add noise to the input spectra.")

        noise     = tf.dtypes.cast(noise, tf.float32)
        noisy_inputs  = tf.identity(spectral_outputs)
        noisy_inputs  = tf.math.add(noisy_inputs, noise)
        self.seed    += 1
        if self.mode == 'p0s0':
            return noisy_inputs
        elif self.mode in ['p0s1', 'p1s1']:
            return noisy_inputs, sigps
    
    
    def add_noise(
        self, 
        spectral_inputs, 
        n_realizations = 1
    ):
        tf.random.set_seed(self.seed);
        # print(spectral_inputs.shape)
        try:
            if self.mode == 'p0s0':
                noise         = tf.random.normal(tf.shape(spectral_inputs), 0, self.sigp, tf.float64)
            elif self.mode == 'p0s1':
                sigps     = tf.random.uniform((tf.shape(spectral_inputs)[0], tf.shape(spectral_inputs)[-1]), minval = self.sigp_min, maxval = self.sigp, dtype =  tf.float64)
                noise     = tf.random.normal((tf.shape(spectral_inputs)[1], tf.shape(spectral_inputs)[0],tf.shape(spectral_inputs)[-1]), self.means_noise, sigps, tf.float64)
                noise     = tf.transpose(noise, perm = [1,0,2])
                if n_realizations == 2:
                    noise2    = tf.random.normal((tf.shape(spectral_inputs)[1], tf.shape(spectral_inputs)[0],tf.shape(spectral_inputs)[-1]), self.means_noise, sigps, tf.float64)
                    noise2    = tf.transpose(noise2, perm = [1,0,2])
            elif self.mode == 'p1s1':
                sigps     = tf.random.normal((tf.shape(spectral_inputs)[-3], tf.shape(spectral_inputs)[-2], tf.shape(spectral_inputs)[-1]), self.sigp, self.sigp*self.sigma_spread_factor, dtype =  tf.float64)
                noise     = tf.random.normal((tf.shape(spectral_inputs)[-3], tf.shape(spectral_inputs)[-2],tf.shape(spectral_inputs)[-1]), self.means_noise, sigps, tf.float64)
                if n_realizations == 2:
                    noise2    = tf.random.normal((tf.shape(spectral_inputs)[-3], tf.shape(spectral_inputs)[-2],tf.shape(spectral_inputs)[-1]), self.means_noise, sigps, tf.float64)
        except NameError:
            raise NotImplementedError("Noise not initialized! Run ClassInstance.init_noise(__) properly before trying to add noise to the input spectra.")
        
        # noisy_inputs  = tf.identity(spectral_inputs)
        noise         = tf.dtypes.cast(noise, tf.float32)
        noisy_inputs  = tf.math.add(spectral_inputs, noise)
        if n_realizations == 2:
            noise2        = tf.dtypes.cast(noise2, tf.float32)
            noisy_inputs2 = tf.math.add(spectral_inputs, noise2)
        self.seed    += 1
        if self.mode == 'p0s0':
            return noisy_inputs
        elif self.mode in ['p0s1', 'p1s1']:
            if n_realizations == 2:
                return noisy_inputs, noisy_inputs2, sigps
            elif n_realizations == 1:
                return noisy_inputs, sigps
            



def get_test_data_for_inference(
    filepath, # path to the input hdf5 file containing spectra computed using SynTH
    N_sk_to_pick, 
    sk_start_id = 0, 
    smooth = True, 
    alternative = False, 
    obs_mean_flux = True, 
    R_FWHM = 6000, # 10820.21 for Sansa, 6000 for nSansa
    cutoff_k_idx = 122+1, # 256+1 for Sansa, 122+1 for nSansa
    N_pix_spectrum = 256, # 512 for Sansa, 256 for nSansa 
):
    test_spectra = np.zeros(
        (N_sk_to_pick, N_pix_spectrum+5, 1), 
        dtype = 'float32',
    )
    with h5py.File(filepath, 'r') as f:
        if obs_mean_flux:
            A_z_0  = f['A_tau_rescaling/A_z_0'][0]
        else:
            A_z_0  = 1.
        F_sk   = np.exp( -A_z_0 * f['tau'][sk_start_id : sk_start_id + N_sk_to_pick, :])
        k_full = f['k_in_skminv'][:]
        T0, gamma = f['tdr_params'][:]
    T0    = rescale_T0(T0, alternative = alternative)
    gamma = rescale_gamma(gamma, alternative = alternative)
    if smooth: 
        F_sk  = discard_large_k_modes_and_smooth(
            F_sk, 
            k_full, 
            R_FWHM = R_FWHM, 
            cutoff_k_idx = cutoff_k_idx,
        )
    else: 
        F_sk  = discard_large_k_modes(
            F_sk, 
            cutoff_k_idx = cutoff_k_idx,
        )
    F_sk  = np.mean(
        F_sk.reshape((-1, N_pix_spectrum, F_sk.shape[1] // N_pix_spectrum)), 
        axis = -1,
    )
    test_spectra[:, :N_pix_spectrum, 0]    = F_sk
    test_spectra[:, N_pix_spectrum :N_pix_spectrum+2, 0] = T0, gamma
    return test_spectra