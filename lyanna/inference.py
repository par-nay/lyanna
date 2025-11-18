#!/usr/bin/env python

import numpy as np
import emcee
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

from lyanna.utils import *
from lyanna.datautils import *


def bootstrap_aggregate(committee_predictions, N_params = 2):
    """ 
    Create a single prediction from a committee of models by bootstrap aggregating ("bagging") the individual predictions.

    Args:
        committee_predictions (_numpy.ndarray_): numpy.ndarray of shape (N_models, N_samples, N_params + N_params*(N_params+1)/2).
        In the last axis, the first N_params elements are the point estimates, and the remaining elements are the cholesky coefficients logL11, logL22, L21.

    Returns:    
        _numpy.ndarray_: Bagged prediction from the committee of models, of shape (N_samples, N_params).
    """
    N_models  = committee_predictions.shape[0]
    N_samples = committee_predictions.shape[1]
    L11s      = np.exp(committee_predictions[:, :, N_params])
    L22s      = np.exp(committee_predictions[:, :, N_params + 1])
    dets_icov = L11s**2 * L22s**2
    weights   = dets_icov**(1/N_params)
    bagged_predictions = np.zeros((N_samples, N_params))
    for n in range(N_params):
        bagged_predictions[:,n] = np.average(
            committee_predictions[:, :, n], 
            axis = 0, 
            weights = weights,
        )
    return bagged_predictions



def figure_of_merit(posterior_chain, N_params = 2):
    if posterior_chain.shape[0]==N_params and posterior_chain.shape[1]!=N_params:
        rowvar = True
    elif posterior_chain.shape[1]==N_params and posterior_chain.shape[0]!=N_params:
        rowvar = False 
    elif posterior_chain.shape[0]==posterior_chain.shape[1]:
        raise ValueError("Unable to compute FoM: the chain doesn't have an adequate shape!") 
    cov = np.cov(
        posterior_chain, 
        rowvar = rowvar,
    )
    assert cov.shape[0] == N_params 
    det = np.linalg.det(cov)
    FoM = 1./ (det)**(1/N_params)
    return FoM



def chisquared_delta_posterior(posterior_chain, truth_vector, N_params = 2):
    if posterior_chain.shape[0]==N_params and posterior_chain.shape[1]!=N_params:
        rowvar = True
    elif posterior_chain.shape[1]==N_params and posterior_chain.shape[0]!=N_params:
        rowvar = False 
    elif posterior_chain.shape[0]==posterior_chain.shape[1]:
        raise ValueError("Unable to compute FoM: the chain doesn't have an adequate shape!")    
    cov = np.cov(posterior_chain, rowvar = rowvar)
    assert cov.shape[0] == N_params 
    inv_cov = np.linalg.inv(cov)
    if rowvar:
        delta   = posterior_chain.T - truth_vector
    else:
        delta   = posterior_chain - truth_vector
    chisq = np.einsum(
        'ij,jk,ki->i', 
        delta, 
        inv_cov, 
        delta.T,
    )
    return np.mean(chisq)


def gaussian_log_likelihood_fixed_C(
    data, 
    model, 
    C_inv
):
    """
    Ignores the log determinant term since the covariance is assumed to be fixed w.r.t. the parameters
    """
    delta = data - model
    #print(delta.shape)
    return -delta.dot(C_inv.dot(delta))/2.


def gaussian_log_likelihood_model_dependent_C(
    data, 
    model, 
    C_inv
):
    delta = data - model
    #print(delta.shape)
    s, logdet = np.linalg.slogdet(C_inv)
    if s <= 0: 
        raise ValueError("Invalid value encountered in log_likelihood_model_dependent_C: C_inv has non-positive determinant!")
    else: 
        return -delta.dot(C_inv.dot(delta))/2. + logdet/2.
    

gaussian_log_likelihood_full = gaussian_log_likelihood_model_dependent_C


class MCMC:
    def __init__(
            self, 
            log_prob : callable,
            N_params : int = 2,
            truth:list|tuple = None, # upscaled 
        ):
        self.log_prob = log_prob
        self.N_params = N_params
        self.truth    = truth

    def run_chain(
            self, 
            N_walkers : int = 100,
            N_iterations : int = 100, 
            init_dist_gauss_centers:list = [0.0, 0.0],
            init_dist_gauss_sigmas:list  = [0.01, 0.01],
            burn_in_chain_fraction:float = 0.5,
            seed = 42,
            show_progress = False,
            **sampler_kwargs,
        ):
        self.sampler = emcee.EnsembleSampler(
            N_walkers, 
            self.N_params, 
            self.log_prob, 
            **sampler_kwargs
        )
        RS = np.random.mtrand.RandomState(seed = seed).get_state()
        # print(Sampler.random_state)
        # print(RS)
        self.sampler.random_state = RS
        # print(Sampler.random_state)
        # Sampler.random_state(RS)
        rng = np.random.default_rng(seed = 100+seed)
        p0 = rng.standard_normal((N_walkers, self.N_params))*init_dist_gauss_sigmas + init_dist_gauss_centers
        self.sampler.run_mcmc(
            p0, 
            N_iterations, 
            progress = show_progress
        )
        # print(Sampler.random_state)
        # chain  = Sampler.get_chain()
        self.chain  = self.sampler.flatchain[int(N_iterations*N_walkers*burn_in_chain_fraction):, :]
        return self.chain
    
    def get_chain(
            self, 
            N_skewers_rescale : int = 100,
            N_skewers_actual : int = 100,
            rel_chain : bool = True,
            alternative_rescaling : bool = False,
        ):
        try:
            chain = self.chain
        except AttributeError:
            raise ValueError("No chain found! Run `run_chain()` first to generate a chain.")
        self.N_skewers_actual = N_skewers_actual
        means       = np.mean(chain, axis = 0)
        deltachain  = self.chain - means
        deltachain *= np.sqrt(self.N_skewers_actual/N_skewers_rescale) 
        chain_rescaled = deltachain + means
        chain_rescaled = rescale_chain(
            chain_rescaled, 
            mode = 'up', 
            alternative = alternative_rescaling
        )
        if rel_chain:
            assert self.truth is not None, "To get a relative chain, you must supply the truth vector when initializing the MCMC object."
            for i in range(self.N_params):
                chain_rescaled[:,i] = chain_rescaled[:,i] / self.truth[i] - 1
        return chain_rescaled
    


class LyaNNA:
    def __init__(
        self,
        model : tf.keras.Model,
        trained_committee_weights : list,
        is_in_prior : callable,
        emulator : callable = None,
        delfi = None,
    ):
        """
        The central class for inference with LyaNNA. Supports the noiseless version "Sansa" (based on Nayak et al. 2024) and the noisy version "nSansa" (based on Nayak et al. 2025)."

        Args:
            model (tf.keras.Model): the keras model object, created with `architecture.create_Sansa()` or `architecture.create_nSansa()`
            trained_committee_weights (list): list of weight files (strings) of the trained committee member networks, each one should be a keras readable weight file, to be loaded with `model.load_weights()`.
            is_in_prior (callable): the (flat) prior function, should take as input the downscaled parameters and return True if the parameters are in your prior, False otherwise.
            emulator (callable, optional): the model emulator in the likelihood case. Should take as input the downscaled parameters and return the model summary vector. Defaults to None. Must supply when using likelihood based inference.
            delfi (sklearn.mixture.GaussianMixture, optional): applicable only to nSansa that supports DELFI. The trained GMM object to be used for inference if desired. Defaults to None. Must supply when using likelihood free inference.
        """
        self.model = model
        self.trained_committee_weights = trained_committee_weights
        self.N_members_committee = len(trained_committee_weights)
        self.is_in_prior = is_in_prior
        self.emulator = emulator
        self.delfi = delfi

    def predict(
        self,
        spectra : np.ndarray,
        noise_sigmas : np.ndarray = None,
        steps : int = 10,
        verbose : int = 0, # keras model.predict verbosity
        W_matrices : np.ndarray = None,
        d_vectors : np.ndarray = None,
    ):
        """
        Run the LyaNNA compression on the supplied spectra.

        Args:
            spectra (np.ndarray): spectra to run the model prediction on, shape (N_spectra, N_pixels, 1)
            noise_sigmas (np.ndarray, optional): only when using nSansaQuery, supply the homoscedastic noise level in gaussian standard deviation (per spectrum) for the actual pixel size, shape (N_spectra,). Defaults to None.
            steps (int, optional): number of steps for `model.predict`. Defaults to 10.
            verbose (int, optional): `model.predict` verbosity. See the keras documentation for more details. Defaults to 0 (silent). In addition, a fourth option '3' for a single line updating over committee members, such as '[0121/2200]'.
            W_matrices (np.ndarray, optional): must be provided for Sansa, the linear transformation matrices for all committee members, shape (N_members, 2, 2). Defaults to None.
            d_vectors (np.ndarray, optional): must be provided for Sansa, the linear transformation bias vectors for all committee members, shape (N_members, 2). Defaults to None.
        Returns:
            np.ndarray: the compressed summary vectors, shape (N_spectra, N_params)
        """
        if verbose == 3:
            verbose = 0
            verbose_line = True
        committee_predictions = []
        for i, weights in enumerate(self.trained_committee_weights):
            if verbose_line:
                print(f'[{i+1:04d}/{len(self.trained_committee_weights):04d}]', end='\r')
            self.model.load_weights(weights)
            if noise_sigmas is not None:
                input = [spectra, noise_sigmas]
            else:
                input = spectra 
            preds = self.model.predict(
                input,
                steps = steps,
                verbose = verbose,
            )
            committee_predictions.append(preds)
        committee_predictions = np.array(committee_predictions)
        if W_matrices is not None and d_vectors is not None:
            # This means the model is Sansa
            committee_predictions_transformed = np.zeros((*committee_predictions.shape[:2], 2))
            for i in range(self.N_members_committee):
                committee_predictions_transformed[i,:,0] = W_matrices[i,0,0]*committee_predictions[i,:,0] + W_matrices[i,0,1]*committee_predictions[i,:,1] + d_vectors[i,0]
                committee_predictions_transformed[i,:,1] = W_matrices[i,0,1]*committee_predictions[i,:,0] + W_matrices[i,1,1]*committee_predictions[i,:,1] + d_vectors[i,1]
            L11s      = np.exp(committee_predictions[:, :, 2])
            L22s      = np.exp(committee_predictions[:, :, 3])
            dets_icov = []
            for i in range(self.N_members_committee):
                dets_icov.append(L11s[i]**2 * L22s[i]**2 / np.linalg.det(W_matrices[i])**2)
            dets_icov = np.array(dets_icov)
            bagged_predictions = np.zeros((committee_predictions.shape[1], 2))
            # for n in range(2):
            #     bagged_predictions[:,n] = np.average(
            #         committee_predictions_transformed[:, :, n], 
            #         axis = 0, 
            #         weights = dets_icov,
            #     )
            for j in range(len(bagged_predictions)):
                bagged_predictions[j] = np.average(committee_predictions_transformed[:,j,:2], axis = 0, weights = dets_icov[:,j])
        else:
            # The model is nSansa
            bagged_predictions    = bootstrap_aggregate(committee_predictions)
        return bagged_predictions 
    
    def infer(
        self,
        mean_compressed_summary : np.ndarray,
        method : str, 
        inv_cov = None,
        noise_sigma : float = None,
        truth : list|tuple = None,
        **mcmc_kwargs,
    ):
        """
        Perform Baysian inference with the compressed summary vector provided and the method specified. Currently supports Gaussian likelihood and DELFI (latter only for nSansa).

        Args:
            mean_compressed_summary (np.ndarray): the summary vector for the actual data to perform inference with, shape (N_params,).
            method (str): 'lkd' (Gaussian likelihood) or 'delfi' (DELFI). DELFI is only supported for nSansa, but the method will allow you to run it with Sansa as well, the results will not be meaningful.
            inv_cov (optional): while doing Gaussian likelihood based inference, the inverse covariance matrix of the summary vector. Supply either the fixed inverse covariance (np.ndarray) or an emulator (callable) that takes as input the downscaled parameters and returns the inverse covariance.
            noise_sigma (float, optional): for DELFI, the (mean) homoscedastic noise level on the input spectra in gaussian standard deviation for the actual pixel size. Defaults to None. Must supply when using nSansa.
            truth (list|tuple, optional): the true parameter vector (upscaled=original) if known, used to get the relative chains. Defaults to None.
            **mcmc_kwargs: keyword arguments to be passed to the MCMC `run_chain()` method.
        Returns:
            lyanna.inference.MCMC: instance of the MCMC class after running the chain as requested.
        """
        if method == 'lkd':
            if not (self.emulator is not None and inv_cov is not None):
                raise ValueError("To use likelihood based inference, you must supply both a model emulator and an inverse covariance.")
            else:
                if callable(inv_cov):
                    log_prob = lambda params: gaussian_log_likelihood_model_dependent_C(
                        data = mean_compressed_summary,
                        model = self.emulator(*params),
                        C_inv = inv_cov(*params), 
                    ) if self.is_in_prior(*params) else -np.inf
                elif isinstance(inv_cov, np.ndarray):
                    log_prob = lambda params: gaussian_log_likelihood_fixed_C(
                        data = mean_compressed_summary,
                        model = self.emulator(*params),
                        C_inv = inv_cov,
                    ) if self.is_in_prior(*params) else -np.inf
                else:
                    raise ValueError("inv_cov must be either a callable (emulator) or a numpy.ndarray (fixed inverse covariance matrix).")
        elif method == 'delfi':
            if self.delfi is None:
                raise ValueError("To use DELFI for inference, you must supply a trained GMM when initializing the LyaNNA object.")
            else:
                sigpq    = rescale_noise_sigma(noise_sigma)
                log_prob = lambda params: self.delfi.score_samples(
                    [[*mean_compressed_summary, *params, sigpq]]
                )[0] if self.is_in_prior(*params) else -np.inf
        else:
            raise ValueError("Invalid method specified. Choose either 'lkd' or 'delfi'.")
        mcmc = MCMC(
            log_prob,
            N_params = mean_compressed_summary.shape[0],
            truth = truth,
        )
        mcmc.run_chain(
            **mcmc_kwargs
        )
        return mcmc
    
    def compute_posterior_metrics(
        self,
        posterior_chain : np.ndarray,
        N_params : int = 2,
        truth : list|tuple = None,
    ):
        """
        Compute the posterior metrics as described in Nayak et al. (2025)

        Args:
            posterior_chain (np.ndarray): the posterior chain to compute the metrics for
            N_params (int, optional): number of parameters inferred. Defaults to 2.
            truth (list|tuple, optional): the true parameter vector (upscaled=original) if known, used to compute the chi-squared delta metric. If not provided, the chisquared metric is not computed. Defaults to None. 
        
        Returns:
            dict: metrics stored in a dictionary with keys 'fom' and 'deltachisq_r'
        """
        fom = figure_of_merit(posterior_chain, N_params = N_params)
        if truth is not None:
            chisq = chisquared_delta_posterior(
                posterior_chain, 
                truth,
                N_params = N_params,
            )
        else:
            chisq = np.nan
        deltachisq_r = chisq / N_params - 1
        return {
            'fom' : fom,
            'deltachisq_r' : deltachisq_r,
        }
