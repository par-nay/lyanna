# LyαNNA
### Lyman- $\alpha$ Neural Network Analysis

LyαNNA is a deep learning framework for field-level inference with the Lyα forest. The framework is presented in  Nayak et al. ([2024](https://arxiv.org/abs/2311.02167), [2025](https://arxiv.org/abs/2510.19899)).

## Quickstart

Install the package with pip
```
pip install git+https://github.com/par-nay/lyanna.git
```

Create a NN model
```Python
nsansa = lyanna.architecture.create_nSansa()
```

Download assets like the trained model weights and emulator(s)
```Python
lyanna.assets.get_assets(assets_path)
```

Run NN compression of your spectra
```Python
lyanna_inf = lyanna.inference.LyaNNA(
    nsansa, 
    weights_list, 
    is_in_prior = is_in_prior, 
    emulator = nsansa_emulator, 
    delfi = nsansa_delfi
)
nsansa_summaries = lyanna_inf.predict(
    noisy_spectra,
    noise_sigmas = sigmas_per_spectrum
)
```

Compute covariance matrix
```Python
stratified_summaries = lyanna.utils.stratified_subsample_means(
    nsansa_summaries, 
    subset_size  = 100, 
    n_iterations = 1_000,
)
covar = np.cov(
    stratified_summaries, 
    rowvar = False
)
```

Run inference with Gaussian likelihood
```Python
mcmc = lyanna_inf.infer(
    np.mean(nsansa_summaries, axis = 0),
    method = 'lkd',
    noise_sigma = sigma_p_scalar,
    inv_cov = np.linalg.inv(covar),
    N_walkers = 100,
    N_iterations = 5_000
)
lkd_chain = mcmc.get_chain()
```

Plot contours
```Python
lyanna.plotting.plot_posterior_contours_from_chains(
    {'Likelihood': lkd_chain, 'DELFI': delfi_chain,}
)
```

Compute posterior metrics
```Python
metrics = lyanna_inf.compute_posterior_metrics(lkd_chain)
figure_of_merit = metrics['fom']
```

## Tutorial
Check out the notebook `tutorial.ipynb` for a lightning tutorial of inference with LyαNNA.

## Getting in touch
Any feedback is welcome and highly appreciated! Drop me an email at [parth3e8@gmail.com](mailto:parth3e8@gmail.com).
