# Imports:
import numpy as np
import cupy as cp
import scipy as sp
from tqdm import tqdm
from kuramoto import KuramotoFast, KuramotoFastDelayed

# Function definitions:
def simulate_single_run(k, node_frequencies, time=350, n_oscillators=500, frequency_spread=3, normalize_external=False, aggregate='mean', noise_sigma=3.0, sr=250, n_nodes=200, omegas=None, weight_matrix=None, use_tqdm=True, omega_seed=42, random_seed=42, delay_matrix = None, **kwargs):
    np.random.seed(random_seed)
    cp.random.seed(random_seed)

    if weight_matrix is None:
        w = np.ones((n_nodes, n_nodes))
    else:
        w = weight_matrix.copy()

    if not(type(k) in (list, np.ndarray)):
        k_values = [k] * n_nodes
    else:
        k_values = k

    if delay_matrix is None:
        model = KuramotoFast(n_oscillators=n_oscillators, sampling_rate=sr,
                             k_list=k_values, weight_matrix=w, use_tqdm=use_tqdm,
                             node_frequencies=node_frequencies,
                             scale=frequency_spread, noise_scale=noise_sigma, custom_omegas=omegas, use_cuda=True)
    else:
        model = KuramotoFastDelayed(n_oscillators=n_oscillators, sampling_rate=sr,
                             k_list=k_values, weight_matrix=w, use_tqdm=use_tqdm,
                             node_frequencies=node_frequencies, delay_matrix=delay_matrix,
                             scale=frequency_spread, noise_scale=noise_sigma, custom_omegas=omegas, use_cuda=True)

    return model.simulate(time=time, random_seed=random_seed, aggregate=aggregate)
