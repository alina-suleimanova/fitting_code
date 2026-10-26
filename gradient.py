import numpy as np
import cupy as cp
import scipy as sp
from cross_analytics.crosspy.core import synchrony, criticality
from kuramoto_functions import simulate_single_run
from optimizers import AdamOptimizer, RMSpropOptimizer
import logging

logging.basicConfig(filename='./logs.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

class Gradient:
    def __init__(self, connectome, f_nodes, parameters, target_plv, target_dfa, target_occ, mask=None, regularization=False, **kwargs):
        self.connectome = connectome
        self.f_nodes = f_nodes
        self.parameters = parameters
        self.target_plv = target_plv
        self.target_dfa = target_dfa
        self.target_occ = target_occ
        self.mask = mask
        self.regularization = regularization
        self.weight_matrix = np.ones_like(self.connectome)
        self.kn = np.ones(self.connectome.shape[0])
        self.loss_dfa = None
        self.loss_plv = None
        self.loss_occ = None 

        
    def calculate_gradients(self):
        if self.regularization:
            self.loss_occ += 1e-3 * np.sign(self.loss_occ) * np.abs(self.weight_matrix - 1)
            self.loss_dfa += 1e-3 * np.sign(self.loss_dfa) * np.abs(self.kn - 1) 
            self.loss_plv += 1e-3 * np.sign(self.loss_plv) * np.abs(self.weight_matrix - 1)
        np.fill_diagonal(self.loss_plv, 0)
        
        grad_k = self.sigmoid_gradient(self.kn, k=1) * np.mean(self.loss_plv, axis=-1)
        grad_pac = self.sigmoid_gradient(self.kn, k=1) * self.loss_pac
        gradient = self.sigmoid_gradient(self.weight_matrix, k=1) * self.loss_plv 
        
        x0 = 1.0
        gamma = 0.5  # Width of Lorentzian
    
        dfa_gradient = self.lorentzian_gradient(self.kn, x0, gamma) * self.loss_dfa
        dfa_grad_weights = self.lorentzian_gradient(self.weight_matrix, x0, gamma) * self.loss_occ

        grad_k_scaled = grad_k / (np.linalg.norm(grad_k) + 1e-6) 
        dfa_gradient_scaled = dfa_gradient / (np.linalg.norm(dfa_gradient) + 1e-6)

        gradient_scaled = gradient /  (np.linalg.norm(gradient) + 1e-6) 
        dfa_grad_weights_scaled = dfa_grad_weights / (np.linalg.norm(dfa_grad_weights) + 1e-6)

        d_weights = gradient_scaled + np.sign(gradient_scaled) * np.abs(dfa_grad_weights_scaled) 
        d = grad_k_scaled + np.sign(grad_k_scaled) * np.abs(dfa_gradient_scaled) 

        cutoff = np.percentile(np.abs(self.loss_plv), 90)
        cutoff_dfa = np.percentile(np.abs(self.loss_dfa), 75)
        
        d[np.abs(self.loss_dfa) < cutoff_dfa] = 0.0

        if self.mask is None:
            d_weights[np.abs(self.loss_plv) < cutoff] = 0.0
        else:
            d_weights *= np.abs(self.mask)
        
        return d, d_weights

    def tune_parameters(self, lr=0.1, n_iters=100, omegas=None, random_seed=42, frequency=10, **kwargs):
        lr_values = [lr] * n_iters if not isinstance(lr, (list, np.ndarray)) else lr
        upper_indices = np.triu_indices_from(self.connectome, 1)

        sr = self.parameters['sr']
        time = self.parameters['time']

        minlength = sr * 20 / frequency
        minlength = min([minlength,sr])
        maxlength = (time - 50) * sr * 0.25
        window_sizes = (2 ** np.arange(np.log2(minlength), np.log2(maxlength), 0.2)).astype(int)

        plv_corr, dfa_corr = [], []
        plv_hist, dfa_hist, ope_hist, weights_hist, kn_hist = [], [], [], [], []

        optimizer_k = AdamOptimizer()
        optimizer_weights = AdamOptimizer()

        for i, coeff in enumerate(lr_values):
            K_ind = np.array(self.parameters['K'] * self.kn)
            simulated = simulate_single_run(
                K_ind, self.f_nodes, n_oscillators=self.parameters['num_oscillators'], 
                weight_matrix=self.connectome *  self.weight_matrix, time=time, 
                noise_sigma=self.parameters['noise_sigma'], frequency_spread=self.parameters['frequency_spread'], 
                n_nodes=self.connectome.shape[0], sr=sr, use_tqdm=False, omegas=omegas, random_seed=random_seed, 
                delay_matrix=self.delay_matrix
            )

            data_complex = simulated[:, sr*50:]
            model_plv = np.abs(synchrony.cplv(data_complex))
            model_dfa = criticality.dfa(np.abs(data_complex), window_sizes)[2]
            model_cc = synchrony.cc(data_complex)

            self.loss_dfa = model_dfa - self.target_dfa
            self.loss_plv = model_plv - self.target_plv
            self.loss_occ = model_cc - self.target_occ

            np.fill_diagonal(self.loss_plv, 0)
            np.fill_diagonal(self.loss_occ, 0)

            k_grad, weights_grad = self.calculate_gradients()

            logging.info(f"FC correlation {i} {sp.stats.pearsonr(model_plv[upper_indices], self.target_plv[upper_indices])[0]}")
            logging.info(f"DFA correlation {i} {sp.stats.pearsonr(model_dfa, self.target_dfa)[0]}")

            plv_corr.append(sp.stats.pearsonr(model_plv[upper_indices], self.target_plv[upper_indices])[0])
            dfa_corr.append(sp.stats.pearsonr(model_dfa, self.target_dfa)[0])
            plv_hist.append(model_plv)
            weights_hist.append(self.weight_matrix.copy())
            dfa_hist.append(model_dfa)
            kn_hist.append(self.kn.copy())

            lr = 0.01 + 0.5 * (coeff - 0.01) * (1 + np.cos(np.pi * i / len(lr_values)))

            self.kn = optimizer_k.update(coeff, self.kn, k_grad)
            self.kn[self.kn < 0] = 0

            self.weight_matrix = optimizer_weights.update(coeff, self.weight_matrix, weights_grad)
            self.weight_matrix[self.weight_matrix < 0] = 0

        return plv_corr, plv_hist, weights_hist, dfa_corr, dfa_hist, ope_hist, kn_hist

    # Lorentzian function
    def lorentzian(self, x, x0, gamma, a=1):
        x0 = np.mean(x)
        return a / (1 + ((x - x0) / gamma) ** 2)
    
    # Lorentzian gradient
    def lorentzian_gradient(self, x, x0, gamma, a=1):
        x0 = np.mean(x)
        return -2 * a * (x - x0) / (gamma**2 * (1 + ((x - x0) / gamma) ** 2) ** 2)
    
    # Sigmoid function
    def sigmoid(self, x, k=1):
        x0 = np.mean(x)
        return 1 / (1 + np.exp(-k * (x - x0)))  # Centered at mean(x)
    
    # Sigmoid gradient
    def sigmoid_gradient(self, x, k=1):
        f_sigma = self.sigmoid(x, k)
        return f_sigma * (1 - f_sigma)