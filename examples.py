# -*- coding: utf-8 -*-
"""
Examples code for Conditional Maximum Likelihood Estimation (C.M.L.E.) in the binary logistic 
model with unobserved heterogeneity.
"""

# Uncomment to set the right path to CMLE.py
#import os
#os.chdir("setpath") 

from matplotlib import pyplot as plt
import numpy as np
from CMLE import BinLogitCMLE

np.random.seed(12)

def err_cdf(x):
    '''Logistic distribution cdf.'''
    return 1 / (1 + np.exp(-x))

def simulate_onebinvar(n, T, beta_0):
    '''Simulate from the panel fixed effect logistic model with T = 3,
    standard normal fixed effect and one random binary covariate.'''
    K = 1
    W = np.ndarray(shape=(n, T, K)) # explanatory variables
    for row in range(n):
        for period in range(T):
            W[row, period] = np.random.binomial(1, 0.5, size=K)
    Y = np.ndarray(shape=(n,3)) # outcome variable
    for i in range(n):
        fe = np.random.normal(0,1)
        Y[i,:] = np.array([float(np.random.binomial(1, err_cdf(np.dot(W[i,j], beta_0) + fe))) for j in range(3)])
    return W, Y

# Simulate data from the model
n = 10000
T = 3
beta_0 = np.array([1.])
W, Y = simulate_onebinvar(n, T, beta_0)
model = BinLogitCMLE(A=W, b=Y)

# Run CMLE with constant step
beta_min, beta_list = model.fit(beta_init=np.zeros(1), n_iter=100, step=0.1, epsilon = 1e-10, hessian=False, BFGS=False)
plt.plot([model.objective(elem, model.A, model.b) for elem in beta_list])
plt.show()
print("CMLE estimator (Raphson-Newton with constant step) : %s" % beta_min) # convergence is OK but slow

# Run CMLE with Hessian step
beta_min, beta_list = model.fit(beta_init=np.zeros(1), n_iter=100, step=0.1, epsilon = 1e-10, hessian=True, BFGS=False)
plt.plot([model.objective(elem, model.A, model.b) for elem in beta_list])
plt.show()
print("CMLE estimator (Raphson-Newton with hessian step) : %s" % beta_min) # convergence in one step

# Run CMLE with L-BFGS-B
beta_min, beta_list = model.fit(beta_init=np.zeros(1), n_iter=100, step=0.1, epsilon = 1e-10, hessian=False)
plt.plot([model.objective(elem, model.A, model.b) for elem in beta_list])
plt.show()
print("CMLE estimator (L-BFGS-B) : %s" % beta_min) # convergence is faster
