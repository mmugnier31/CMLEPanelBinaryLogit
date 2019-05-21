# -*- coding: utf-8 -*-
"""
Created on Mon May 20 08:52:37 2019

@author: Martin
"""

"""
Examples code for Conditional Maximum Likelihood Estimation (C.M.L.E.) in the binary logistic 
model with unobserved heterogeneity.
"""

# Uncomment to set the right path to CMLE.py
#import os
#os.chdir("setpath") 

import numpy as np
from CMLE import BinLogitCMLE

# Logistic model : T = 3, standard normal fixed effect and two random binary covariates
def err_cdf(x):
    return 1 / (1 + np.exp(-x))

def simulate_twobinvar(n, T, beta_0=np.array([1., 1.])):
    K = 2
    W = np.ndarray(shape=(n, T, K)) # explanatory variables
    for raw in range(n):
        for period in range(T):
            W[raw, period] = np.random.binomial(1, 0.5, size=2)
            #W[raw, period] = np.random.normal(1, 0.5, size=2)
    Y = np.ndarray(shape=(n,3)) # outcome variable
    for i in range(n):
        Y[i] = np.random.binomial(1, err_cdf(np.dot(W[i],beta_0) + np.random.normal(0,1)), size = T)

    return W, Y

beta_0 = np.array([1., 1])
W, Y = simulate_twobinvar(10000, 3)

model = BinLogitCMLE(A= W, b = Y)

# Run CMLE with constant step
beta_min, beta_list = model.fit(beta_init=np.zeros(2), n_iter=100, step=0.1, epsilon = 1e-10, hessian=False)
plt.plot([model.objective(elem) for elem in beta_list])
plt.show()
print("CMLE estimator : %s" % beta_min) # convergence seems OK

# Run CMLE with hessian step
beta_min, beta_list = model.fit(beta_init=np.zeros(2), n_iter=100, step=0.1, epsilon = 1e-10, hessian=True)
plt.plot([model.objective(elem) for elem in beta_list])
plt.show()
print("CMLE estimator : %s" % beta_min) # convergence seems OK
