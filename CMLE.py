# -*- coding: utf-8 -*-
"""
Created on Mon May 20 08:52:37 2019

@author: Martin
"""

"""
Conditional Maximum Likelihood Estimation (C.M.L.E.) in the binary logistic 
model with unobserved heterogeneity.

References
----------
 Woolridge, "Econometric Analysis of Cross-Section and Panel Data" (Chap 15.8.3).

"""

import numpy as np
from scipy import stats
from sympy.utilities.iterables import multiset_permutations
from scipy.special import comb
from scipy.optimize import fmin_l_bfgs_b

class BinLogitCMLE():
    """
    Implement CMLE for the conditionnal binary logit with unobserved heterogeneity and
    T periods.
    
    Reference : Woolridge, "Econometric Analysis of Cross-Section and Panel Data" (Chap 15.8.3).
    
    Parameters
    ------------
    `A' : numpy.ndarray that constains covariates at each time for each individual (n x T x K)
    `b' :  numpy numpy.ndarray that contains oucome 0/1 for each individual at each time period (n x T)
    """
    
    def __init__(self, A, b):
        self.A = A
        # make sure b is of correct format
        # make sure it contains only ones
        # make sure A is numeric
        self.b = b
        if len(A.shape) != 3:
            print('The design matrix A must have shape (N x T x K).')
        else:
            self.n, self.T, self.K = A.shape
        
        self.R = self.compute_perm()   
        
    def compute_perm(self):
        """
        Returns
        `R' : numpy.ndarray that contains all vectors of 1/0 of size T such 
        that their coordinates sum equals k, for k in [0,T]
        """
        R = list()
        for k in range(self.T + 1):
            array = np.ndarray(shape=(int(comb(self.T, k)), self.T))
            select = np.zeros(self.T)
            select[:k] = np.ones(k)
            for idx, p in enumerate(multiset_permutations(select)):
                array[idx] = p
            R.append(array)
        return R

    def objective_i(self, i, A, b, beta):
        """
        Returns the i-th contribution to the objective function (CMLE is a finite-sum).
        
        Parameters
        ------------
        `A' : design matrix
        `b' : outcome variable
        """
        R = self.R
        Xprime_beta = A[i].dot(beta)
        n_i = int(np.sum(b[i]))
        res = 0
        if((n_i!=0) & (n_i!=self.T)):
            omega_i = (1. / (np.sum(np.exp(R[n_i].dot(Xprime_beta)), axis=0)))
            res = np.log(np.exp(np.dot(b[i], Xprime_beta)) * omega_i)
        return res
    
    def objective(self, beta, A, b):
        objective = 0
        for i in range(self.n):
            objective += self.objective_i(i, A, b, beta)
        return objective / self.n
    
    def loss(self, beta, A, b):
        objective = 0
        for i in range(self.n):
            objective += self.objective_i(i, A, b, beta)
        return - objective / self.n
    
    def loss_grad(self, beta, A, b):
        """
        Computes the global gradient.
        """
        g = np.zeros_like(beta)
        for i in range(self.n):
            g += self.comp_grad_i(i, beta)
        return - g / self.n
    
    def comp_grad_i(self, i, beta): 
        """
        Returns the pointwise gradient evaluated on datum i.
        """
        A = self.A
        b = self.b
        R = self.R
        n_i = int(np.sum(b[i]))
        Xprime_beta = A[i].dot(beta)
        if((n_i!=0) & (n_i!=self.T)):
            omega_i = (1. / (np.sum(np.exp(R[n_i].dot(Xprime_beta)))))
            g = (np.dot(b[i], A[i]) - np.sum(np.dot(R[n_i], A[i]) * 
                        np.tile(np.exp(R[n_i].dot(Xprime_beta)), 
                                (self.K, 1)).T, axis=0) * omega_i)
        else: # case where no matrix but a unique vector in R[n_i]
            omega_i = (1. / (np.sum(np.exp(R[n_i][0].dot(Xprime_beta)))))
            #print(omega_i)
            g = (np.dot(b[i], A[i]) - np.sum(np.dot(R[n_i][0], A[i]) * 
                        np.tile(np.exp(R[n_i][0].dot(Xprime_beta)), 
                                (self.K, 1)).T[0], axis=0) * omega_i)
        return g
       
    def comp_hessian_i(self, i, beta):
        """
        Returns the pointwise Hessian matrix evaluated on datum i.
        """
        A = self.A
        b = self.b
        R = self.R
        n_i = int(np.sum(b[i]))
        Xprime_beta = A[i].dot(beta)
        if((n_i!=0) & (n_i!=self.T)):
            omega_i = (1. / (np.sum(np.exp(R[n_i].dot(Xprime_beta)))))
            b_i = np.sum(np.dot(R[n_i], A[i]) * 
                        np.tile(np.exp(R[n_i].dot(Xprime_beta)), (self.K, 1)).T, axis=0)
            hess = (np.outer(b_i, b_i) * omega_i ** 2 - 
                    np.sum([np.outer(elem, elem) for idx, elem in enumerate(np.dot(R[n_i], A[i]))]) 
                    * omega_i)
        else:
            omega_i = (1. / (np.sum(np.exp(R[n_i].dot(Xprime_beta)))))
            b_i = np.sum(np.dot(R[n_i][0], A[i]) * 
                        np.tile(np.exp(R[n_i][0].dot(Xprime_beta)), (self.K, 1)).T[0], axis=0)
            #print(b_i)
            hess = (np.outer(b_i, b_i) * omega_i ** 2 - 
                    np.sum([np.outer(elem, elem) for idx, elem in enumerate(np.dot(R[n_i][0], A[i]))]) 
                    * omega_i)
        return hess

    def comp_gradient(self, beta, A, b):
        """
        Computes the global gradient.
        """
        g = np.zeros_like(beta)
        for i in range(self.n):
            g += self.comp_grad_i(i, beta)
        return g / self.n

    def comp_hessian(self, beta): # warning : one-dimensional case
        """Computes the global Hessian.
        """
        hess = np.ndarray(shape=(self.K,self.K), buffer=np.zeros(self.K**2))
        for i in range(self.n):
            hess += self.comp_hessian_i(i, beta)
        return hess / self.n

    def fit(self, beta_init, n_iter, step=0.01, epsilon=1e-10, hessian=False, verbose=True, BFGS=True):
        beta = beta_init.copy()
        beta_list = [beta_init]
        if BFGS:
            beta, f_min, _ = fmin_l_bfgs_b(self.loss, beta_init, self.loss_grad, args=(self.A, self.b), pgtol=1e-6, factr=1e-30)
        elif not hessian:
            stop = True
            for t in range(n_iter):
                if stop:
                    beta = beta - step * self.comp_gradient(beta)
                    if verbose:
                        print("Iteration %s completed" % t)
                    beta_list.append(beta)
                    if self.objective(self.A, self.b, beta) -self.objective(self.A, self.b, beta_list[t]) < epsilon:
                        stop = False
        else:
            stop = True
            for t in range(n_iter):
                if stop:
                    beta = beta - np.linalg.inv(self.comp_hessian(beta)).dot(self.comp_gradient(beta))
                    if verbose:
                        print("Iteration %s completed" % t)
                    beta_list.append(beta)
                    if self.objective(self.A, self.b, beta) -self.objective(self.A, self.b, beta_list[t]) < epsilon:
                        stop = False
        return beta, beta_list
    
    def AsympVariance(self, beta, score=True):
        avar = np.ndarray(shape=(self.K,self.K), buffer=np.zeros(self.K**2))
        if score:
            for i in range(self.n):
                grad = self.comp_grad_i(i, beta)
                avar += np.outer(grad, grad)
        else:
            for i in range(self.n):
                avar += - self.comp_hessian_i(i, beta) # Fix the minus -> get negative variance
        return np.linalg.inv(avar)
    
    def lr_nulltest(self, bet_hat, lvl=0.05):
         """
         Performs the LR test of the global null.
         ----------
         Outputs : LR statistics, p-value
         """
         lu = self.objective(self.A, self.b, bet_hat)
         lr = self.objective(self.A, self.b, np.zeros(self.K))
         ratio = 2 *self.n * (lu - lr)
         pval = stats.chi2.sf(ratio, self.K)
         if pval<lvl:
             print("H0 : \beta_j =0 \forall j is rejected at level : ", lvl)
         return ratio, pval
