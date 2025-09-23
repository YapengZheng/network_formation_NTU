import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.optimize as opt
from scipy.stats import norm
import time
from tqdm import tqdm
import itertools
def sigmoid(x):
    # x: np matrix
    # This function computes the sigmoid function of x
    return(1/(1+np.exp(-x)))

def alpha_MM(beta, x, y, tol = 1e-3, maxit =1e6):
    n = y.shape[0]  # network size
    N = x.shape[0]  # number of dyads
    # generate matrix type of xij*beta
    xbeta = np.zeros((n, n))
    xbeta[np.where(np.eye(n)==0)] = (x @ beta).reshape(N)
    # MM method to find alpha(beta)
    error = 1
    t = 0
    at = np.zeros(n)
    # Iteration step
    while (error >= tol) & (t <= maxit):
        t = t + 1
        at0 = at
        prob = sigmoid(at0.reshape((n, 1)) @ np.ones((1, n)) + xbeta)
        prob = prob * prob.T
        yhat = y - prob
        np.fill_diagonal(yhat, 0)
        at = at0 + np.sum(yhat, axis = 1)/(n-1)
        # prevent overfitting, alpha should not go to infinity
        at[np.where(abs(at)>=2*np.log(n))] = at0[np.where(abs(at)>=2*np.log(n))]
        error = np.sum(abs(at - at0))
        #print(error)
    #print(at)
    return (at)

def profile_moment_beta(beta, x, y, z, maxit =1e4):
    n = y.shape[0]  # network size
    N = x.shape[0]  # number of dyads
    K = beta.shape[0] # number of covariates
    MM = np.zeros(K) # moment vector
    # generate matrix type of xij*beta
    xbeta = np.zeros((n, n))
    xbeta[np.where(np.eye(n)==0)] = (x @ beta).reshape(N)
    # MM method to find alpha(beta)
    alpha_beta = alpha_MM(beta, x, y, tol = 1e-1, maxit = 1e4)
    prob = sigmoid(alpha_beta.reshape((n, 1)) @ np.ones((1, n)) + xbeta)
    prob1 = prob * prob.T
    yhat = ((y - prob1))[np.where(np.eye(n)==0)].reshape((N,1))
    for k in range(K):
        MM[k] = np.sum(np.multiply(z[:, k], yhat[:, 0]))
    return (MM)

def var_MM(x,y,beta):
    n = y.shape[0]
    N = x.shape[0]
    k = beta.shape[0]
    alpha_1 = alpha_MM(beta, x, y, tol=1e-1, maxit=1e4)
    xbeta = np.zeros((n, n))
    xbeta[np.where(np.eye(n) == 0)] = (x @ beta).reshape(N)
    prob = sigmoid(alpha_1.reshape((n, 1)) @ np.ones((1, n)) + xbeta)
    prob[np.where(np.eye(n) == 1)] = 0
    prob1 = prob * prob.T
    # calculate the variance matrix
    v_aa = prob1 * (1 - prob1)
    np.fill_diagonal(v_aa, 0)
    v_aa[np.where(np.eye(n) == 1)] = np.sum(v_aa, axis=1)
    v_ab = np.zeros((n, k))
    for j in range(k):
        xj = np.zeros((n, n))
        xj[np.where(np.eye(n) == 0)] = x[:, j]
        mat = xj * prob1 * (1 - prob1)
        np.fill_diagonal(mat, 0)
        v_ab[:, j] = np.sum(mat, axis=1)
    v_bb = np.zeros((k, k))
    for j in range(k):
        xj = np.zeros((n, n))
        xj[np.where(np.eye(n) == 0)] = x[:, j]
        for l in range(k):
            xl = np.zeros((n, n))
            xl[np.where(np.eye(n) == 0)] = x[:, l]
            mat = xj * xl * prob1 * (1 - prob1)
            np.fill_diagonal(mat, 0)
            v_bb[j, l] = np.sum(mat) / 2
    # calculate the Jacobian matrix
    J_aa = -prob1 * (1 - prob.T)
    np.fill_diagonal(J_aa, 0)
    J_aa[np.where(np.eye(n) == 1)] = np.sum(J_aa, axis=0)
    J_ab = np.zeros((n, k))
    for j in range(k):
        xj = np.zeros((n, n))
        xj[np.where(np.eye(n) == 0)] = x[:, j]
        mat = xj * prob1 * (1 - prob) + xj * prob1 * (1 - prob.T)
        np.fill_diagonal(mat, 0)
        J_ab[:, j] = -np.sum(mat, axis=1)
    J_ba = np.zeros((k, n))
    for j in range(k):
        xj = np.zeros((n, n))
        xj[np.where(np.eye(n) == 0)] = x[:, j]
        mat = xj * prob1 * (1 - prob)
        np.fill_diagonal(mat, 0)
        J_ba[j, :] = -np.sum(mat, axis=1)
    J_bb = np.zeros((k, k))
    for j in range(k):
        xj = np.zeros((n, n))
        xj[np.where(np.eye(n) == 0)] = x[:, j]
        for l in range(k):
            xl = np.zeros((n, n))
            xl[np.where(np.eye(n) == 0)] = x[:, l]
            mat = xj * xl * prob1 * (1 - prob)
            np.fill_diagonal(mat, 0)
            J_bb[j, l] = -np.sum(mat)
    J_inv_a = np.linalg.inv(J_aa)
    J = J_bb - J_ba @ J_inv_a @ J_ab
    # calculate the variance of beta_hat
    J_inv = np.linalg.inv(J)
    Var = v_bb + J_ba @ J_inv_a @ v_aa @ J_inv_a.T @ J_ba.T - J_ba @ J_inv_a @ v_ab - (J_ba @ J_inv_a @ v_ab).T
    Omega = J_inv @ Var @ J_inv.T
    return(Var, J_inv)




def est_beta_MM(x, y, z, beta0, maxit =1e4):
    n = y.shape[0]  # network size
    result = opt.root(lambda beta: profile_moment_beta(beta, x, y, z, maxit=maxit), beta0, method='hybr')
    beta_hat = result.x
    return beta_hat

    

def est_one_step(x,y,beta_1):
    # calculate score function vector
    # calculate hessian matrix
    n = y.shape[0]  # network size
    N = x.shape[0]  # number of dyads
    k = x.shape[1]  # number of covariates
    # generate matrix type of xij*beta
    xbeta = np.zeros((n, n))
    xbeta[np.where(np.eye(n) == 0)] = (x @ beta_1).reshape(N)
    alpha_1 = alpha_MM(beta_1, x, y, tol=1e-1, maxit=1e4)
    prob = sigmoid(alpha_1.reshape((n, 1)) @ np.ones((1, n)) + xbeta)
    prob[np.where(np.eye(n) == 1)] = 0
    prob1 = prob * prob.T
    s_alpha = np.sum((1-prob)*(y - prob1)/(1-prob1), axis = 1)
    s_beta = np.zeros(k)
    for j in range(k):
        xj = np.zeros((n, n))
        xj[np.where(np.eye(n) == 0)] = x[:, j]
        s_beta[j] = np.sum(xj*(y - prob1)/(1-prob1)*(1-prob))
    hess_aa = -prob1*(1-prob)*(1-prob.T)*(1-y)/(1-prob1)**2
    I_aa = prob1*(1-prob)*(1-prob.T)/(1-prob1)
    mat = prob1*(1-prob)*(1-prob)/(1-prob1)+(y-prob1)*prob*(1-prob.T)/(1-prob1)**2
    np.fill_diagonal(mat,0)
    hess_aa[np.where(np.eye(n) == 1)] = -np.sum(mat,axis=1)
    mat = prob1*(1-prob)*(1-prob)/(1-prob1)
    np.fill_diagonal(mat,0)
    I_aa[np.where(np.eye(n) == 1)] = np.sum(mat,axis=1)
    hess_ab = np.zeros((n,k))
    I_ab = np.zeros((n,k))
    for j in range(k):
        xj = np.zeros((n, n))
        xj[np.where(np.eye(n) == 0)] = x[:, j]
        mat = xj*prob1*(1-prob)*(1-prob)/(1-prob1)+xj*prob1*(1-prob)*(1-prob.T)/(1-prob1)+xj*(y-prob1)*prob*(1-prob.T)/(1-prob1)**2
        np.fill_diagonal(mat,0)
        hess_ab[:,j] = -np.sum(mat, axis = 1)
        mat = xj*prob1*(1-prob)*(1-prob)/(1-prob1)+xj*prob1*(1-prob)*(1-prob.T)/(1-prob1)
        np.fill_diagonal(mat,0)
        I_ab[:,j] = np.sum(mat, axis = 1)
    hess_bb = np.zeros((k,k))
    I_bb = np.zeros((k,k))
    for j in range(k):
        xj = np.zeros((n, n))
        xj[np.where(np.eye(n) == 0)] = x[:, j]
        for l in range(k):
            xl = np.zeros((n, n))
            xl[np.where(np.eye(n) == 0)] = x[:, l]
            mat = xj*xl*prob1*(1-prob)*(1-prob)/(1-prob1)+xj*xl*prob1*(1-prob)*(1-prob.T)/(1-prob1)+xj*xl*(y-prob1)*prob*(1-prob.T)/(1-prob1)**2
            np.fill_diagonal(mat,0)
            hess_bb[j,l] = -np.sum(mat)
            mat = xj*xl*prob1*(1-prob)*(1-prob)/(1-prob1)+xj*xl*prob1*(1-prob)*(1-prob.T)/(1-prob1)
            np.fill_diagonal(mat,0)
            I_bb[j,l] = np.sum(mat)
    hess_inv_a = np.linalg.inv(hess_aa)
    I_inv_a = np.linalg.inv(I_aa)
    H = hess_bb-hess_ab.T@hess_inv_a@hess_ab
    I = I_bb - I_ab.T@I_inv_a@I_ab
    beta_2 = beta_1 + np.linalg.inv(I)@(s_beta-I_ab.T@I_inv_a@s_alpha)
    return [beta_2, H, I]

def est_MLE_HJ(x,y,beta_1):
    n = y.shape[0]  # network size
    n1 = int(n / 2)  # half size of network
    N1 = n1 * (n1 - 1)  # number of dyads in the half network
    n2 = n - n1  # half size of network
    N2 = n2 * (n2 - 1)  # number of dyads in the half network
    k = x.shape[1]  # number of covariates
    # separate network into two parts
    inda = np.random.choice(n, n1, replace=False)
    ya = y[inda, :][:, inda]
    indb = np.setdiff1d(np.arange(n), inda)
    yb = y[indb, :][:, indb]
    # separate covariates into two parts
    xa = np.zeros((N1, k))
    xb = np.zeros((N2, k))
    non_diag_indices = np.where(np.eye(n) == 0)
    non_diag_indices1 = np.where(np.eye(n1) == 0)
    non_diag_indices2 = np.where(np.eye(n2) == 0)
    for j in range(k):
        x0 = np.zeros((n, n))
        x0[non_diag_indices] = x[:, j]
        xa[:, j] = (x0[inda, :][:, inda])[non_diag_indices1].reshape(N1)
        xb[:, j] = (x0[indb, :][:, indb])[non_diag_indices2].reshape(N2)
    result = est_one_step(x,y,beta_1)
    result_a = est_one_step(xa,ya,beta_1)
    result_b = est_one_step(xb,yb,beta_1)
    beta_MLE = result[0]
    I_inv = np.linalg.inv(result[2])
    sd = np.sqrt(np.diag(I_inv))
    beta_MLE_a = result_a[0]
    beta_MLE_b = result_b[0]
    var_HJ = I_inv@(4*result[2]-4*(result_a[2]+result_b[2]))@I_inv
    beta_MLE_HJ = 2*beta_MLE -(beta_MLE_a+beta_MLE_b)/2
    sd_HJ = np.sqrt(np.diag(var_HJ))
    return [beta_MLE, beta_MLE_HJ, sd, sd_HJ]
def est_MLE_Jackknife(x, y, beta_1):
    n = y.shape[0]  # network size
    k = x.shape[1]  # number of covariates
    N1 = (n-1)* (n-2)  # number of dyads
    result = est_one_step(x, y, beta_1)
    beta_MLE = result[0]
    sd = np.sqrt(np.diag(np.linalg.inv(result[2])))
    beta_Jack = np.zeros((n, k))
    for i in range(n):
        ind = np.setdiff1d(np.arange(n), i)
        xi = np.zeros((N1, k))
        non_diag_indicesi = np.where(np.eye(n-1) == 0)
        for j in range(k):
            x0 = np.zeros((n, n))
            x0[np.where(np.eye(n)==0)] = x[:, j]
            xi[:, j] = (x0[ind, :][:, ind])[non_diag_indicesi].reshape(N1)
        yi = y[ind, :][:, ind]
        resulti = est_one_step(xi, yi, beta_1)
        beta_Jack[i, :] = resulti[0]
    beta_MLE = n*beta_MLE-(n-1)*np.mean(beta_Jack, axis=0)
    return [beta_MLE, sd]

def est_MLE_bagging(x, y, beta_1, B=200):
    n = y.shape[0]  # network size
    k = x.shape[1]  # number of covariates
    N1 = (n-1)* (n-2)  # number of dyads
    result = est_one_step(x, y, beta_1)
    beta_MLE = result[0]
    cov = np.linalg.inv(result[2])
    sd = np.sqrt(np.diag(cov))
    beta_MLE_bagging = np.zeros((B,k))
    for b in range(B):
        n1 = int(n / 2)  # half size of network
        N1 = n1 * (n1 - 1)  # number of dyads in the half network
        n2 = n - n1  # half size of network
        N2 = n2 * (n2 - 1)  # number of dyads in the half network
        # separate network into two parts
        inda = np.random.choice(n, n1, replace=False)
        ya = y[inda, :][:, inda]
        indb = np.setdiff1d(np.arange(n), inda)
        yb = y[indb, :][:, indb]
        # separate covariates into two parts
        xa = np.zeros((N1, k))
        xb = np.zeros((N2, k))
        non_diag_indices = np.where(np.eye(n) == 0)
        non_diag_indices1 = np.where(np.eye(n1) == 0)
        non_diag_indices2 = np.where(np.eye(n2) == 0)
        for j in range(k):
            x0 = np.zeros((n, n))
            x0[non_diag_indices] = x[:, j]
            xa[:, j] = (x0[inda, :][:, inda])[non_diag_indices1].reshape(N1)
            xb[:, j] = (x0[indb, :][:, indb])[non_diag_indices2].reshape(N2)
        # try to estimate beta_hat_a and beta_hat_b
        try:
            result_a = est_one_step(xa,ya,beta_1)
            result_b = est_one_step(xb,yb,beta_1)
            beta_MLE_bagging[b,:] = 2*beta_MLE-(result_a[0]+result_b[0])/2
        except:
            beta_MLE_bagging[b,:] = 999
    # exclude extreme values and nan values
    beta_MLE_bagging = beta_MLE_bagging[~np.any(np.isnan(beta_MLE_bagging),axis=1)]
    beta_MLE_bagging = beta_MLE_bagging[~np.any(np.abs(beta_MLE_bagging)>10,axis=1)]
    beta_MLE = np.mean(beta_MLE_bagging,axis=0)
    return(beta_MLE, sd)

