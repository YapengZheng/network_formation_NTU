import functions as f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def data_preprocess(data, y_name, X_name):
    # data is a pandas dataframe, we process it.
    N = data.shape[0]
    # N = n(n-1)/2
    n = int((1 + np.sqrt(1 + 8*N))/2)
    y = np.zeros((n,n))
    y[np.triu_indices(n, 1)] = data[y_name].values
    y = y + y.T
    y.astype(int)
    # process X
    data = data.drop(columns = y_name)
    p = len(X_name)
    X = np.zeros((n*(n-1), p))
    for i in range(p):
        X_mat = np.zeros((n,n))
        X_mat[np.triu_indices(n, 1)] = data[X_name[i]].values
        X_mat = X_mat + X_mat.T
        X[:,i] = X_mat[np.where(np.eye(n) == 0)].reshape(n*(n-1))
    return y, X, X_name
    
def estimate(y, X):
    n = y.shape[0]
    p = X.shape[1]
    beta_init = np.zeros(p)
    beta_mm = f.est_beta_MM(X, y, X, beta_init, maxit = 1e4)
    beta_os = f.est_MLE_HJ(X,y,beta_mm[0]) 
    est = f.est_MLE_bagging(X, y, beta_mm[0], B=2*n)
    beta_bg = est[0]
    sd_bg = est[1]
    pval_bg = 2*(1 - f.norm.cdf(np.abs(beta_bg/sd_bg)))
    alpha_bg = f.alpha_MM(beta_bg, X, y, tol=1e-10, maxit= 1e6)
    APE = f.est_APE(X,y)
    sd_APE = f.est_APE_variance(X, y, alpha_bg, beta_bg)
    pval_APE = 2*(1 - f.norm.cdf(np.abs(APE/sd_APE)))
    return beta_bg, sd_bg, pval_bg, APE, sd_APE, pval_APE, alpha_bg

def NTU_est(data, y_name, X_name, plot_alpha=True, random_seed = 3):
    y, X, X_name = data_preprocess(data, y_name, X_name)
    np.random.seed(random_seed)
    beta_bg, sd_bg, pval_bg, APE, sd_APE, pval_APE, alpha_bg = estimate(y, X)
    result = pd.DataFrame({'beta': beta_bg, 'sd': sd_bg, 'pval': pval_bg}, index = X_name)
    result['95p lower'] = beta_bg - 1.96*sd_bg
    result['95p upper'] = beta_bg + 1.96*sd_bg
    result_APE = pd.DataFrame({'APE': APE, 'sd': sd_APE, 'pval': pval_APE}, index = X_name)
    result_APE['95p lower'] = APE - 1.96*sd_APE
    result_APE['95p upper'] = APE + 1.96*sd_APE
    # take 4 digits
    result = result.round(4)
    result_APE = result_APE.round(4)
    print('*************************** Coef. ************************')
    print(result)
    print('*************************** APE **************************')
    print(result_APE)
    if plot_alpha:
        print('*************************** FEs histogram ************************')
        plt.figure(figsize=(4, 3))
        sns.histplot(alpha_bg, kde=False, edgecolor='black', color= 'blue')
        plt.xlabel('Heterogeneity, $\\hat{\\alpha}$')
        plt.ylabel('Count')
        plt.show()
    return result, result_APE, alpha_bg



