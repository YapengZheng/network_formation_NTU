# network_formation_NTU
## A demo for estimating network formation models with non-transferable utilities.

This is a demo code for applying bagging estimator for network formation models with nontransferable utilities and individual fixed effects.

### files in /demo_for_application
NAT_application.ipynb: a jupyter notebook, containing the demo code for Nyakatoke network in our paper.
network_Na_panel.dta: an example dataset, Nyakatoke network data (please cite De Weerdt (2004) if you want to use this dataset).

estimation.py: contains the main function: NTU_est(data, y_name, X_name, plot_alpha=True). 
Inputs:
data: Users need to prepare a network panel data, in which each row represents a dyad (so N=n(n-1)/2 rows, n is the number of nodes in the network). The network panel data should be sorted.
y_name: the variable name for the undirected link.
X_name: the variable names for the covariates.
plot_alpha: if = True, the function will plot the histogram of estimated heterogeneities alpha_hat; if = False, will not plot.

Returns:
1. a table for bagging coefficients estimates, including point estimators, standard deviations, p-values and 95% confidence intervals.
2. a table for plug-in average partial effects, including point estimators, standard deviations, p-values and 95% confidence intervals.
3. an array for estimated fixed effects.

functions.py: contains some functions used in the estimation and inference process.

### Citation
Please cite
"Estimation and Inference in Dyadic Network Formation Models with Nontransferable Utilities" 2024, working paper, by Ming Li, Zhentao Shi and Yapeng Zheng
if you are willing to use our codes, Thank you!

