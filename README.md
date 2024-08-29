# GMM


A Gaussian Mixture Model (GMM) is a probabilistic model that assumes the data is generated from a mixture of several Gaussian distributions, each with its own mean and covariance. It is commonly used for clustering, density estimation, and as a generative model.


In a mixture model, the data is assumed to come from a mixture of different distributions. For a GMM, these distributions are Gaussian (normal) distributions.
The probability density function for a GMM is given by:
P(x) = sum_{k=1}^{K} pi_k * N(x | mu_k, Sigma_k)
Where:
x is a data point.
K is the number of Gaussian components.
pi_k is the mixing coefficient (weight) for the k-th Gaussian component, where sum_{k=1}^{K} pi_k = 1 and pi_k >= 0.
N(x | mu_k, Sigma_k) is the probability density function of the k-th Gaussian component with mean mu_k and covariance Sigma_k.

A Gaussian (normal) distribution for a data point x in d dimensions is given by:
N(x | mu, Sigma) = (1 / ((2*pi)^(d/2) * |Sigma|^(1/2))) * exp(-0.5 * (x - mu)^T * Sigma^(-1) * (x - mu))
Where:
mu is the mean vector (center) of the distribution.
Sigma is the covariance matrix, which defines the shape of the distribution.
|Sigma| is the determinant of the covariance matrix.
Sigma^(-1) is the inverse of the covariance matrix.


The EM algorithm is an iterative method used to estimate the parameters of the GMM (i.e., the means mu_k, covariances Sigma_k, and mixing coefficients pi_k).

The EM algorithm consists of two main steps:
1) E-step (Expectation): Compute the responsibilities (probabilities) that each data point belongs to each Gaussian component, given the current parameter estimates.
gamma_ik = P(z_i = k | x_i, theta) = (pi_k * N(x_i | mu_k, Sigma_k)) / sum_{j=1}^{K} (pi_j * N(x_i | mu_j, Sigma_j))
Where gamma_ik is the responsibility that component k takes for data point x_i.
2) M-step (Maximization): Update the parameters (mu_k, Sigma_k, pi_k) to maximize the expected log-likelihood of the data given the responsibilities computed in the E-step.

Update the mixing coefficients:
pi_k = (1 / n) * sum_{i=1}^{n} gamma_ik

Update the means:
mu_k = (sum_{i=1}^{n} gamma_ik * x_i) / sum_{i=1}^{n} gamma_ik

Update the covariances:
Sigma_k = (sum_{i=1}^{n} gamma_ik * (x_i - mu_k) * (x_i - mu_k)^T) / sum_{i=1}^{n} gamma_ik
This process is repeated until the log-likelihood of the data converges (i.e., stops changing significantly).
