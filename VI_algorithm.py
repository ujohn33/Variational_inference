#Authored by Evgeny Genov

import numpy as np
from math import exp, pi
import matplotlib.pyplot as plt
from scipy.special import gamma


# A factorized variational approximation to the posterior as expressed in (10.24)
def q(mu, mu_N, lambda_N, tau, a_N, b_N)
    # Gaussian distribution
    q_mu = (lambda_N / (2 * pi))**0.5
    q_mu *= exp(-0.5 np.dot(lambda_N, ((mu - mu_n)**2).T))
    q_tau = gamma(a_N)**(-1) * b_N**a_N * tau**(a_N-1) * exp(-b_N * tau)
    q = q_mu * q_tau
    return q

# expectations to implement parameter updates
def expectations(mu_N, lambda_N, a_N, b_N)
    expected_mu = mu_N
    expected_mu2 = lambda_N ** (-1) * mu_N ** 2
    expected_lambda = a_N / b_N
    return expected_mu, expected_mu2, expected_lambda

# updating scheme for parameters of q_mu and q_tau
def update_parameters(x, lambda_0, mu_0, a_0, b_0, iterations)  # x - data points
    N = len(x)
    # mu_N and a_N are constants
    mu_N = (lambda_0 * mu_0 + N * np.mean(x)) / (lambda_0 + N)
    a_N = a_0 + (N + 1)/2
    # lambda_N and b_N are updated iteratively
    expected
    while i < iterations:
        b_N = b_0 + lambda_0(ex)
        lambda_N = (lambda_0 +)
