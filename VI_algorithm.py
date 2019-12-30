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
    return q_mu * q_tau
