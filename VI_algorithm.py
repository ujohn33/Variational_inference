#Authored by Evgeny Genov

import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.special import gamma


# A factorized variational approximation to the posterior as expressed in (10.24)
def q(mu, mu_N, lambda_N, tau, a_N, b_N):
    # Gaussian distribution
    q_mu = (lambda_N / (2 * pi))**0.5
    q_mu *= np.exp(-0.5 * np.dot(lambda_N, ((mu - mu_N)**2).T))
    q_tau = gamma(a_N)**(-1) * b_N**a_N
    q_tau *= tau**(a_N-1) * np.exp(-b_N * tau)
    q = q_mu * q_tau
    return q

# expectations to implement parameter updates
def expectations(mu_N, lambda_N, a_N, b_N):
    expected_mu = mu_N
    expected_mu2 = lambda_N ** (-1) * mu_N ** 2
    expected_lambda = a_N / b_N
    return expected_mu, expected_mu2, expected_lambda

# updating scheme for parameters of q_mu and q_tau
def update_parameters(x, lambda_0, mu_0, a_0, b_0, iterations):  # x - data points
    N = len(x)
    # mu_N and a_N are constants
    mu_N = (lambda_0 * mu_0 + N * np.mean(x)) / (lambda_0 + N)
    a_N = a_0 + (N + 1)/2
    lambda_N = 1
    b_N = 1
    # lambda_N and b_N are updated iteratively
    for i in range(iterations):
        expected_mu, expected_mu2, expected_lambda = expectations(mu_N, lambda_N, a_N, b_N)
        b_N = b_0 + lambda_0*(expected_mu2 + mu_0 - 2*expected_mu*mu_0) + 0.5*np.sum(x**2 + expected_mu2 - 2*expected_mu*x)
        lambda_N = (lambda_0 + N) * a_N/b_N
    return mu_N, lambda_N, a_N, b_N
# generate random dataset X
# m - mean, p - precision
def generate_data(m, p, N):
    return np.random.normal(m, np.sqrt(p ** (-1)), N)

# plot the contours for a distribution
def plotPost(mus, taus, approx, colour, i):
	muGrid, tauGrid = np.meshgrid(mus, taus)
	plt.contour(muGrid, tauGrid, approx, colors = colour)
	plt.title('Posterior approximation after '+str(i)+' iterations')
	plt.axis([-1.,1.,0,2])
	plt.xlabel('$\mu$')
	plt.ylabel('tau')
	plt.show()

# true posterior parameters of of q_mu and q_tau
def trueParameters(x, a_0, b_0, mu_0, lambda0):
    N = len(x)
	x_mean = (1/N)*sum(x)
	mu_T = (lambda_0*mu_0 + N*x_mean)/(lambda_0 + N)
	lambda_T = lambda_0 + N
	a_T = a_0 + N/2
	b_T = b_0 + 0.5*sum((D-x_mean)**2)
	b_T = b_T + (lambda0*N*(x_mean-mu_0)**2)/(2*(lambda_0+N))
	return(a_T, b_T, mu_T, lambda_T)

# Compute q for true parameters
def truePost(aT, bT, muT, lambdaT, mus, taus):
	post = ((bT**aT)*sqrt(lambdaT))/(gamma(aT)*sqrt(2*pi))
	post = post*(tau**(aT-0.5))*np.exp(-bT*tau)
	post = post*np.exp(-0.5*(lambdaT*np.dot(tau,((mu-muT)**2).T)))
	return(post)


# Plot the true posterior
def plotTrue(mus, taus, true):
	muGrid, tauGrid = np.meshgrid(mu, tau)
	plt.contour(muGrid, tauGrid, true)

# generate random data from Gaussian
x = generate_data(0, 1, 100)
# initialize parameters
mu_0, lambda_0, a_0, b_0 = 0,1,0,1
# number of iterations to converge parameters
iterations = 10
# update parameters
mu_N, lambda_N, a_N, b_N = update_parameters(x, lambda_0, mu_0, a_0, b_0, iterations)
# generate mus and taus for conjugate Gaussian, Gamma distributions
mus = np.linspace(-2.0, 2.0, 100)
taus = np.linspace(0, 4.0, 100)
# calculate factored posterior
q_posterior = q(mus[:,None], mu_N, lambda_N, taus[:,None], a_N, b_N)
