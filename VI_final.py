#Authored by Evgeny Genov

import numpy as np
from math import pi, sqrt
import matplotlib.pyplot as plt
from scipy.special import gamma

# A factorized variational approximation to the posterior as expressed in (10.24)
def q(a_N, b_N, mu_N, lambda_N, mu, tau):
	# Gaussian distribution
	q_mu = (lambda_N/(2*pi))**(0.5)
	q_mu *= np.exp(-0.5*np.dot(lambda_N,((mu-mu_N)**2).T))
	# Gamma distribution
	q_tau = (1/gamma(a_N))*(b_N**a_N * tau**(a_N-1) * np.exp(-b_N*tau))
	q = q_mu*q_tau
	return(q)

# Compute q for true parameters
def exactPost(mu, mu_T, lambda_T, tau, a_T, b_T):
	post = ((b_T**a_T)*sqrt(lambda_T))/(gamma(a_T)*sqrt(2*pi))
	post = post*(tau**(a_T-0.5))*np.exp(-b_T*tau)
	post = post*np.exp(-0.5*(lambda_T*np.dot(tau,((mu-mu_T)**2).T)))
	return(post)

# expectations to implement parameter updates
def expectations(mu_N, lambda_N, a_N, b_N):
	expected_mu = mu_N
	expected_mu2 = (1/lambda_N) + mu_N**2
	expected_tau = a_N/b_N
	return(expected_mu, expected_mu2, expected_tau)

# updating scheme for parameters of q_mu and q_tau
def update_parameters(x, lambda_0, mu_0, a_0, b_0, iterations, mu, tau):  # x - data points
	N = len(x)
	x_mean = (1/N)*sum(x)
	# mu_N and a_N are constants
	mu_N = (lambda_0*mu_0 + N*x_mean)/(lambda_0 + N)
	a_N = a_0 + (N+1)/2
    # declare parameters of an exact posterior
	a_T, b_T, mu_T, lambda_T = trueParameters(x,a_0,b_0,mu_0,lambda_0)
	# Initalized values before they get updated
	# INITALIZE b_N AND lambda_N HERE
	b_N = 0.1
	lambda_N = 0.1

	# lambda_N and b_N are updated iteratively
	for i in range(iterations):
		expected_mu, expected_mu2, expected_tau = expectations(mu_N, lambda_N, a_N, b_N)
		lambda_N = (lambda_0+N)*expected_tau
		b_N = b_0-expected_mu*(lambda_0*mu_0+sum(x))
		b_N = b_N+0.5*(expected_mu2*(lambda_0+N)+lambda_0*mu_0**2+sum(x**2))

        # calculate factored posterior
		q_posterior = q(a_N,b_N,mu_N,lambda_N,mu[:,None],tau[:,None])

        # calculate exact posterior
		exact = exactPost(mu[:,None], mu_T, lambda_T, tau[:,None], a_T, b_T)

		# Plot the posterior approximation and the exact one in the same canvas
		plotexact(mu, tau, exact)
		colour = 'b'
		if i == iterations-1:
			colour = 'r'

		plotPost(mu, tau, q_posterior, colour, i)
# generate random dataset X
# m - mean, p - precision
def generate_data(m, p, N):
    return np.random.normal(m, np.sqrt(p ** (-1)), N)

# build the canvas and plot the posterior
def plotPost(mu, tau, q_posterior, colour, i):
	muGrid, tauGrid = np.meshgrid(mu, tau)
	plt.contour(muGrid, tauGrid, q_posterior, colors = colour)
	plt.title('Posterior approximation after '+str(i)+' iterations')
	plt.axis([-1.5,1.5,0,3])
	plt.xlabel('$\mu$')
	plt.ylabel('tau')
	plt.savefig('./plots/morepoints_iteration_'+str(i)+'.png')
	plt.clf()


# parameters of of q_mu and q_tau for an exact posterior found analytically
def trueParameters(x, a_0, b_0, mu_0, lambda_0):
	N = len(x)
	x_mean = (1/N)*sum(x)
	mu_T = (lambda_0*mu_0 + N*x_mean)/(lambda_0 + N)
	lambdaT = lambda_0 + N
	aT = a_0 + N/2
	b_T = b_0 + 0.5*sum((x-x_mean)**2)
	b_T = b_T + (lambda_0*N*(x_mean-mu_0)**2)/(2*(lambda_0+N))
	return(aT, b_T, mu_T, lambdaT)


# Plot the exact posterior
def plotexact(mu, tau, exact):
	muGrid, tauGrid = np.meshgrid(mu, tau)
	plt.contour(muGrid, tauGrid, exact, colors = 'g')

# generate random data from Gaussian
x = generate_data(0, 1, 100)

# initialize parameters
a_0 = 0
b_0 = 0
mu_0 = 0
lambda_0 = 0


# generate mus and taus for conjugate Gaussian, Gamma distributions
mu = np.linspace(-2,2,100)
tau = np.linspace(0,4,100)

# update parameters and plot factored posterior over iterations paired with the exact posterior
update_parameters(x, lambda_0, mu_0, a_0, b_0, iterations, mu, tau)
