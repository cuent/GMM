import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

class Gaussian():
    def __init__(self, mu, var):
        self.mu = mu
        self.var = var

    def pdf(self, x):
        v =  -(x - self.mu)**2 / (2 * self.var)
        norm = 1 / np.sqrt(2 * np.pi * self.var)
        return norm * np.exp(v)

    def pdf1(self, x):
        return stats.norm(self.mu, self.var).pdf(x)

    def __repr__(self):
        return 'Gaussian({0:4.6}, {1:4.6})'.format(self.mu, self.var)


class GMM1D:
    def __init__(self, X, K=2):
        self.X = X # dataset

        self.K = K # components
        self.N, self.D = X.shape # Number of observations X Dimension

        # Matrix of responsabilities
        self.gamma = np.zeros((self.N, self.K))

        # initialization of parameters
        self.mu = np.random.rand(self.K)
        self.var = np.random.rand(self.K)
        self.alfa = np.array([1/self.K] * self.K)

    def expectation(self):
        prob = np.zeros((self.N, self.K))

        # Calculate P(X|Z)P(Z)
        for k in range(self.K):
            g = Gaussian(self.mu[k], self.var[k])
            prob[:, k] = self.alfa[k] * g.pdf1(self.X).reshape(self.N)

        # Calculate posterior P(Z|X) and update responsabilities matrix
        marginal = prob.sum(axis=1).reshape(400,1)
        self.gamma = prob / marginal

    def maximization(self):
        total = self.gamma.sum(axis=0).reshape(-1)
        # Re-estimate mean
        for k in range(self.K):
            tot = 0
            for n in range(self.N):
                tot += self.gamma[n, k] * self.X[n]
            self.mu[k] = tot / total[k]
        # self.mu = self.gamma.T.dot(self.X).reshape(self.K,) / total

            # Re-estimate var
            diff = 0
            for n in range(self.N):
                diff_ = (self.X[n] - self.mu[k]) ** 2
                diff += self.gamma[n, k] * diff_
            self.var[k]  = diff / total[k]
        # self.var = (self.gamma * diff).sum(axis=0) / total
        # var_ = self.gamma * (self.X.dot(self.mu.reshape(self.K, -1).T) ** 2)
        # self.var = var_.sum(axis=0) / total

        # Re-estimate coefficients
        self.alfa = total / total.sum()

    def evaluation(self):
        probs = np.zeros((self.N, self.K))

        for k in range(self.K):
            g = Gaussian(self.mu[k], self.var[k])
            probs[:,k] = self.alfa[k] * g.pdf(self.X).reshape(self.N)
        probs = np.log(probs.sum(axis=1))
        return probs.sum()

    def fit(self, convergence=0.01, verbose=False, plot=False, iterations=2):
        if plot:
            sns.distplot(self.X, bins=20, kde=False, norm_hist=True)

        for i in range(iterations):
            eval = self.evaluation()
            if verbose:
                print("Iteration {}: Evaluation: {}\t {}".format(i, eval, self))

            if plot:
                c = np.random.rand(3)
                for k in range(self.K):
                    g = Gaussian(self.mu[k], self.alfa[k])
                    plt.plot(self.X, g.pdf(self.X), marker='o', linestyle='', color= c, label="it {}, c {}".format(i,k))

            self.expectation()
            self.maximization()

        if plot:
            plt.legend();
            plt.show()


    def plot(self):
        sns.distplot(self.X, bins=20, kde=False, norm_hist=True)

        for k in range(self.K):
            g = Gaussian(self.mu[k], self.alfa[k])
            plt.plot(self.X, g.pdf(self.X), "bo", label="c {}".format(k))
        plt.legend();
        plt.show()


    def __repr__(self):
        return "X:{}\tMean:{}\tVariance:{}\tCoefficients:{}\t"\
            .format(self.X.shape, self.mu, self.var, self.alfa)