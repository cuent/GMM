import numpy as np
from scipy.stats import multivariate_normal

class GaussianDistribution():
    def __init__(self, mean, cov):
        self.mean= mean
        self.cov = cov

    def pdf(self, X):
        #TODO: validate, weird error with singular matrices
        l = len(X.shape)
        if l > 1:
            n, d = X.shape
        else:
            n = 1
            d, = X.shape
            X = X.reshape(n, d)


        r = np.zeros(n)
        for i in range(n):
            det = np.linalg.det(self.cov)
            diff = X[i] - self.mean
            t_ = -(diff).T.dot(np.linalg.inv(self.cov)).dot(diff) / 2
            t = 1 / np.sqrt(det * (2 * np.pi) ** d)
            r[i] = t * np.exp(t_)

        return r[0] if len(r) == 1 else r

    def pdf1(self, X):
        return multivariate_normal.pdf(X, mean=self.mean, cov=self.cov)


class GMM():
    def __init__(self, X, K=2):
        self.N, self.D = X.shape
        self.X = X
        self.K = K

        # parameters
        self.mean = np.random.rand(K, self.D)
        self.cov = np.array([np.identity(self.D)] * K)
        self.coefficients = np.array([1/K] * K)

        # responsabilities
        self.gamma = np.zeros((self.N, K))

    def expectation(self):
        probs = np.zeros((self.N, self.K))

        for k in range(self.K):
            dist = GaussianDistribution(self.mean[k], self.cov[k])
            probs[:, k] = self.coefficients[k] * dist.pdf1(self.X)

        evidence = probs.sum(axis=1).reshape(self.N, -1)
        self.gamma = probs / evidence

    def maximization(self):
        norm = self.gamma.sum(axis=0)

        # Calculate new mean
        self.mean = np.dot(self.gamma.T, self.X) / norm

        # Calculate new covariances
        # TODO: use matrices
        for k in range(self.K):
            acc = np.zeros((self.D, self.D))
            for n in range(self.N):
                diff = np.mat(self.X[n] - self.mean[k])
                acc += self.gamma[n, k] * (diff).T * (diff)
            self.cov[k] = acc / norm[k]

        # Calculate new coefficients
        self.coefficients = norm / norm.sum()

    def fit(self, it=5):
        for i in range(it):
            print("it", i)
            self.expectation()
            self.maximization()
