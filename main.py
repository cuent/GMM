from gmm1d import GMM1D
from gmm import GMM
from utils import generateData

if "__main__" == __name__:

    # 1D data
    X1 = generateData((400, 1), seed=42)
    gmm = GMM1D(X1, K=2)
    gmm.fit(verbose=True, plot=False)
    # gmm.plot()

    #ND data
    X2 = generateData((400, 2), seed=42)
    g = GMM(X2)
    g.fit()