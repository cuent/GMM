from sklearn.datasets.samples_generator import make_blobs

def generateData(size, seed):
    X, y = make_blobs(n_samples=size[0], n_features=size[1], centers=2,
                           cluster_std=0.60, random_state=seed)
    return scale_data(X)
    # return X

def scale_data(X):
    D = X.shape[1]
    for i in range(D):
        max = X[:, i].max()
        min = X[:, i].min()
        X[:, i] = (X[:, i] - min) / (max - min)
    return X