import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture


# Sum of Squares Whitin - validated
def ssw(X, labels):
    k_range = np.unique(labels)
    k_range.sort()

    means = [np.mean(X[labels == l], axis=0) for l in k_range]
    means = np.reshape(means, [k_range.__len__(), X.shape[1]])
    return np.sum([np.linalg.norm(X[labels == l] - mean, axis=0)**2 for l, mean in zip(k_range, means)])


# Sum of Squares Between - validated
def ssb(X, labels):
    k_range = np.unique(labels)
    k_range.sort()
    means = [np.mean(X[labels == l], axis=0) for l in k_range]
    means = np.reshape(means, [k_range.__len__(), X.shape[1]])
    
    return np.sum([X[labels == l].shape[0]*\
        np.linalg.norm(np.mean(X, axis=0) - mean, axis=0)**2 for l, mean in zip(k_range, means)])


# Sum of Squares Total - validated
def sst(X, labels):
    return np.sum(np.linalg.norm(X - X.mean(axis=0), axis=0)**2)
    # or
    # return ssw(X, labels) + ssb(X, labels)


# CH-index - validated
def ch_index(X, labels):
    n_k = np.unique(labels).shape[0]
    return ((X.shape[0] - n_k)*ssb(X, labels)/((n_k - 1)*ssw(X, labels)))


# Ball-Hall 
def ball_hall(X, labels):
    return ssw(X, labels)/X.shape[0]


# Hartigan
def hartigan(X, labels):
    return np.log(ssb(X, labels), ssw(X, labels))


# Xu-index
def xu_index(X, labels):
    n, d = X.shape
    n_k = np.unique(labels).shape[0]
    return d*np.log((ssw(X, labels)/(d*(n**2)))**(1/2)) + np.log(n_k)


# Ratkowsky & Lance index
def rl_index(X, labels):
    n_k = np.unique(labels).shape[0]
    d_range = np.arange(X.shape[1])
    rl = 0
    for d in d_range:
        X_d = X[:, d].reshape(-1, 1)
        rl += (ssb(X_d, labels)/sst(X_d, labels))**(1/2)
    rl = (1/d_range.shape[0])*rl
    return rl/(n_k**(1/2))


# WB-index
def wb(X, labels):
    n_k = np.unique(labels).shape[0]
    return n_k*ssw(X, labels)/ssb(X, labels)


# Davies-Bouldin index - validated
def db_index(X, labels):
    k_range = np.unique(labels)
    rk_result = []
    for k in k_range:
        X_k = X[labels == k]
        labels_k = labels[labels == k]
        rk_list = []
        for l in k_range[k_range != k]:
            X_l = X[labels == l]
            labels_l = labels[labels == l]
            rk = (np.linalg.norm(X_k - X_k.mean(axis=0), axis=1)).mean() + \
                (np.linalg.norm(X_l - X_l.mean(axis=0), axis=1)).mean()
            rk = np.divide(rk, np.linalg.norm(
                np.mean(X_k, axis=0) - np.mean(X_l, axis=0), axis=0))
            rk_list.append(rk)
        rk_result.append(np.max(rk_list))
    return np.sum(rk_result)/k_range.shape[0]


# Silhouette - validated
# Silhouette - ai
def sil_ai(i, X, labels, labelk):
    ai = 0
    xi = X[i]
    X = np.delete(X, i, axis=0) # removing xi
    labels = np.delete(labels, i, axis=0) # removing label of xi
    for xj in X[labels == labelk]:
        ai += np.sum(np.linalg.norm(xi - xj, axis=0))
    ai = ai/(X[labels == labelk].shape[0]) # ai/ Nk - 1
    return ai

# Silhouette - bi
def sil_bi(i, X, labels, labelk):
    bi = 0
    xi = X[i]
    bi_list = []
    for k in np.unique(labels)[np.unique(labels) != labelk]:
        bi = 0
        for xj in X[labels == k]:
            bi += np.sum(np.linalg.norm(xi - xj, axis=0))
        bi_list.append(bi/X[labels == k].shape[0])
    return np.min(bi_list)

# Silhouette - silhouette per data point
def silhouette(X, labels):
    sil = []
    for i in range(X.shape[0]):
        ai = sil_ai(i, X, labels, labels[i])
        bi = sil_bi(i, X, labels, labels[i])
        sil.append((bi - ai)/np.max([bi, ai]))
    return np.asarray(sil)


# Gap Statistics - validated
# Gap - Uniformly random reference generator
def random_reference(X):
    randX = np.empty(shape=X.shape)
    for feat in range(X.shape[1]):
        randX[:, feat] = np.random.uniform(
            X[:, feat].min(), X[:, feat].max(), X[:, feat].shape)
    return randX

# Gap - Expected SSW from random reference
def random_ssw(X, k, algorithm):
    randX = random_reference(X)
    
    model = randomModel(randX, k, algorithm)
   
    labels_ref = model.predict(X=randX)
    
    return ssw(randX, labels_ref)

# Gap - Model from random reference
def randomModel(randX, k, algorithm):
    if algorithm == 'KMeans':
        model = KMeans(n_clusters=k, n_init=1).fit(X=randX)
    elif algorithm == 'Agglomerative Ward':
        model = AgglomerativeClustering(
            n_clusters=k, linkage='ward').fit(X=randX)
    elif algorithm == 'GMM':
        model = GaussianMixture(
            n_components=k, n_init=1, max_iter=200).fit(X=randX)

    return model
    
# Gap - Gap mean and error
def gap(X, labels, nrefs=30, algorithm='KMeans'):
    k = np.unique(labels).shape[0]
    refDisps = np.zeros(nrefs)

    for i in range(nrefs):
        refDisps[i] = random_ssw(X, k, algorithm)

    origDisp = ssw(X, labels)

    # Calculate gap statistic
    gap_mean = np.mean(np.log(refDisps)) - np.log(origDisp)

    # Calculate gap statistic std
    gap_std = np.sqrt(1 + (1/nrefs))*np.std(np.log(refDisps))

    expec = np.mean(np.log(refDisps))
    ob = np.log(origDisp)

    return ob, expec, gap_mean, gap_std

# Gap - optimal k = smallest k such that 𝐺𝑎𝑝(𝑘)≥𝐺𝑎𝑝(𝑘+1)−𝑆𝑘+1
def gap_optimal_k(gap_means, gap_stds, k_range):
    for k in k_range[:-1]:
        if gap_means[k-1] >= (gap_means[k] - gap_stds[k]):
            return k
    return -1

    
# Bayes Information Criterion (BIC)
def bic():
    pass


# Degree of membership matrix
"""Reference:
    Fuzzy C-means
    @misc{fuzzy-c-means,
    author       = "Madson Luiz Dantas Dias",
    year         = "2019",
    title        = "fuzzy-c-means: An implementation of Fuzzy $C$-means clustering algorithm.",
    url          = "https://github.com/omadson/fuzzy-c-means",
    institution  = "Federal University of Cear\'{a}, Department of Computer Science" 
    }
    ----------
    .. [1] `Pattern Recognition with Fuzzy Objective Function Algorithms
        <https://doi.org/10.1007/978-1-4757-0450-1>`_
    .. [2] `FCM: The fuzzy c-means clustering algorithm
        <https://doi.org/10.1016/0098-3004(84)90020-7>`_
    """
def u_ij(X, labels, m):
    k_range = np.unique(labels)

    means = [np.mean(X[labels == l], axis=0) for l in k_range]
    means = np.reshape(means, [k_range.__len__(), X.shape[1]])
    
    power = float(2 / (m - 1))
    temp = cdist(X, means) ** power
    denominator_ = temp.reshape(
        (X.shape[0], 1, -1)).repeat(temp.shape[-1], axis=1)
    denominator_ = temp[:, :, np.newaxis] / denominator_

    return 1 / denominator_.sum(2)
