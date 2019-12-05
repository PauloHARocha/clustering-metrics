import numpy as np
from scipy.spatial import distance
from operator import truediv
from numpy import inf
from sklearn.metrics import silhouette_score
import math


class Metrics:

    @staticmethod
    def gap_statistic(clusters, random_clusters):
        Wk = Metrics.compute_Wk(clusters)
        En_Wk = Metrics.compute_Wk(random_clusters)
        return np.log(En_Wk) - np.log(Wk)

    @staticmethod
    def compute_Wk(clusters):
        # Funcao do Gap Statistic
        wk_ = 0.0
        for r in range(len(clusters)):
            nr = len(clusters[r])
            if nr > 1:
                dr = distance.pdist(clusters[r], metric='sqeuclidean')
                dr = distance.squareform(dr)
                dr = sum(np.array(dr, dtype=np.float64).ravel())
                wk_ += (1.0 / (2.0 * nr)) * dr
        return wk_

    @staticmethod
    def silhouette(clusters, len_data):
        sil = 0.0
        for k in range(len(clusters)):
            for d_out in range(len(clusters[k])):
                ai = Metrics.silhouette_a(clusters[k][d_out], k, clusters)
                bi = Metrics.minimum_data_data(clusters[k][d_out], k, clusters)
                max_a_b = bi
                if ai > bi:
                    max_a_b = ai
                if max_a_b > 0:
                    sil += truediv(bi - ai, max_a_b)
        return truediv(sil, len_data)

    @staticmethod
    def silhouette_a(datum, cluster_in, clusters):
        sum_d = 0.0
        for d_out in range(len(clusters[cluster_in])):
            sum_d += distance.euclidean(datum, clusters[cluster_in][d_out])
        sum_d = truediv(sum_d, len(clusters[cluster_in]))
        return sum_d

    @staticmethod
    def minimum_data_data(datum, cluster_in, clusters):
        min_D = float('inf')
        for k in range(len(clusters)):
            if cluster_in != k:
                x = 0.0
                for d_out in range(len(clusters[k])):
                    x += distance.euclidean(datum, clusters[k][d_out])
                if len(clusters[k]) > 0:
                    x = truediv(x, len(clusters[k]))
                if min_D > x:
                    min_D = x
        return min_D

    @staticmethod
    def inter_cluster_statistic(centroids):
        centers = []
        # cluster = len(centroids)
        for c in centroids:
            centers.append(centroids[c])
        centers = np.array(centers, dtype=float)
        centers = distance.pdist(centers, metric='sqeuclidean')
        centers = distance.squareform(centers)
        centers = sum(np.array(centers, dtype=np.float64).ravel())
        # centers = (1.0 / cluster) * (1.0 / (cluster - 1)) * centers
        return centers

    @staticmethod
    def intra_cluster_statistic(data, centroids):
        # minimization
        clusters = {}
        for k in centroids:
            clusters[k] = []

        for xi in data:
            dist = [np.linalg.norm(xi - centroids[c]) for c in centroids]
            class_ = dist.index(min(dist))
            clusters[class_].append(xi)

        inter_cluster_sum = 0.0
        for c in centroids:
            if len(clusters[c]) > 0:
                for point in clusters[c]:
                    inter_cluster_sum += np.linalg.norm(point - centroids[c])
        return inter_cluster_sum

    @staticmethod
    def variance_based_ch(data, centroids):
        return truediv(len(data) - len(centroids), len(centroids) - 1) * truediv(Metrics.inter_cluster_statistic(centroids), Metrics.intra_cluster_statistic(data, centroids))

     #Sep(C)
    @staticmethod
    def cluster_separation(centroids):
        return 2*Metrics.inter_cluster_statistic(centroids) / (len(centroids) * (len(centroids)-1))

    #Average Between Group Sum of Squares (ABGSS) - maximization
    @staticmethod
    def abgss(data, centroids, clusters):
        abgss = 0.0
        for k in range(len(centroids)):
            euclidean = 0.0
            for d in range(len(centroids[k])):
                euclidean += (centroids[k][d] - np.mean(data[d])) ** 2.0
            abgss += len(clusters[k])*euclidean
        return truediv(abgss, len(centroids))
    
    #Edge index - minimized
    @staticmethod
    def edge_index(data, centroids, clusters, number_neighbors=4):
        ei = 0.0
        np.sort(data, axis=0)
        for d in range(len(data)-number_neighbors):
            idx = Metrics.ismember(data[d], clusters)
            for neighbor in range(1, number_neighbors):
                idx2 = Metrics.ismember(data[d+neighbor], clusters)
                if idx != idx2:
                    ei += distance.euclidean(data[d], data[d+neighbor])
        return -ei
    
    #Edge index - Cluster Connectedness
    @staticmethod
    def ismember(d, c):
        for a1 in range(len(c)):
            for a2 in range(len(c[a1])):
                if np.array_equal(c[a1][a2], d):
                    return a1
    
    #Cluster Connectedness - minimization
    @staticmethod
    def cluster_connectedness(data, centroids, clusters, number_neighbors=4):
        cc = 0.0
        np.sort(data, axis=0)
        for d in range(len(data)-number_neighbors):
            index_1 = idx = Metrics.ismember(data[d], clusters)
            for neighbor in range(1, number_neighbors):
                index_2 = idx = Metrics.ismember(data[d+neighbor], clusters)
                if index_1 != index_2:
                    cc += truediv(1, neighbor)
        return cc

    #Ball & Hall index
    @staticmethod
    def ball_hall_index(data, centroids):
        return Metrics.intra_cluster_statistic(data, centroids) / len(data)
    
    #Total Within Cluster Variance
    @staticmethod
    def total_within_cluster_variance(data, centroids, clusters):
        # minimization
        twcv = 0.0
        n_d = len(data)
        n_c = len(centroids)
        n_dim = len(centroids[0])
        data_sum = 0.0
        data_sum_by_cluster = 0.0
        for k in range(n_c):
            for d in range(n_dim):
                by_dimension = 0.0
                for i in range(len(clusters[k])):
                    by_dimension += clusters[k][i] ** 2
            if len(clusters[k]) == 0.0:
                data_sum_by_cluster = 0.0
            else:
                data_sum_by_cluster += truediv(by_dimension, len(clusters[k]))
        for i in range(n_d):
            for d in range(n_dim):
                data_sum += data[i][d] ** 2.0
        data_sum -= data_sum_by_cluster
        return data_sum
    
    #Intracluster Entropy - maximization
    @staticmethod
    def intra_cluster_entropy(data, centroids):
        h = 0.0
        for k in range(len(centroids)):
            h += (1 - Metrics.intra_cluster_entropy_h(data, centroids[k], k) * Metrics.intra_cluster_entropy_g(data, centroids[k], k)) ** truediv(1, len(centroids))
        return h

    #Intracluster Entropy
    @staticmethod
    def intra_cluster_entropy_h(data, centroid_k, k):
        g = Metrics.intra_cluster_entropy_g(data, centroid_k, k)
        return -((g * math.log(g, 2)) + ((1 - g) * (math.log(1 - g))))

    #Intracluster Entropy
    @staticmethod
    def intra_cluster_entropy_g(data, centroid_k, k):
        g = 0.0
        for i in range(len(data)):
            g += (0.5 + truediv(Metrics.co(data[i], centroid_k), 2))
        return truediv(g, len(data))

    #Intracluster Entropy
    @staticmethod
    def co(data_i, centroids_k):
        co = 0.0
        data_sum = 0.0
        centroids_sum = 0.0
        for d in range(len(centroids_k)):
            co += data_i[d] * centroids_k[d]
            data_sum += data_i[d] ** 2
            centroids_sum += centroids_k[d] ** 2
        data_sum **= truediv(1, 2)
        centroids_sum **= truediv(1, 2)
        if data_sum == 0.0 or centroids_sum == 0.0:
            return 0.0
        co = truediv(co, data_sum * centroids_sum)
        return co

    #Hartigan
    @staticmethod
    def hartigan_index(data, centroids):
        return np.log(Metrics.inter_cluster_statistic(centroids) / Metrics.intra_cluster_statistic(data, centroids))

    #Xu Index
    @staticmethod
    def xu_index(data, centroids):
        D = len(centroids[0])
        N = len(data)
        K = len(centroids)
        return D * np.log(Metrics.intra_cluster_statistic(data, centroids) / (D * N**2)) + np.log(K)

    #Ratkowsky & Lance index - maximized
    @staticmethod
    def rl(data, centroids, clusters):
        dimensions = len(centroids[0])
        rl = 0.0
        for d in range(dimensions):
            for k in range(0,len(centroids)):
                for l in range(1,len(centroids)):
                    if k != l:
                        rl += truediv(Metrics.sum_c_c(centroids, k, l), Metrics.sum_x_c(clusters, centroids, k)) ** truediv(1, 2.0)
        return truediv(rl, len(centroids)**truediv(1,2.0))
    
    #Ratkowsky & Lance index
    @staticmethod
    def sum_c_c(centroidsx, k, l):
        s = 0.0
        for d in range(len(centroidsx[k])):
            s += (centroidsx[k][d] - centroidsx[l][d]) ** 2.0
        return s ** truediv(1,2.0)
    
    #Ratkowsky & Lance index
    @staticmethod
    def sum_x_c(clusters, centroids, k):
        s = 0.0
        for d_1 in range(len(clusters[k])):
            for d_2 in range(len(centroids[k])):
                s += (clusters[k][d_1] - centroids[k][d_2]) ** 2.0
        return s ** truediv(1, 2.0)

    #WB index
    @staticmethod
    def wb_index(data, centroids):
        return len(centroids) * (Metrics.intra_cluster_statistic(data, centroids) / Metrics.inter_cluster_statistic(centroids))

    #Dunn Index - maximization
    @staticmethod
    def dunn_index(data, centroids, clusters):
        di = 0.0
        min_l = float('inf')
        min_k = float('inf')
        max_k = 0.0
        for k in range(len(centroids)):
            x = Metrics.maximum_distance_between_data_between_centroids(clusters[k])
            if max_k < x:
                max_k = x
        for k in range(0, len(centroids)-1):
            for l in range(1, len(centroids)):
                if k != l:
                    md = truediv(Metrics.minimum_distance_between_data_between_centroids(clusters[k], clusters[l]), max_k)
                    if min_l > md:
                        min_l = md
            if min_k > min_l:
                min_k = min_l
        return min_k
    
    #Dunn Index
    @staticmethod
    def minimum_distance_between_data_between_centroids(cluster_1, cluster_2):
        min_de = float('inf')
        for data_centroid_one in range(len(cluster_1)):
            for data_centroid_two in range(len(cluster_2)):
                de = distance.euclidean(cluster_1[data_centroid_one], cluster_2[data_centroid_two])
                if min_de > de:
                    min_de = de
        return min_de

    #Dunn Index
    @staticmethod
    def maximum_distance_between_data_between_centroids(cluster_1):
        max = 0.0
        for data_centroid_one in range(len(cluster_1)):
            for data_centroid_two in range(len(cluster_1)):
                de = distance.euclidean(cluster_1[data_centroid_one], cluster_1[data_centroid_two])
                if max < de:
                    max = de
        return max

    #Davies Bouldin minimization
    @staticmethod
    def davies_bouldin(data, centroids, clusters):
        n_c = len(centroids)
        n_d = len(data)
        db = 0.0
        for k in range(n_c):
            db += Metrics.maximum_d_b(data, centroids, k, clusters)
        return truediv(db, n_c)
    
    #Davies Bouldin minimization
    @staticmethod
    def maximum_d_b(data, centroids, k, clusters):
        max_distance = 0.0
        for l in range(len(centroids)):
            if k != l:
                db = truediv(Metrics.sc_r(centroids, clusters, k) + Metrics.sc_r(centroids, clusters, l), distance.euclidean(centroids[k], centroids[l]))
                if max_distance < db:
                    max_distance = db
        return max_distance
    
    #Davies Bouldin minimization
    @staticmethod
    def sc_r(centroids, clusters, k):
        sc = 0.0
        for i in range(len(clusters[k])):
            sc += distance.euclidean(clusters[k][i], centroids[k])
        if len(clusters[k]) == 0:
            return 0.0
        return truediv(sc, len(clusters[k]))

    #CS-measure
    @staticmethod
    def cs_measure(data, centroids, clusters):
        cs = 0.0
        min_sum = 0.0
        max_k = 0.0
        min_k = distance.euclidean(centroids[0], centroids[1])
        for k in range(len(centroids)):
            x = Metrics.maximum_distance_between_data_between_centroids(clusters[k])
            if max_k < x:
                max_k = x
            for l in range(len(centroids)):
                if k != l:
                    y = distance.euclidean(centroids[k], centroids[l])
                    if min_k > y:
                        min_k = y
            min_sum += min_k
        for k in range(len(centroids)):
            if len(clusters[k]) > 0:
                cs += truediv(max_k, len(clusters[k]))
        cs = truediv(cs, min_sum)
        return cs
    
    #Min-Max cut
    @staticmethod
    def min_max_cut(data, centroids, clusters):
        mmc = 0.0
        mmc_den = 0.0
        for k1 in range(len(clusters)):
            for k2 in range(len(clusters)):
                for kk1 in range(len(clusters[k1])):
                    for kk2 in range(len(clusters[k2])):
                        if k1 != k2:
                            mmc += distance.euclidean(clusters[k1][kk1], clusters[k2][kk2])
                        if k1 == k2:
                            mmc_den += distance.euclidean(clusters[k1][kk1], clusters[k2][kk2])
        mmc = truediv(mmc, mmc_den)
        return -mmc

    @staticmethod
    def evaluate(metric, dataset, centroids, clusters, algorithm, k):
        if metric == 'inter-cluster':
            return Metrics.inter_cluster_statistic(centroids)
        elif metric == 'cluster-separation':
            return Metrics.cluster_separation(centroids)
        elif metric == 'abgss':
            return Metrics.abgss(dataset, centroids, clusters)
        elif metric == 'edge-index':
            return Metrics.edge_index(dataset, centroids, clusters)
        elif metric == 'cluster-connectedness':
            return Metrics.cluster_connectedness(dataset, centroids, clusters)
        elif metric == 'intra-cluster':
            return Metrics.intra_cluster_statistic(dataset, centroids)
        elif metric == 'ball-hall':
            return Metrics.ball_hall_index(dataset, centroids)
        elif metric == 'twc-variance':
            return Metrics.total_within_cluster_variance(dataset, centroids, clusters)
        elif metric == 'intracluster-entropy':
            return Metrics.intra_cluster_entropy(dataset, centroids)
        elif metric == 'ch-index':
            return Metrics.variance_based_ch(dataset, centroids)
        elif metric == 'hartigan':
            return Metrics.hartigan_index(dataset, centroids)
        elif metric == 'xu-index':
            return Metrics.xu_index(dataset, centroids)
        elif metric == 'rl-index':
            return np.mean(Metrics.rl(dataset, centroids, clusters))
        elif metric == 'wb-index':
            return Metrics.wb_index(dataset, centroids)
        elif metric == 'dunn-index':
            return Metrics.dunn_index(dataset, centroids, clusters)
        elif metric == 'davies-bouldin':
            return Metrics.davies_bouldin(dataset, centroids, clusters)
        elif metric == 'cs-measure':
            return Metrics.cs_measure(dataset, centroids, clusters)
        elif metric == 'silhouette':
            return Metrics.silhouette(clusters, len(dataset))
        elif metric == 'min-max-cut':
            return Metrics.min_max_cut(dataset, centroids, clusters)
        elif metric == 'gap':
            random_data = np.random.uniform(0, 1, dataset.shape)
            ag_aux = algorithm(data=random_data)
            ag_aux.fit(k=k)
            random_clusters = ag_aux.clusters
            return Metrics.gap_statistic(clusters, random_clusters)