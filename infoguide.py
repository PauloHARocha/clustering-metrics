import pandas as pd
import numpy as np
import scipy.stats as stats

#InfoGuide
def infoguide(X, k_min, k_max, labels, f_equal):
    df_info = pd.DataFrame()
    for k in range(k_min, k_max):
        l_k = labels['labels'][labels['k'] == k].reset_index(drop=True)
        l_k_1 = labels['labels'][labels['k'] == (k + 1)].reset_index(drop=True)
        for i_k_1 in l_k_1.unique():
            c_k_1 = X[l_k_1 == i_k_1]
            for i_k in l_k.unique():
                c_k = X[l_k == i_k]
                df_info = pd.concat(
                    [df_info, f_equal(c_k_1, c_k, i_k_1, i_k, k)]
                    ,ignore_index=True)
    return df_info


def f_equal_ks(X_k_1, X_k, i_k_1, i_k, k):
    feats = X_k_1.columns
    n_feats = feats.shape[0]
    
    statistics, pvalues = [], []
    for feat in feats:
        stat, pval = stats.ks_2samp(X_k[feat], X_k_1[feat])
        statistics.append(stat)
        pvalues.append(pval)

    return pd.DataFrame({
        'c_k+1': np.full(n_feats, i_k_1),
        'c_k': np.full(n_feats, i_k),
        'feat': feats,
        'k': np.full(n_feats, k),
        'k+1': np.full(n_feats, k+1),
        'statistic': statistics,
        'pvalue': pvalues
    })



def infoguide_optimal_k(df_info, k_min, k_max, p_ref, min_feat=0):

    for k in range(k_min, k_max):
        df_info_k = df_info[df_info['k'] == k]

        df_info_k.loc[:, 'pvalue_ref'] = df_info_k.apply(
            lambda x: p_ref/(x['k']*(x['k'] + 1)*x['n_feat']), axis=1)

        df_info_k.loc[:, 'match_feat'] = df_info_k.apply(
            lambda x: True if x['pvalue'] > x['pvalue_ref'] else False, axis=1)

        df_info_k = df_info_k.groupby(['c_k+1', 'c_k', 'k', 'n_feat',
                                     'dataset', 'algorithm', 'pvalue_ref']).agg({'match_feat': 'sum'}).reset_index()

        df_info_k.loc[:, 'match_cluster'] = df_info_k.apply(
            lambda x: True if x['match_feat'] >= (x['n_feat']-min_feat) else False, axis=1)

        df_info_k = df_info_k.groupby(
            ['c_k+1', 'k', 'dataset', 'algorithm', 'pvalue_ref']).agg({'match_cluster': 'sum'})

        if 0 not in df_info_k['match_cluster'].unique():
            return k
    return -1
