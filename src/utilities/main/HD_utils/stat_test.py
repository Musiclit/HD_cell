'''Functions that perform statistical testing'''
from statsmodels.formula.api import ols
import pandas as pd
import numpy as np
from scipy import stats

from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from statsmodels.stats.multitest import multipletests


def stat_test_89(indv, dependv, df):
    lm_fnoise = ols(f'{dependv} ~ C({indv})', data=df).fit()
    a = lm_fnoise.conf_int().iloc[1:].rename(columns={0: '2.5%', 1: '97.5%'})
    b = df.loc[:,[indv, dependv]].groupby([indv]).mean().reset_index().to_string(index=False)
    print(f'{a}\n\n{b}')


def stat_test_85(indv, dependv, df):
    lm_fnoise = ols(f'{dependv} ~ {indv}', data=df).fit()
    a = lm_fnoise.conf_int().iloc[1:].rename(columns={0: '2.5%', 1: '97.5%'})
    b = df.loc[:,[indv, dependv]].groupby([indv]).mean().reset_index().to_string(index=False)
    print(f'{a}\n\n{b}')


def cal_loc_sep_cohens_d(stat_df, type1='left', type2='right', thres=0, center_stat='mean', ifprint=True):

    agg_stats = stat_df.groupby(['fish', 'ring'], observed=True)[['x', 'y', 'z']].agg([center_stat, 'std', 'count'])
    
    agg_stats.columns = [f'{stat}_{col}' for col, stat in agg_stats.columns]
    ring_loc = (agg_stats.reset_index()
                .drop(columns=['count_y', 'count_z'])  # Remove duplicate count columns
                .rename(columns={'count_x': 'count'})
                .dropna()
                .reset_index(drop=True))


    ring_loc_sel = ring_loc.loc[ring_loc['ring'].isin([type1, type2])]

    unqi = np.unique(ring_loc_sel['fish'], return_counts=True)
    fish_ids = unqi[0][unqi[1] > 1]
    
    ring_loc_sel = ring_loc_sel.loc[ring_loc_sel['fish'].isin(fish_ids)].reset_index(drop=True)
    
    cohen_d_list = []
    coor_names = ['x', 'y', 'z']
    for coor in coor_names:
        
        mean_dis = ring_loc_sel.groupby('fish').apply(lambda x: x.loc[x.ring==type1, f'{center_stat}_{coor}'].values[0] - x.loc[x.ring==type2, f'{center_stat}_{coor}'].values[0]).values
        sd_weight = ring_loc_sel.groupby('fish')[[f'std_{coor}', 'count']].apply(lambda x: np.average(x[f'std_{coor}']**2, weights=x['count']-1)).reset_index()[0].values
        cohen_d = mean_dis / np.sqrt(sd_weight)
        cohen_d_list.append(cohen_d)

    min_count = ring_loc_sel.groupby('fish')['count'].min().values
    cohen_d_df = pd.DataFrame({'fish': fish_ids, 'cohen_d_x': cohen_d_list[0], 'cohen_d_y': cohen_d_list[1], 'cohen_d_z': cohen_d_list[2], 'cell_num': min_count})

    cohen_ds = ['cohen_d_x', 'cohen_d_y', 'cohen_d_z']
    test_df = cohen_d_df.loc[cohen_d_df.cell_num >= thres]
    if ifprint:
        print(f'Threshold: {thres}; ', len(test_df), 'fish')
    for cohen_d in cohen_ds:
        t_stat, p_val = stats.ttest_1samp(test_df[cohen_d].values, 0)
        if ifprint:
            print(f"{cohen_d}: t ={t_stat: .3f}, p ={p_val: .3f}")
        
    return cohen_d_df


def cal_loc_sep_mean(stat_df, type1='left', type2='right', center_stat='mean', ifprint=True):

    agg_stats = stat_df.groupby(['fish', 'ring'], observed=True)[['x', 'y', 'z']].agg([center_stat, 'std', 'count'])
    
    agg_stats.columns = [f'{stat}_{col}' for col, stat in agg_stats.columns]
    ring_loc = (agg_stats.reset_index()
                .drop(columns=['count_y', 'count_z'])  # Remove duplicate count columns
                .rename(columns={'count_x': 'count'})
                .dropna()
                .reset_index(drop=True))
    ring_loc_sel = ring_loc.loc[ring_loc['ring'].isin([type1, type2])]

    unqi = np.unique(ring_loc_sel['fish'], return_counts=True)
    # print(unqi)
    fish_ids = unqi[0][unqi[1] > 1] # Have both left and right rings
    
    ring_loc_sel = ring_loc_sel.loc[ring_loc_sel['fish'].isin(fish_ids)].reset_index(drop=True) 
    
    mean_dif_list = []
    coor_names = ['x', 'y', 'z']
    for coor in coor_names:
        
        mean_dis = ring_loc_sel.groupby('fish').apply(lambda x: x.loc[x.ring==type1, f'{center_stat}_{coor}'].values[0] - x.loc[x.ring==type2, f'{center_stat}_{coor}'].values[0]).values
        mean_dif_list.append(mean_dis)

    meandif_d_df = pd.DataFrame({'fish': fish_ids, 'left_right': mean_dif_list[0], 'posterior_anterior': mean_dif_list[1], 'ventral_dorsal': mean_dif_list[2]})
    
    coors = ['left_right', 'posterior_anterior', 'ventral_dorsal']

    t_and_p_list = []
    for coor in coors:
        t_stat, p_val = stats.ttest_1samp(meandif_d_df[coor].values, 0)
        if ifprint:
            print(f"{coor}: t({len(meandif_d_df[coor].values)}) ={t_stat: .3f}, p ={p_val: .3f}")
        else:
            t_and_p_list.append((coor, t_stat, p_val))
            
    return meandif_d_df, t_and_p_list


def cal_loc_cluster_sep_cohens_d(stat_df, type1='left', type2='right', thres=0):

    agg_stats = stat_df.groupby(['fish', 'anatomical_loc'])[['x', 'y', 'z']].agg(['mean', 'std', 'count'])
    
    agg_stats.columns = [f'{stat}_{col}' for col, stat in agg_stats.columns]
    ring_loc = (agg_stats.reset_index()
                .drop(columns=['count_y', 'count_z'])  # Remove duplicate count columns
                .rename(columns={'count_x': 'count'})
                .dropna()
                .reset_index(drop=True))


    ring_loc_sel = ring_loc.loc[ring_loc['anatomical_loc'].isin([type1, type2])]

    unqi = np.unique(ring_loc_sel['fish'], return_counts=True)
    fish_ids = unqi[0][unqi[1] > 1]
    
    ring_loc_sel = ring_loc_sel.loc[ring_loc_sel['fish'].isin(fish_ids)].reset_index(drop=True)
    
    cohen_d_list = []
    coor_names = ['x', 'y', 'z']
    for coor in coor_names:
        
        mean_dis = ring_loc_sel.groupby('fish').apply(lambda x: x.loc[x.anatomical_loc==type1, f'mean_{coor}'].values[0] - x.loc[x.anatomical_loc==type2, f'mean_{coor}'].values[0]).values
        sd_weight = ring_loc_sel.groupby('fish')[[f'std_{coor}', 'count']].apply(lambda x: np.average(x[f'std_{coor}']**2, weights=x['count']-1)).reset_index()[0].values
        cohen_d = mean_dis / np.sqrt(sd_weight)
        cohen_d_list.append(cohen_d)

    min_count = ring_loc_sel.groupby('fish')['count'].min().values
    cohen_d_df = pd.DataFrame({'fish': fish_ids, 'cohen_d_x': cohen_d_list[0], 'cohen_d_y': cohen_d_list[1], 'cohen_d_z': cohen_d_list[2], 'cell_num': min_count})

    cohen_ds = ['cohen_d_y', 'cohen_d_z']
    test_df = cohen_d_df.loc[cohen_d_df.cell_num >= thres]
    print(f'Threshold: {thres}; ', len(test_df), 'fish')
    for cohen_d in cohen_ds:
        t_stat, p_val = stats.ttest_1samp(test_df[cohen_d].values, 0)
        print(f"{cohen_d}: t ={t_stat: .3f}, p ={p_val: .3f}")
        
    return cohen_d_df


def cal_loc_sep_confusion_matrix(stat_df):
    
    fish_ids = stat_df['fish'].unique()
    
    cms = []
    for fish_count, fishi in enumerate(fish_ids):
        df_fish = stat_df.loc[stat_df['fish'] == fishi]
        
        cm = confusion_matrix(df_fish.anatomical_loc, df_fish.ring, labels=['left', 'right']) # labels: only use left and right rings and anatomical_loc
        cms.append(cm)
        
    cm_all = np.zeros((2, 2))
    for fish_count, fishi in enumerate(fish_ids):
        cm = cms[fish_count]
        cm_all = cm_all + cm
        
    return cm_all, cms


def cal_loc_sep_confusion_matrix_all_ring(stat_df):
    
    stat_df = stat_df.copy()
    stat_df['ring'] = stat_df['ring'].cat.rename_categories(lambda x: x.lower())
    stat_df['anatomical_loc'] = stat_df['anatomical_loc'].apply(lambda x: x.lower())
    fish_ids = stat_df['fish'].unique()
    cms = []
    for fish_count, fishi in enumerate(fish_ids):
        df_fish = stat_df.loc[stat_df['fish'] == fishi]
        cm = confusion_matrix(df_fish.anatomical_loc, df_fish.ring, labels=['symmetric', 'left', 'right'])
        # print(cm)
        cms.append(cm)

    return cms


def corrected_p_df(stat_df, method='Bonferroni'):
    
    pvals = stat_df['beta2_p']
    _, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method=method)
    stat_df['beta2_p_corrected'] = pvals_corrected

    stat_df['ring'] = ['center'] * len(stat_df)
    stat_df.loc[    (stat_df['beta2_p_corrected'] < 0.05) & (stat_df['beta2'] > 0)   , 'ring'] = 'right'
    stat_df.loc[    (stat_df['beta2_p_corrected'] < 0.05) & (stat_df['beta2'] < 0)   , 'ring'] = 'left'
    
    return stat_df


'''Archive '''

# def cal_loc_sep_anatomical_cluster(stat_df):
#     stat_df2 = stat_df.copy()
    
#     # Clust each fish HD cells' anatomical location into 2 clusters
#     # Get unique fish
#     fish_ids = stat_df['fish'].unique()
#     kmeans_list = []
#     labels_list = []

#     for i, fish in enumerate(fish_ids):
#         # Get data for this fish
#         df_fish = stat_df[stat_df['fish'] == fish]
        
#         X = df_fish[['z', 'x', 'y']].values
        
#         # Apply KMeans clustering
#         kmeans = KMeans(n_clusters=2, random_state=0, n_init=1).fit(X)
#         kmeans_list.append(kmeans)
        
#         # assign names to clusters
#         centers = kmeans.cluster_centers_
#         x1 = centers[0, 1]
#         x2 = centers[1, 1]
#         # print(x1, x2)
#         labels = np.zeros_like(kmeans.labels_, dtype='U30')
#         if x1 < x2:
#             labels[kmeans.labels_ == 0] = 'left'
#             labels[kmeans.labels_ == 1] = 'right'
#         else:
#             labels[kmeans.labels_ == 0] = 'right'
#             labels[kmeans.labels_ == 1] = 'left'
        
#         labels_list.append(labels)
        
#     labels_all = np.concatenate(labels_list)
    
#     stat_df2['anatomical_loc'] = labels_all

#     cms = []

#     for fish_count, fishi in enumerate(fish_ids):
#         df_fish = stat_df2.loc[stat_df2['fish'] == fishi]
        
#         cm = confusion_matrix(df_fish.anatomical_loc, df_fish.ring, labels=['left', 'right'])
#         cms.append(cm)
        
#     cm_all = np.zeros((2, 2))
#     for fish_count, fishi in enumerate(fish_ids):
#         cm = cms[fish_count]
#         cm_all = cm_all + cm
        
#     return cm_all, cms, stat_df2

# from sklearn.mixture import GaussianMixture

# def cal_loc_sep_anatomical_cluster_GM(stat_df):
#     stat_df2 = stat_df.copy()
    
#     # Clust each fish HD cells' anatomical location into 2 clusters
#     # Get unique fish
#     fish_ids = stat_df['fish'].unique()
#     fitob_list = []
#     labels_list = []

#     for i, fish in enumerate(fish_ids):
#         # Get data for this fish
#         df_fish = stat_df[stat_df['fish'] == fish]
        
#         X = df_fish[['z', 'x', 'y']].values
        
#         # Apply KMeans clustering
#         fitob = GaussianMixture(n_components=2, random_state=0).fit(X)
#         fitob_list.append(fitob)
        
#         # assign names to clusters
#         centers = fitob.means_
#         x1 = centers[0, 1]
#         x2 = centers[1, 1]
#         # print(x1, x2)
#         labels_num = fitob.predict(X)
#         labels = np.zeros_like(labels_num, dtype='U30')
#         if x1 < x2:
#             labels[labels_num == 0] = 'left'
#             labels[labels_num == 1] = 'right'
#         else:
#             labels[labels_num == 0] = 'right'
#             labels[labels_num == 1] = 'left'
        
#         labels_list.append(labels)
        
#     labels_all = np.concatenate(labels_list)
    
#     stat_df2['anatomical_loc'] = labels_all

#     cms = []

#     for fish_count, fishi in enumerate(fish_ids):
#         df_fish = stat_df2.loc[stat_df2['fish'] == fishi]
        
#         cm = confusion_matrix(df_fish.anatomical_loc, df_fish.ring, labels=['left', 'right'])
#         cms.append(cm)
        
#     cm_all = np.zeros((2, 2))
#     for fish_count, fishi in enumerate(fish_ids):
#         cm = cms[fish_count]
#         cm_all = cm_all + cm
        
#     return cm_all, cms, stat_df2