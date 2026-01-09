import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from scipy import stats
from scipy.stats import pearsonr

from HD_utils.stat_test import cal_loc_sep_mean, cal_loc_sep_confusion_matrix_all_ring
from HD_utils.defaults import *

COLORMAP_S = {'Left': 'deepskyblue', 'Right': 'red', 'Symmetric': 'Black', 'CW': 'deepskyblue', 'CCW': 'red'}
COLORMAP = {'Left ring': 'deepskyblue', 'Right ring': 'red', 'Symmetric ring': 'Black', 'CW ring': 'deepskyblue', 'CCW ring': 'red'}
NBINS = 21
THRESHOLDS = [(0,np.inf), (0,5), (5,10), (10,20), (20,40), (40,np.inf)]

def cal_ratio_ring_num(stat_noHD_sel):
    
    stat_noHD_ring_num = stat_noHD_sel.groupby(['fish', 'ring'], observed=False).size().reset_index(name='cell num')
    stat_noHD_ring_num['cell num'] += 0.1  # To avoid zero values when calculating log
    
    stat_noHD_total_num = stat_noHD_sel.groupby('fish').agg(total_num=('cell', 'count')).reset_index()
    stat_noHD_ring_num = stat_noHD_ring_num.merge(stat_noHD_total_num, on='fish', how='left')
    
    stat_noHD_ring_num['cell ratio'] = stat_noHD_ring_num['cell num'] / stat_noHD_ring_num['total_num']
    
    return stat_noHD_ring_num

def plot_fig_corr_turn_ring_ratio(ring_num_df, turn_stat):
    # Create figure
    ncol = 3
    nrow = 4
    fig, axes = plt.subplots(nrow,ncol,figsize=(7.5,10.5))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    
    ring_label_dict = {'left': 'CW', 'right': 'CCW', 'symmetric': 'symmetric'}
    turn_label_dict = {'No. of CCW tunrs': 'Number of turns:\nCCW', 
                       'No. of CW turns': 'Number of turns:\nCW', 
                       'Ratio: CCW/CW': 'Ratio of turns:\nlog(CCW / CW)'}
    
    for i, ring in enumerate(['left', 'right', 'symmetric', 'ratio']):
        if ring != 'ratio':
            aring_num = np.log(ring_num_df.loc[ring_num_df.ring == ring, 'cell ratio'].values)
        else:
            right_num = ring_num_df.loc[ring_num_df.ring == 'right', 'cell ratio'].values
            left_num = ring_num_df.loc[ring_num_df.ring == 'left', 'cell ratio'].values
            aring_num = np.log(right_num / left_num)
        for j, turn in enumerate(['No. of CCW tunrs', 'No. of CW turns', 'Ratio: CCW/CW']):
            ax = axes[i, j]
            aturn = turn_stat[turn][FISH_IDS] if j < 2 else np.log(turn_stat[turn][FISH_IDS])  # Log transform for ratio
            ax.scatter(aturn, aring_num, label=f'{ring}+{turn}', alpha=0.5, color='k')
            # ax.legend()
            if j == 0:
                if i != 3:
                    ax.set_ylabel(f'Ratio of cells:\nlog({ring_label_dict[ring]} / all)')
                else:
                    ax.set_ylabel(f'Ratio of cells:\nlog(CCW / CW)')
                ax.yaxis.set_label_coords(-0.24, 0.5)
            else:
                ax.set_yticklabels([])
            
            if i == 3:
                ax.set_xlabel(f'{turn_label_dict[turn]}')
            else:
                ax.set_xticklabels([])
            
            # calculate correlation
            corr, pval = pearsonr(aturn, aring_num)
            ax.set_title(f'Pearson\'s $r = {corr:.3f}, p = {pval:.3f}$', fontsize=9)
            
    return fig, axes

def plot_fig_corr_turn_ring(ring_num_df, turn_stat):
    # Create figure
    fig, axes = plt.subplots(4,3,figsize=(7.5,10.5))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    for i, ring in enumerate(['left', 'right', 'symmetric', 'ratio']):
        if ring != 'ratio':
            aring_num = ring_num_df.loc[ring_num_df.ring == ring, 'cell num'].values
        else:
            right_num = ring_num_df.loc[ring_num_df.ring == 'right', 'cell num'].values
            left_num = ring_num_df.loc[ring_num_df.ring == 'left', 'cell num'].values
            aring_num = np.log(right_num / left_num)
        for j, turn in enumerate(['No. of CCW tunrs', 'No. of CW turns', 'Ratio: CCW/CW']):
            ax = axes[i, j]
            aturn = turn_stat[turn][FISH_IDS] if j < 2 else np.log(turn_stat[turn][FISH_IDS])  # Log transform for ratio
            ax.scatter(aturn, aring_num, label=f'{ring}+{turn}', alpha=0.5, color='k')
            # ax.legend()
            if j == 0:
                if i != 3:
                    ax.set_ylabel(f'No. of {ring}-ring cells')
                else:
                    ax.set_ylabel(f'Log ratio: right/left')
                ax.yaxis.set_label_coords(-0.2, 0.5)
            else:
                ax.set_yticklabels([])
            
            if i == 3:
                if j < 2:
                    ax.set_xlabel(f'{turn}')
                else:
                    ax.set_xlabel(f'Log ratio: CCW/CW')
            else:
                ax.set_xticklabels([])
            
            # calculate correlation
            corr, pval = pearsonr(aturn, aring_num)
            ax.set_title(f'Pearson\'s $r = {corr:.3f}, p = {pval:.3f}$', fontsize=9)
            
    return fig, axes

def calculate_pij_minuspipj(stat_df):
    
    cm33s = cal_loc_sep_confusion_matrix_all_ring(stat_df)

    prob_lonrs = []
    prob_ronls = []
    for i, cm in enumerate(cm33s):
        total_num = np.sum(cm)
        prob_lonrs.append( cm[2,1]/total_num - (cm[2,1]+cm[2,0]+cm[2,2])*(cm[2,1]+cm[1,1]+cm[0,1])/total_num**2 ) 
        prob_ronls.append( cm[1,2]/total_num - (cm[1,2]+cm[1,1]+cm[1,0])*(cm[1,2]+cm[2,2]+cm[0,2])/total_num**2 )


    joint_prob_df = pd.DataFrame({'variable': ['$a$: left ring\n$b$: right side']*len(prob_lonrs) + ['$a$: right ring\n$b$: left side']*len(prob_ronls), 'value': prob_lonrs + prob_ronls})

    lstats = stats.wilcoxon(prob_lonrs)
    print(f'L ring on R side, stat: {lstats.statistic:.3e}, p: {lstats.pvalue:.3e}')
    rstats = stats.wilcoxon(prob_ronls)
    print(f'R ring on L side, stat: {rstats.statistic:.3e}, p: {rstats.pvalue:.3e}')
    
    return joint_prob_df

def plot_pij_minuspipj(joint_prob_df, ax):

    im1 = sns.barplot(data=joint_prob_df, x='variable', y='value', ax=ax, errorbar='se', color='black', fill=False, linewidth=1)
    im2 = sns.stripplot(
        x="variable", 
        y="value", 
        data=joint_prob_df, dodge=True, alpha=0.4, ax=ax, jitter=0.2, legend=False, color='black'
    )

    ax.set_ylabel('$p(i,j)-p(i)p(j)$')
    ax.set_xlabel('')
    remove_top_right_spines(ax)
    ax.set_xticks([0,1], ['$i:$CW-ring cell           \n$j:$right side           ', '             $i:$CCW-ring cell\n             $j:$left side'], ha='center', va='top')
    
    return [im1, im2]
    

def plot_median_loc_ttest(stat_df, ax):

    meandif_df, t_and_p_list = cal_loc_sep_mean(stat_df, center_stat='median', ifprint=True, type1='Left', type2='Right')
    meandif_df.rename(columns={'left_right': 'Right\nleft', 'posterior_anterior': 'Anterior\nposterior', 'ventral_dorsal': 'Dorsal\nventral'}, inplace=True)

    meandif_df_long = pd.melt(meandif_df, id_vars='fish', value_vars=meandif_df)
    img = sns.barplot(data=meandif_df_long, y='variable', x='value', ax=ax, errorbar='se', color='black', orient='h', fill=False, linewidth=1)
    sns.stripplot(
        y="variable", 
        x="value", 
        data=meandif_df_long, dodge=True, alpha=0.3, ax=ax, jitter=0.2, legend=False, color='black', orient='h'
    )

    # for i in range(3):
    #     ax.text(100, i+0.08, f'$p={t_and_p_list[i][2]:.3f}$', ha='left', va='center')

    ax.set_xlabel('Difference of median:\nleft ring $-$ right ring (μm)')
    ax.set_ylabel('')
    remove_top_right_spines(ax)
    
    return img

def plot_median_loc(stat_df, ax):
    stat_df_sum = stat_df.groupby(['fish', 'ring'], observed=True)[['x', 'y', 'z']].median().reset_index()
    
    ims = []
    for fish in stat_df.fish.unique():
        df_fish = stat_df_sum[stat_df_sum.fish == fish]
        
        # Left ring
        df_ring = df_fish.loc[df_fish.ring == 'Left']
        
        have_left = len(df_ring) > 0
        if have_left:
            left_x = df_ring.x.values[0]
            left_y = df_ring.y.values[0]
            im3 = ax.scatter(left_x, left_y, color='deepskyblue', alpha=0.7, label='left', s=10)
            ims.append(im3)

        # Right ring
        df_ring = df_fish.loc[df_fish.ring == 'Right']
        
        have_right = len(df_ring) > 0
        if have_right:
            right_x = df_ring.x.values[0]
            right_y = df_ring.y.values[0]
            im4 = ax.scatter(right_x, right_y, color='red', alpha=0.7, label='right', s=10)
            ims.append(im4)
        
        # Symmetric ring
        df_ring = df_fish.loc[df_fish.ring == 'Symmetric']
        if len(df_ring) > 0:
            sx = df_ring.x.values[0]
            sy = df_ring.y.values[0]
            im2 = ax.scatter(sx, sy, color='black', alpha=0.4, label='Symmetric', s=10)
            ims.append(im2)
        
        # Connect left and right
        if have_left and have_right:
            im1 = ax.plot([left_x, right_x], [left_y, right_y], marker='o', label=fish, alpha=0.2, c='grey', markersize=0, lw=1)
            ims.append(im1)
        
    ax.set_ylabel('Posterior to anterior (µm)')
    remove_top_right_spines(ax)
    ax.set_aspect('equal', adjustable='datalim')
    
    return ims

def plot_prop_each_cel_type(prop, ax):
    
    prop = prop.copy()
    prop.prop *= 100  # Convert to percentage
    prop.speed_modu = prop.speed_modu.cat.rename_categories({'Increase': 'Positive', 'Decrease': 'Negative', 'None': 'Independent'})
    prop.ring = prop.ring.cat.rename_categories({'Left': 'CW', 'Right': 'CCW', 'Symmetric': 'Symmetric'})
    # print(np.unique(prop.speed_modu))
    
    sns.stripplot(
        x="speed_modu", 
        y="prop", 
        hue='ring',
        data=prop, dodge=True, alpha=0.2, ax=ax, palette=COLORMAP_S, jitter=0.2, legend=False, edgecolor='black', linewidth=1, zorder=2,
    )
    im = sns.barplot(prop, x='speed_modu', y='prop', hue='ring', edgecolor='black', ax=ax, alpha=0.4, palette=COLORMAP_S, errorbar='se', capsize=0.2, zorder=1)
    
    # # Use lines to connect dots
    # # Get the actual positions of the stripplot points
    # collections = [child for child in ax.get_children() if hasattr(child, 'get_offsets')]

    # # Create a mapping of (fly_id, cell_type_s, speed_modu) to actual x,y positions
    # point_positions = {}
    # counter = 0
    # for speed_modu in prop['speed_modu'].unique():
    #     for ring in (prop['ring'].unique()):
            
    #         collection = collections[counter]
    #         counter += 1
    #         offsets = collection.get_offsets()
    #         cell_data = prop[(prop['ring'] == ring) & (prop['speed_modu'] == speed_modu)].reset_index(drop=True)
            
    #         for j, (_, row) in enumerate(cell_data.iterrows()):
    #             if j < len(offsets):  # Safety check
    #                 point_positions[(row['fish'], row['ring'], row['speed_modu'])] = offsets[j]

    # # Add lines using actual positions
    # for fish in prop['fish'].unique():
    #     fish_data = prop[prop['fish'] == fish]
        
    #     if len(fish_data) > 1:
    #         fish_data = fish_data.sort_values(['speed_modu', 'ring'])
            
    #         x_coords = []
    #         y_coords = []
            
    #         for _, row in fish_data.iterrows():
    #             key = (row['fish'], row['ring'], row['speed_modu'])
    #             if key in point_positions:
    #                 pos = point_positions[key]
    #                 x_coords.append(pos[0])
    #                 y_coords.append(pos[1])
            
    #         if len(x_coords) > 1:  # Only plot if we have multiple points
    #             ax.plot(x_coords, y_coords, color='gray', alpha=0.2, linewidth=0.8, zorder=0)
    
    remove_top_right_spines(ax)
    
    return im

def cal_prop_each_cel_type(stat_noHD_sel):
    stat_noHD_ahb_prop = stat_noHD_sel.groupby(['ring', 'fish', 'speed_modu'], observed=False).agg(count=('fish', 'count')).reset_index()
    stat_noHD_ahb_propbase = stat_noHD_sel.groupby(['fish']).agg(count_base=('fish', 'count')).reset_index()
    stat_noHD_ahb_prop = stat_noHD_ahb_prop.merge(stat_noHD_ahb_propbase, on=['fish'])
    stat_noHD_ahb_prop['prop'] = stat_noHD_ahb_prop['count']/stat_noHD_ahb_prop.count_base
    return stat_noHD_ahb_prop

def plot_acv_on_AHV(data_sum, nbins, ax, iflegend, angVel_bin_edge):
    im = sns.lineplot(data=data_sum, x='angVel_bin', y='activity_norm', hue='ring', errorbar='se', ax=ax, palette=COLORMAP_S, markers='.', lw=2)

    maxbin_center = int((angVel_bin_edge[0] + angVel_bin_edge[1])/2 /np.pi * 180)
    midbini = (nbins-1)//4
    midbin_center = int((angVel_bin_edge[midbini] + angVel_bin_edge[midbini+1])/2 /np.pi * 180)
    ax.set_xlabel('AHV (degrees/s)')
    ax.set_xticks([0,1,midbini+1,(nbins+1)//2,(nbins+1)//2+midbini,nbins+1,nbins+2], [f"$<${maxbin_center}", maxbin_center, midbin_center, 0, -midbin_center, -maxbin_center, f"$>${-maxbin_center}"], rotation=90)

    ax.set_ylabel('Normalized $\\Delta F/F_0$')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if iflegend:
        legend = ax.get_legend()
        legend.set_title('')
        # legend.set_loc('upper left')
        # legend.set_bbox_to_anchor((0.3, 1.05))
    else:
        ax.get_legend().remove()
    
    return im

def plot_cell_location(ax, stat_df_fish, iflegend=True, ylim=(-100, 22), xlim=(-100, 90), equal_adjust='box', alpha=0.5):
    
    g = sns.scatterplot(stat_df_fish, x='x', y='y', hue='ring', 
                        ax=ax, palette=COLORMAP_S, alpha=alpha, s=20)

    ax.set_ylabel('Posterior to anterior (µm)')
    
    ax.set_xlabel('Left to right (µm)')
    ax.set_aspect('equal', adjustable=equal_adjust)
    
    if equal_adjust != 'datalim':
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
    
    if iflegend:
        legend = ax.get_legend()
        legend.set_title('Ring')
    else:
        ax.get_legend().remove()
    
    remove_top_right_spines(ax)
    
    return g



def plot_left_or_right_ring(ax, stat_df, ringtype, bin_num, legend=True):
    
    im = None
    
    stat_df_plot2 = stat_df[stat_df.ring == ringtype].copy()
    label_replace_dict = {'Left': 'CW', 'Right': 'CCW'}
    
    if len(stat_df_plot2) > 0:
        stat_df_plot2['ring'] = stat_df_plot2['ring'].astype(str).replace(label_replace_dict)
        stat_df_plot2['ring'] = stat_df_plot2['ring'] + ' ring'
        im = sns.histplot(data=stat_df_plot2, x='preferred_angle', hue='ring', palette=COLORMAP, ax=ax, binwidth=2*np.pi/bin_num, binrange=[-np.pi,np.pi], stat='count')

    ax.set_ylabel('')
    ax.set_ylim([0,28])
    if legend:

        legend = ax.get_legend()
        legend.set_title('')
        # legend.set_loc('upper right')
        legend.set_bbox_to_anchor((1, 1.1))
    else:
        ax.get_legend().remove()
    
    return im

def plot_symmetric_ring(ax, stat_df, bin_num, legend=True):
    
    stat_df_plot2 = stat_df[stat_df.ring == 'Symmetric'].copy()
    stat_df_plot2['ring'] = stat_df_plot2['ring'].astype(str)
    stat_df_plot2['ring'] = 'Symmetric ring'
    im = sns.histplot(data=stat_df_plot2, x='preferred_angle', hue='ring', palette={'Symmetric ring': 'black'}, ax=ax, binwidth=2*np.pi/bin_num, binrange=[-np.pi,np.pi], stat='count')

    ax.set_ylabel('')
    if legend:
        ax.set_ylim([0,125])

        legend = ax.get_legend()
        legend.set_title('')
        # legend.set_loc('upper right')
        legend.set_bbox_to_anchor((1, 1.1))
    else:
        ax.get_legend().remove()
    
    return im

def plot_all_rings(ax, stat_df, bin_num, legend=True):

    stat_df_plot2 = stat_df.copy()
    stat_df_plot2['ring'] = stat_df_plot2['ring'].astype(str)
    stat_df_plot2['ring'] = 'All rings'
    im = sns.histplot(data=stat_df_plot2, x='preferred_angle', hue='ring', palette={'All rings': 'orange'}, ax=ax, binwidth=2*np.pi/bin_num, binrange=[-np.pi,np.pi], stat='count')

    ax.set_ylabel('')
    if legend:
        ax.set_ylim([0,275])
        
        legend = ax.get_legend()
        legend.set_title('')
        # legend.set_loc('upper right')
        legend.set_bbox_to_anchor((1, 1.2))
    else:
        ax.get_legend().remove()
    
    return im

def set_radian_ticks(ax, bin_num, ticklabels=True):
    theta_range = np.linspace(-np.pi, np.pi, bin_num+1)
    deg_center = (theta_range[1:] + theta_range[:-1])/2*180/np.pi
    ax.set_xticks(deg_center[::1]/180*np.pi, np.round(deg_center[::1]).astype(int), rotation=45, ha='center')
    if not ticklabels:
        ax.tick_params(axis='x', labelbottom=False)
    ax.set_xlim([-np.pi, np.pi])
    
def remove_top_right_spines(ax):
    for pos in ['top', 'right']:
        ax.spines[pos].set_visible(False)
        
def remove_xticks_labels(ax):   
    ax.set_xticklabels([])
    ax.set_xlabel('')
    
def legend_only_at_top_and_with_title(ax, i, legendtitle=''):
    if i == 0:
        legend = ax.get_legend()
        legend.set_title(legendtitle)
    else:
        ax.get_legend().remove()
        
def xlabel_only_at_bottom(ax, i, nrow, xlabel):
    if i == nrow-1:
        ax.set_xlabel(xlabel)
        ax.xaxis.set_label_coords(0.5, -0.15)
    else:
        ax.set_xlabel('')
        
def no_xticklabel_before_bottom(ax, i, nrow):
    if i != nrow-1:
        ax.set_xticklabels([])