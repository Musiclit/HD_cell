import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

from lotr import LotrExperiment
from HD_utils.defaults import DATA_FOLDERS


def get_subset(df, col_name, start, end=None):
    """
    Get a subset of the DataFrame based on the values in the 'col' column.
    Uses binary search for efficiency.
    """
    end = start if end is None else end
    col_values = df[col_name].values
    i_start = np.searchsorted(col_values, start, side='left')
    i_end = np.searchsorted(col_values, end, side='right')
    return df.iloc[i_start:i_end]

def add_coor_preferred_angle_to_lm_HD(lm_HD):
    lm_HD['fish'] = lm_HD['fish'].astype(int)
    lm_HD['cell'] = lm_HD['cell'].astype(int)

    fish_ids = lm_HD['fish'].unique()

    for fish_count, fishi in enumerate(fish_ids):

        exp = LotrExperiment(DATA_FOLDERS[fishi])
        coords = exp.morphed_coords_um[exp.hdn_indexes]
        preferred_angles = exp.rpc_angles

        data_fish = lm_HD[lm_HD['fish'] == fishi].copy()

        cell_ids = data_fish['cell'].values
        lm_HD.loc[lm_HD['fish'] == fishi,'z'] = coords[cell_ids, 0]
        lm_HD.loc[lm_HD['fish'] == fishi,'x'] = coords[cell_ids, 1]
        lm_HD.loc[lm_HD['fish'] == fishi,'y'] = coords[cell_ids, 2]
        lm_HD.loc[lm_HD['fish'] == fishi,'preferred_angle'] = preferred_angles[cell_ids]
        
    return lm_HD

def add_coor_to_lm_noHD(lm_noHD):
    lm_noHD['fish'] = lm_noHD['fish'].astype(int)
    lm_noHD['cell'] = lm_noHD['cell'].astype(int)

    fish_ids = lm_noHD['fish'].unique()

    for fish_count, fishi in enumerate(fish_ids):

        exp = LotrExperiment(DATA_FOLDERS[fishi])
        coords = np.delete(exp.morphed_coords_um, exp.hdn_indexes, axis=0)

        data_fish = lm_noHD[lm_noHD['fish'] == fishi].copy()

        cell_ids = data_fish['cell'].values
        lm_noHD.loc[lm_noHD['fish'] == fishi,'z'] = coords[cell_ids, 0]
        lm_noHD.loc[lm_noHD['fish'] == fishi,'x'] = coords[cell_ids, 1]
        lm_noHD.loc[lm_noHD['fish'] == fishi,'y'] = coords[cell_ids, 2]
        
    return lm_noHD

def add_speed_modu_2_lm_df(lm_df):
    
    lm_df['speed_modu'] = ['None'] * len(lm_df)
    lm_df.loc[    (lm_df['beta1_p_corrected'] < 0.05) & (lm_df['beta1'] > 0)   , 'speed_modu'] = 'Increase'
    lm_df.loc[    (lm_df['beta1_p_corrected'] < 0.05) & (lm_df['beta1'] < 0)   , 'speed_modu'] = 'Decrease'
    
    return lm_df