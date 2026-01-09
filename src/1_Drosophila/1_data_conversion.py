'''
Convert fly's data to a more convenient format.

The fly's data "PEN1data.pkl" is at https://github.com/DanTurner-Evans/AngularVelocityData/tree/main/data

Put "PEN1data.pkl" under the folder "data/Drosophila/".

Siyuan Mei
2026-01-03
'''
import os

import scipy.io
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

from HD_utils import circular_stats as cstat
from HD_utils.IO import *

if __name__ == '__main__':
    ######## Convert to a whole df ########

    input = open(FLY_DATA_PATH / 'PEN1data.pkl', 'rb')
    allDF = pickle.load(input)
    input.close()

    gain_conds = ['Dark', '1x', '2x']

    fly_num_PB = 14
    cell_type_names = ['PEN_L', 'PEN_R', 'EPG_L', 'EPG_R'] # cell type names for PB

    fly_num_EB = 19
    cell_type_EB_names = ['PEN', 'EPG_L', 'EPG_R']


    dtheta_PB = np.pi * 2 / 8
    theta_range_PB_L = np.arange(-np.pi + dtheta_PB / 2, np.pi, dtheta_PB)
    theta_range_PB_R = np.roll(theta_range_PB_L, 1)
    theta_range_PB_all = np.concatenate((theta_range_PB_L, theta_range_PB_R, theta_range_PB_L, theta_range_PB_R))
    theta_range_PB_list = [theta_range_PB_L, theta_range_PB_R, theta_range_PB_L, theta_range_PB_R]

    df = pd.DataFrame() # columns=['fly', 'gain', 'trial', 't', 'cell_id', 'cell', 'activity', 'cell_type', 'preferred_angle', 'angVel', 'HD', 'netPhase_sp', 'netPhase']
    netPhase_sp = [0,0,0,0]
    for flyi, fly in enumerate(tqdm(allDF['PEN1-PB'].keys())):
        # determin the color of PEN and EPG
        if (allDF['PEN1-PB'][fly]['colors'] == ['6fx60D05', 'jRGC1ax37F06']):
            PEN_color = 'Red'
            EPG_color = 'Green'
        else:
            PEN_color = 'Green'
            EPG_color = 'Red'
        PEN_L_roi = [f'roi {i} {PEN_color}' for i in range(8)]
        PEN_R_roi = [f'roi {i} {PEN_color}' for i in range(10,18)]
        EPG_L_roi = [f'roi {i} {EPG_color}' for i in range(1, 9)]
        EPG_R_roi = [f'roi {i} {EPG_color}' for i in range(9, 17)]
        cell_types = [PEN_L_roi, PEN_R_roi, EPG_L_roi, EPG_R_roi]
        cell_types_all = PEN_L_roi + PEN_R_roi + EPG_L_roi + EPG_R_roi
        data_temp = allDF['PEN1-PB'][fly]
        # Calculate the Angular Velocity
        for gaini, gain in enumerate(gain_conds):
            for triali, trial in enumerate(data_temp[gain].keys()):
                data_temp2 = data_temp[gain][trial].iloc[:-1].copy() # The final row't is zero, so we'll drop it

                ang_expan = cstat.rerange_expand(data_temp2['angle']) # unit: rad
                fs = 1 / np.median(np.diff(data_temp2['t']))
                angVel = np.diff(ang_expan) * fs # unit rad/s

                # Calculate the network phase and network phase specific to each cell type
                df_phase = np.array(data_temp2[cell_types_all])
                netPhase = cstat.mean(theta_range_PB_all.reshape(-1,1).repeat( len(df_phase), axis=1 ), df_phase.T, axis=0)
                angphase_expan = cstat.rerange_expand(netPhase) # unit: rad
                angVel_phase = np.diff(angphase_expan) * fs # unit rad/s
                
                for cell_typei, cell_type in enumerate(cell_types):
                    df_phase_sp = np.array(data_temp2[cell_type])
                    netPhase_sp[cell_typei] = cstat.mean(theta_range_PB_list[cell_typei].reshape(-1,1).repeat( len(df_phase_sp), axis=1 ), df_phase_sp.T, axis=0)
                    # Create dataframe
                    for celli, cell in enumerate(cell_type):
                        fly_id_col = [flyi] * len(data_temp2)
                        fly_col = [fly] * len(data_temp2)
                        gain_col = [gain] * len(data_temp2)
                        trial_col = [trial] * len(data_temp2)
                        t_col = data_temp2['t']
                        cell_id_col = [celli + cell_typei * len(cell_type)] * len(data_temp2)
                        cell_col = [cell] * len(data_temp2)
                        activity_col = data_temp2[cell]
                        cell_type_col = [cell_type_names[cell_typei]] * len(data_temp2)
                        preferred_angle_col = [theta_range_PB_list[cell_typei][celli]] * len(data_temp2)
                        angVel_col = np.append(0, angVel)
                        angVel_phase_col = np.append(0, angVel_phase)
                        angle_col = data_temp2['angle']
                        network_phase_sp_col = netPhase_sp[cell_typei]
                        network_phase_col = netPhase
                        df_temp = pd.DataFrame({'fly_id': fly_id_col ,'fly': fly_col, 'gain': gain_col, 'trial': trial_col, 't': t_col, 'roi_id': cell_id_col, 
                                                'roi': cell_col, 'activity': activity_col, 'cell_type': cell_type_col, 
                                                'preferred_angle': preferred_angle_col, 'angVel': angVel_col, 'HD': angle_col, 
                                                'netPhase_sp': network_phase_sp_col, 'netPhase': network_phase_col, 'angVel_phase': angVel_phase_col})
                        df = pd.concat([df, df_temp], ignore_index=True)
    df.to_csv(FLY_RESULT_PATH / 'data_PB_roi.csv', index=False)