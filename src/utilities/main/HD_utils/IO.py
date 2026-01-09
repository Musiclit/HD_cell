'''
To conveniently store and load data and results
'''
from typing import Union, List, Callable, Optional
import pickle
from pathlib import Path

import numpy as np

from HD_utils.dataclass import *


# ==========================================
#           PATH CONFIGURATION
# ==========================================
def find_project_root(marker_files=None, max_depth=8):
    """
    Find project root by looking for .git
    """
    if marker_files is None:
        marker_files = ['.git']
    
    current = Path(__file__).parent.resolve()
    
    for _ in range(max_depth):
        # Check if any marker file exists in current directory
        if any((current / marker).exists() for marker in marker_files):
            return current
        
        # Go up one level
        if current.parent == current:  # Reached filesystem root
            break
        current = current.parent
    
    return False  # Not found within max_depth


SRC_PATH = find_project_root()
FISH_DATA_PATH = SRC_PATH.parent / 'data' / 'zebrafish' / 'published' / 'lightsheet'
FISH_RESULT_PATH = SRC_PATH.parent / 'results' / 'zebrafish' / 'reanalyzed'

FLY_DATA_PATH = SRC_PATH.parent / 'data' / 'Drosophila'
FLY_RESULT_PATH = SRC_PATH.parent / 'results' / 'Drosophila'

SIM_RESULT_PATH = SRC_PATH.parent / 'results' / 'simulations'

STAT_PATH = SRC_PATH.parent / 'results' / 'simulations' / 'network_stats'
TABLE_PATH = SRC_PATH.parent.parent / 'Manuscript_Tables'
FIGURE_PATH = SRC_PATH.parent.parent / 'Manuscript_Figures'


# ==========================================
#           FILE I/O FUNCTIONS
# ==========================================
file_name = {'max0x': 'max0x', 'vonmises_weight': 'VonMises', 'hyperb': 'tanh', 'cos_weight': 'cos',
                'cos_weight2': 'cos2', 'piecewise_linear': 'piecewise_linear', 'box_weight': 'box', 'hyperb3':'hyperb_custom',
                'sym_w': 'cosS1', 'asym_vw': 'cosS1', 'asym_vw_vonMises': 'vonMisesS1', 'asym_vw_wde': 'cosS1_wde', 'sigmoid': 'sigmoid', 
                'asym_vw_wde_general': 'cosS1_wde_general', 'sigmoid2': 'sigmoid2', 'cos_weight_3r': 'cos', 'vonMises_weight_3r': 'VonMises', 'cos_weight_3r_Icos': 'cos',
                'cos_weight_3r_Icos_decrease': 'cos', 'cos_weight_3r_Icos_half_lateral': 'cos', 'max0x4': '4max0x', 'softplus': 'softplus',
                'cos_weight3': 'cos', 'tanh1': 'tanh','vonmises_weight3': 'VonMises', 'cos_weight_3r_Icos_s_IIs': 'cos',
                'cos_weight_3r_Icos_s': 'cos', 'vonMises_weight_3r_v2': 'vonMises', 'vonMises_weight_3r_v2_II': 'vonMises', 'vonmises_weight_s': 'vonMises',
                'cos_weight_3r_allI':'cos', 'cos_weight_3r_allI_more': 'cos', 'cos_weight_2r_r': 'cos', 'vonmises_weight_s_2rr': 'vonMises',
                'cos_weight_2rr_3p': 'cos', 'vonmises_weight_2rr_3p': 'vonMises', 'vonmises_weight_2i1r_3p_cut_excit': 'vonMises',
                'asym_vw_wincrease': 'cos', 'vonmises_weight_2i1r_3p_cut_excit_no_self': 'vonMises', 'vonmises_weight_2i1r_2': 'vonMises',
                'vonmises_weight_2i1r_unequal_theta': 'vonMises', 'vonmises_weight_drosophila': 'drosophila', 'vonmises_weight_2i1r_unequal_theta_v2': 'vonMises',
                'vonmises_weight_2i1r_unequal_theta_acv_scale': 'vonMises'}

# These are used in the theoretical prediction test notebook
waf_df_names = file_name.copy()
waf_df_names['vonmises_weight'] = 'vonMises'
waf_df_names['cos_weight2'] = 'cos'
waf_df_names['sym_w'] = 'cos'
waf_df_names['asym_vw'] = 'cos'
waf_df_names['asym_vw_vonMises'] = 'vonMises'
waf_df_names['asym_vw_wde'] = 'cos'
waf_df_names['asym_vw_wde_general'] = 'cos'
waf_df_names['vonMises_weight_3r'] = 'vonMises'
waf_df_names['vonmises_weight3'] = 'vonMises'


def save_grid_search_results(type: str, 
                             result: Union[GridSearchResultStationary, GridSearchResultMoving], 
                             config: SimulationConfig):
    '''
    Save grid search results to pickle files in folder results/simulations/ with "grid_search" suffix.
    Use config dataclass and grid search result class for clean variables
    
    wrapper function of store_pickle
    '''
    if type == 'stationary':
        arraylist = [result.eval, result.eval_des, result.activity, result.par, result.time]
        arraynames = ['evals', 'eval_des', 'acvs', 'pars', 'ts']
        store_pickle(arraylist, arraynames, config.weight_fun, config.actfun, config.file_pre_name)
    elif type == 'moving':
        arraylist = [result.velocity, result.eval, result.correlation, result.activity, result.time, result.eval_sum]
        arraynames = ['moving_slope', 'moving_eval', 'moving_eval_des', 'moving_acvs', 'moving_ts', 'moving_eval_sum']
        store_pickle(arraylist, arraynames, config.weight_fun, config.actfun, config.file_pre_name)
    else:
        raise ValueError("Unknown type")
    
    
def load_gridsearch_results(type: str, config: SimulationConfig) -> Union[GridSearchResultStationary, GridSearchResultMoving]:
    '''
    Load grid search results from pickle files in folder results/simulations/ with "grid_search" suffix.
    Use config dataclass and grid search result class for clean variables
    
    wrapper function of load_pickle_v2
    '''
    if type == 'stationary':
        arraynames = ['evals', 'eval_des', 'acvs', 'pars', 'ts']
        result = GridSearchResultStationary(*load_pickle_v2(arraynames, config))
        return result
    elif type == 'moving':
        arraynames = ['moving_slope', 'moving_eval', 'moving_eval_des', 'moving_acvs', 'moving_ts', 'moving_eval_sum']
        result = GridSearchResultMoving(*load_pickle_v2(arraynames, config))
        return result
    else:
        raise ValueError("Unknown type")


def store_pickle(arraylist: List[np.ndarray], 
                 arraynames: List[str], 
                 weight_fun: Callable, 
                 actfun: Callable, 
                 additional_text='') -> None:
    '''
    Store variables into results/simulations/ folder with suffix "grid_search" as pickle files.
    
    Parameters:
        arraylist: list of variables to be stored
        arraynames: list of file names of the stored variables
        weight_fun: weight function included in the file name
        actfun: activation function included in the file name
        additional_text: additional text included in the file name
    '''
    for i, var in enumerate(arraylist):
        wname = weight_fun.__name__
        actname = actfun.__name__
        pathstr = SIM_RESULT_PATH / f"grid_search{additional_text}_{file_name[wname]}_{file_name[actname]}_network_{arraynames[i]}.p"
        with open(pathstr, 'wb') as f:
            pickle.dump( var, f)


def load_pickle_v2(arraynames: List[str], config: SimulationConfig):
    '''
    Load variables from pickle files in folder results/simulations/ with "grid_search" suffix.
    use config dataclass for clean variables
    
    Parameters:
        arraynames: list of file names of the stored variables
        config: configuration dataclass including
            weight_fun: weight function included in the file name
            actfun: activation function included in the file name
            file_pre_name: additional text included in the file name
    '''
    additional_text = config.file_pre_name
    length = len(arraynames)
    variables = [0]*length
    for i in range(length):
        wname = config.weight_fun.__name__
        actname = config.actfun.__name__
        pathstr = SIM_RESULT_PATH / f"grid_search{additional_text}_{file_name[wname]}_{file_name[actname]}_network_{arraynames[i]}.p"
        with open(pathstr, 'rb') as f:
            variables[i] = pickle.load(f)
    return variables


def savefig(fig, figname, weight_fun, actfun, dpi=300, additional_text=''):
    '''
    Save figure with a specific name into folder results/zebrafish/ with suffix "grid_search".
    Parameters:
        fig: figure object
        figname: name of the figure
        weight_fun: weight function included in the file name
        actfun: activation function included in the file name
        dpi: resolution of the saved figure
        additional_text: additional text included in the file name
    '''
    wname = weight_fun.__name__
    actname = actfun.__name__
    pathstr = FISH_RESULT_PATH / f"grid_search{additional_text}_{file_name[wname]}_{file_name[actname]}_parameter_on_{figname}.jpg"
    fig.savefig(pathstr, dpi=dpi)


def store_pickle_s(arraylist, arraynames, additional_text=''):
    '''
    Store variables into results/simulations/ folder as pickle files.
    Not for grid search results.
    
    Parameters:
        arraylist: list of variables to be stored
        arraynames: list of file names of the stored variables
        additional_text: additional text included in the file name
    '''
    for i, var in enumerate(arraylist):
        pathstr = SIM_RESULT_PATH / f"{additional_text}_{arraynames[i]}.p"
        with open(pathstr, 'wb') as f:
            pickle.dump(var, f)


def load_pickle_s(arraynames, additional_text=''):
    '''
    Load variables from pickle files in folder results/simulations/ .
    Not for grid search results.
    
    Parameters:
        arraynames: list of file names of the stored variables
        additional_text: additional text included in the file name
    '''
    length = len(arraynames)
    variables = [0]*length
    for i in range(length):
        pathstr = SIM_RESULT_PATH / f"{additional_text}_{arraynames[i]}.p"
        with open(pathstr, 'rb') as f:
            variables[i] = pickle.load(f)
    return variables


def save_vars(folder_path, var_list, var_name_list, prefix=''):
    '''
    Save variables into a specific folder with a specific suffix as pickle files.
    
    Parameters:
        var_list: list of variables to be stored
        var_name_list: list of file names of the stored variables
        suffix: suffix included in the file name
        folder_path: path to the folder where files will be saved
    '''
    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)
    
    for i, var in enumerate(var_list):
        pathstr = folder_path / f"{prefix}{var_name_list[i]}.p"
        with open(pathstr, 'wb') as f:
            pickle.dump(var, f)
            
            
def load_vars(folder_path, var_name_list, prefix=''):
    '''
    Load variables from pickle files in a specific folder with a specific suffix.
    
    Parameters:
        var_name_list: list of file names of the stored variables
        suffix: suffix included in the file name
        folder_path: path to the folder where files are stored
    '''
    folder_path = Path(folder_path)
    length = len(var_name_list)
    variables = [0]*length
    
    for i in range(length):
        pathstr = folder_path / f"{prefix}{var_name_list[i]}.p"
        with open(pathstr, 'rb') as f:
            variables[i] = pickle.load(f)
            
    if length == 1:
        return variables[0]
    return variables


def load_pickle(arraynames, weight_fun, actfun, additional_text=''):
    '''
    deprecated
    Load variables from pickle files in folder results/simulations/ with "grid_search" suffix.
    
    Parameters:
        arraynames: list of file names of the stored variables
        weight_fun: weight function included in the file name
        actfun: activation function included in the file name
        additional_text: additional text included in the file name
    '''
    length = len(arraynames)
    variables = [0]*length
    for i in range(length):
        wname = weight_fun.__name__
        actname = actfun.__name__
        pathstr = SIM_RESULT_PATH / f"grid_search{additional_text}_{file_name[wname]}_{file_name[actname]}_network_{arraynames[i]}.p"
        with open(pathstr, 'rb') as f:
            variables[i] = pickle.load(f)
    return variables