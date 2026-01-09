from dataclasses import dataclass, field
from typing import Dict, Any, Callable, List, List, Optional

import numpy as np
from sklearn.model_selection import ParameterGrid

from HD_utils.network import net_diff_equa_f_in, steady_inputb_withb0, steady_inputb_withb0_3r
from HD_utils.comput_property import cal_linear_range


@dataclass
class SimulationConfig:
    """Configuration for ring attractor network simulation with grid search"""
    # Storage
    file_pre_name: str
    
    # Search parameters
    search_pars: Dict[str, np.ndarray]
    
    # Network architecture
    ring_num: int
    actfun: Callable
    weight_fun: Callable
    theta_num: int = 50
    dtheta: float = None
    theta_range: np.ndarray = field(default_factory=lambda: np.array([None]))
    tau: float = 20 # ms
    net_diff_equa: Callable = net_diff_equa_f_in 
    b0: float = 1
    phi: float = np.pi * 1/9
    alpha: float = np.pi * -1/9
    bE: float = 5
    bI: float = 0
    
    # Input parameters
    inputs: np.ndarray = \
        field(default_factory=lambda: np.array([-1, -0.6, -0.3, -0.1, 0, 0.1, 0.3, 0.6, 1])) # = k omega
        
    # Maximum velocity search configuration
    # See function determine_refract_v and 
    # manuscripts_supplementary for details
    maxV_thres: float = 0.1
    v_precision: float = 0.01
    vvr_thres: float = 0.99

    ini_b_stop_count: int = 5
    base_v: float = 0.2
    base_v_tol: float = 0.05
    search_mult: int = 3
    
    def __post_init__(self):
        if self.dtheta is None:
            self.dtheta = (2 * np.pi) / self.theta_num
        if (len(self.theta_range) != self.theta_num):
            self.theta_range = np.arange(-np.pi + self.dtheta/2, np.pi, self.dtheta)
        
        self.par_num = len(self.search_pars)
        self.par_names = list(self.search_pars.keys())
        self.search_num = len(ParameterGrid(self.search_pars))
        self.zeroid = np.where(self.inputs == 0)[0][0]
        if self.ring_num in (1,2):
            self.bs = self.b0
        elif self.ring_num == 3:
            self.bs = [self.bE, self.bI]
    
    def get_barray(self, ratiov=0) -> np.ndarray:
        '''get the inputs to neurons'''
        if self.ring_num == 1:
            return steady_inputb_withb0(self.b0, 0, self.theta_num)[:self.theta_num]
        elif self.ring_num == 2:
            return steady_inputb_withb0(self.b0, ratiov, self.theta_num)
        elif self.ring_num == 3:
            return steady_inputb_withb0_3r(self.bE, self.bI, ratiov, self.theta_num)

    def get_weight(self, par_list: List, ratiov: float = 0) -> np.ndarray:
        '''
        Because the interface of weight functions are different, and does not support config class as input,
        this function is used to map the config parameters to the weight function parameters
        '''

        if self.ring_num == 1:
            weight = self.weight_fun(*par_list, self.theta_range, ratiov)
        elif self.ring_num == 2:
            weight = self.weight_fun(*par_list, self.phi, self.theta_num, self.theta_range)
        elif self.ring_num == 3:
            weight = self.weight_fun(*par_list, self.alpha, self.theta_range)

        return weight
    
    def init_activity(self) -> np.ndarray:
        '''initialize the activity for the network'''
        # For the case where theta_range is different among rings
        # i.e. unequal theta density
        if isinstance(self.theta_range, list):
            theta_range = np.concatenate(self.theta_range)
        else:
            theta_range = self.theta_range
        
        # A small bump in the center
        bump = (np.cos(theta_range) + 1) * 1e-8
        s = bump + 1e-8 # avoid zero division
        s = np.concatenate([s]*self.ring_num)
        return s


@dataclass
class GridSearchResultStationary:
    '''Results from grid search in the stationary case'''
    eval: np.ndarray
    eval_des: np.ndarray
    activity: np.ndarray
    par: np.ndarray
    time: np.ndarray
    
    def __post_init__(self) -> None:
        self.valid_id = np.where(self.eval == 'valid')[0]
        self.valid_num = len(self.valid_id)
        
    def get_final_activity(self, neti) -> np.ndarray:
        '''
        Get the final activity of a specific network, 
        concatenated for all rings
        '''
        activity = self.activity[neti]
        if activity.ndim == 2:  # one ring case
            return activity[:, -1]
        else:  # two ring or three ring case
            return np.concatenate([activity[i, :, -1] for i in range(activity.shape[0])])

    def print_valid_number(self) -> None:
        print(f"Number of valid stationary networks: {self.valid_num}")


@dataclass
class GridSearchResultMoving:
    '''Results from grid search in the moving case'''
    velocity: np.ndarray
    eval: np.ndarray
    correlation: np.ndarray
    activity: np.ndarray
    time: np.ndarray
    eval_sum: np.ndarray

    def __post_init__(self) -> None:
        self.stable_id_rough = np.where(np.isin(self.eval_sum, ('linear moving', 'mid-linear moving', 'nonlinear moving')))[0]
        self.half_linear_id_rough = np.where(self.eval_sum == 'mid-linear moving')[0]
        self.linear_id_rough = np.where(self.eval_sum == 'linear moving')[0]

    def print_number_of_each_type(self, net_sta: GridSearchResultStationary) -> None:
        '''Deprecated, use linear range calculation instead'''
        values = [
            net_sta.valid_num,
            len(self.stable_id_rough),
            len(self.half_linear_id_rough),
            len(self.linear_id_rough),
        ]
        colwidth = 18
        print(f"{'Valid stationary':<{colwidth}} {'Stable moving':<{colwidth}} {'Partly linear':<{colwidth}} {'Fully linear':<{colwidth}}")
        print(f"{values[0]:<{colwidth}} {values[1]:<{colwidth}} {values[2]:<{colwidth}} {values[3]:<{colwidth}}")
        
    def calculate_linear_range(self, net_stationary: GridSearchResultStationary, config: SimulationConfig) -> None:
        '''Calculate the linear range and stable range'''
        self.stable_range, self.stable_range_id, \
        self.linear_range, self.linear_range_id = \
            cal_linear_range(self.eval, self.velocity, config.inputs, net_stationary.valid_id)

        self.part_linear_id = np.where( self.linear_range[:,1] > 0.1 )[0]
        self.linear_id = np.where( self.linear_range[:,1] == 1 )[0]
        self.stable_id = np.where( self.stable_range[:,1] == 1 )[0]


