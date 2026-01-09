'''
Grid search configurations for the ring attractor network simulations.
'''
from HD_utils import dataclass as dc
import HD_utils.network as net
from dataclasses import replace
import numpy as np


# Define configurations for network with 
# different activation and weight functions
configs = [None] * 20

# Synaptic modulation network with 1 ring
configs[0] = dc.SimulationConfig(
    ring_num=1,
    actfun=net.max0x,
    weight_fun=net.asym_vw,
    search_pars={'JI': np.linspace(-100,0,26),
                'JE': np.linspace(0,100,26)},
    file_pre_name='61_3_1',) # legacy name

configs[1] = replace(configs[0],
    actfun=net.tanh1,
    file_pre_name='61_1_6',) # legacy name

configs[2] = replace(configs[0],
    weight_fun=net.asym_vw_vonMises,
    search_pars = {'JI': np.linspace(-100,0,26), 
                'JE': np.linspace(0,100,26), 
                'kappa': np.logspace(-0.6,0.6,6)},
    file_pre_name='61_3_3',) # legacy name

configs[3] = replace(configs[2],
    actfun=net.tanh1,
    file_pre_name='61_2_5',) # legacy name


# Shifter-ring network with two rings
# same weight across rings
configs[4] = dc.SimulationConfig(
    ring_num=2,
    actfun=net.max0x,
    weight_fun=net.cos_weight,
    search_pars={'JI': np.linspace(-100,0,26),
                'JE': np.linspace(0,100,26)},
    file_pre_name='62_3_1',) # legacy name

configs[5] = replace(configs[4],
    actfun=net.tanh1,
    file_pre_name='62_3_2',) # legacy name

configs[6] = replace(configs[4],
    weight_fun=net.vonmises_weight_s,
    search_pars = {'JI': np.linspace(-100,0,26), 
                   'JE': np.linspace(0,100,26), 
                   'kappa': np.logspace(-0.6,0.6,6)},
    file_pre_name='62_4_1',) # legacy name

configs[7] = replace(configs[6],
    actfun=net.tanh1,
    file_pre_name='62_4_2',) # legacy name

# different weight across rings
configs[8] = replace(configs[4],
    weight_fun=net.cos_weight3,
    search_pars = {'JI': np.linspace(-100,0,11), 
                   'JE': np.linspace(0,100,11), 
                   'K0': np.linspace(-100,80,10)},
    file_pre_name='62_1_1',) # legacy name

configs[9] = replace(configs[8],
    actfun=net.tanh1,
    file_pre_name='62_1_2',) # legacy name

configs[10] = replace(configs[8],
    weight_fun=net.vonmises_weight3,
    search_pars = {'JI': np.linspace(-100,0,11), 
                   'JE': np.linspace(0,100,11), 
                   'K0': np.linspace(-100,80,10), 
                   'kappa': np.logspace(-0.6,0.6,6)},
    file_pre_name='62_2_1',) # legacy name

configs[11] = replace(configs[10],
    actfun=net.tanh1,
    file_pre_name='62_2_2',) # legacy name


# Shifter-ring network with three rings
# same weight across shifter rings
configs[12] = dc.SimulationConfig(
    ring_num=3,
    actfun=net.max0x,
    weight_fun=net.cos_weight_3r_Icos_s_IIs,
    search_pars={
        'CI': np.linspace(10,100,10), 
        'CE': np.linspace(10,100,10), 
        'LI': np.linspace(0,100,11)
    },
    file_pre_name='63_1_1'
)

configs[13] = replace(configs[12],
    actfun=net.tanh1,
    file_pre_name='63_1_2'
)

configs[14] = replace(configs[12],
    weight_fun=net.vonMises_weight_3r_v2_II,
    search_pars = {'CI': np.linspace(10,100,10), 
                   'CE': np.linspace(10,100,10), 
                   'LI': np.linspace(0,100,11), 
                   'kappa': np.logspace(-0.6, 0.6, 6)
                   },
    file_pre_name = '63_2_1'
)

configs[15] = replace(configs[14],
    actfun=net.tanh1,
    file_pre_name='63_2_2'
)

# different weight across shifter rings
configs[16] = replace(configs[12],
    weight_fun=net.cos_weight_3r_Icos_s,
    search_pars = {
        'CI': np.linspace(2,20,10), 
        'CE': np.linspace(2,20,10), 
        'LI': np.linspace(0,5,11)
                   },
    file_pre_name = '63_3_1_copy'
)

configs[17] = replace(configs[13],
    weight_fun=net.cos_weight_3r_Icos_s,
    file_pre_name='63_3_2'
)

configs[18] = replace(configs[14],
    weight_fun=net.vonMises_weight_3r_v2,
    file_pre_name='63_4_1'
)

configs[19] = replace(configs[15],
    weight_fun=net.vonMises_weight_3r_v2,
    file_pre_name='63_4_2'
)
