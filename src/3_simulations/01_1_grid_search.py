'''
Perform grid search for ring attractor network with 
max0x or tanh+1 activation function AND
cosine or von Mises weight function.

Siyuan Mei (mei@bio.lmu.de)
2025-9-14
'''
from dataclasses import replace

import HD_utils.dataclass as dc
import HD_utils.network as net
from HD_utils.adap_sim_move import grid_search_moving
from HD_utils.adap_sim_stable import grid_search_stationary
from HD_utils.gridsearch_configs import configs


def main():
    # Run simulations for all configurations
    for i, config in enumerate(configs):
        
        # For Test:
        if i < 14:
            continue
        
        print(f"\n=== Running configuration {i+1}/{len(configs)} ===")
        print(f"Running grid search for configuration: {config.ring_num} rings, {config.actfun.__name__} activation & {config.weight_fun.__name__} weight function.")
        print("==================================\n")
        net_sta = grid_search_stationary(config)
        net_sta.print_valid_number()

        net_move = grid_search_moving(config, net_sta)
        net_move.print_number_of_each_type(net_sta)
        # Tested the first 3, it works well.
        
        
if __name__ == "__main__":
    main()