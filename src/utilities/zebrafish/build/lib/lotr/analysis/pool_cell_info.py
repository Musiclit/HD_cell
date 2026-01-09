import numpy as np
import pandas as pd

from lotr import dataset_folders
from lotr.experiment_class import LotrExperiment


def get_pooled_cell_info():
    data_df = []
    for path in dataset_folders:
        exp = LotrExperiment(path)
        coords = exp.coords_um
        cent_coords = exp.ipnref_coords_um

        data_dict = {f"c{i}": coords[:, i] for i in range(3)}
        data_dict.update({f"centered{i}": cent_coords[:, i] for i in range(3)})
        data_dict["fid"] = path.name
        data_dict["hdn"] = np.full(exp.n_rois, False)
        data_dict["hdn"][exp.hdn_indexes] = True
        # data_dict["new"] = path in new_dataset_folders

        data_df.append(pd.DataFrame(data_dict))

    return pd.concat(data_df, ignore_index=1)
