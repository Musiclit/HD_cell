import os
import numpy as np

from HD_utils.IO import *

def list_second_order_subfolders(directory):
    """
    List all second-order subfolders using pathlib (more modern approach).
    
    Args:
        directory: Path to the parent directory (can be str or Path object)
        
    Returns:
        list: List of Path objects for second-order subfolders
    """
    directory = Path(directory)
    data_folders = []
    
    # Get all first-level subdirectories
    for first_level in directory.iterdir():
        if first_level.is_dir():
            # Get all second-level subdirectories
            for second_level in first_level.iterdir():
                if second_level.is_dir():
                    data_folders.append(second_level)
    
    return data_folders

DATA_FOLDERS = list_second_order_subfolders(FISH_DATA_PATH)

FISH_NUM_ALL = 31
FISH_EXCLUDE = np.array([12, 20, 26, 27]).astype(int)
FISH_NUM = 27
FISH_IDS = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 14, 15, 16, 17,
       18, 19, 21, 22, 23, 24, 25, 28, 29, 30]).astype(int)

FS = 5 # sampling frequency, it is 3 for fish 20, but that fish is excluded

GREENT = np.array((135,202,90))/255
BLUET = np.array((16,128,189))/255
REDT = np.array((255,151,151))/255