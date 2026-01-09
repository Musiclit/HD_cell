"""Here we define some standard values that will be used in multiple parts
of the analysis. Some additional defaults that are used only in one place might
be defined in the analysis notebooks.
"""

import numpy as np

# Specify remote location for the dataset:
DATASET_URL = "https://zenodo.org/record/7715001/files/"
DATASET_HASH = "md5:c9e48cfbd875dd88af702b639a9f5bb5"
# Minimum bias value that defines a turn.
# This was defined based on trimodal curve fit over all bouts in the dataset
TURN_BIAS = 0.239

# Tau for the computation of the regressor. Longer than expected GCaMP tau
# as from trace observation motor-associated transients seem to decay slower
REGRESSOR_TAU_S = 5

# Window in seconds for smoothing traces when calculating the PCA/phase
TRACES_SMOOTH_S = 5

# Standard sampling frequency to which experiments will be resampled if needed
# (most experiments have this FN already):
DEFAULT_FN = 5

# window before and after swims for cropping:
PRE_BOUT_WND_S, POST_BOUT_WND_S = 10, 25

# window in which we will calculate the swm-triggered phase change
WND_DELTA_PHASE_S = np.array([15, 20])

# Pad time at beginning and end of experiment in seconds when calculating PCA:
PCA_TIME_PAD_S = 150

# Anatomical orientation of source data
ANATOMICAL_ORIENT_SOURCE = "ilp"

# Anatomical orientation of figures:
ANATOMICAL_ORIENT_FIGURES = "ipl"

# lightsheet microscope pixel size in um, from calibration:
LIGHTSHEET_CAMERA_RES_XY = (0.6, 0.6)

# Matrix from centered ls coordinates to IPN reference:
TO_IPNREF_MTX = np.array(
    [
        [0.0, 0.0, -1.0, 30.0],
        [-1.0, 0.0, 0.0, 60.0],
        [0.0, -1.0, 0.0, 107.5],
        [0.0, 0.0, 0.0, 0.5],
    ]
)

RESULTS_LOG_FILE = "results_log.txt"

RESULTS_NDIGITS = 3
