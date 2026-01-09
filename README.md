# Overview
Analyze HD cells from zebrafish and Drosophila.
Simulation of head direction (HD) system using ring attractor models.

Figures in src/folder4 needs results in folders 1, 2, 3.


# Set the environment
conda create -n HD python==3.8.8

cd src/utilities/main
pip install -e .

cd ../zebrafish
pip install -e .

pip install scikit-learn==1.1.3 %% turn to an older version of scikit-learn to avoid errors
pip install statsmodels

Some packages may missing, please install them as needed.

Some 3D plots need a higher version of matplotlib and python, please use another environment to plot them.


# Data
Data is publicly available from previous publications.

Drosophila: 
https://github.com/DanTurner-Evans/AngularVelocityData/tree/main/data
Only the "PEN1data.pkl" file is used. Put it to data/Drosophila/

Zebrafish: 
https://zenodo.org/records/7715850
Only the lightsheet data is used. Put it to data/zebrafish/published/lightsheet/


# Acknowledgements
utilities/zebrafish is copied from https://github.com/portugueslab/lotr/tree/v1.0.2
It is used to read data and perform preprocessing.


# Author
Siyuan Mei (mei@bio.lmu.de)
2026-1-9
