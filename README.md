# Overview
Analyze HD cells from zebrafish and Drosophila.  
Simulation of head direction (HD) system using ring attractor models.

Figures in src/folder4 needs results in folders 1, 2, 3.

The code accompanying the paper:  
A multi-ring shifter network computes head direction in zebrafish  
Siyuan Mei, Hagar Lavian, You Kure Wu, Martin Stemmler, Ruben Portugues, Andreas V.M. Herz  
bioRxiv 2025.12.29.696831; doi: https://doi.org/10.64898/2025.12.29.696831  


# Set the environment
conda create -n HD python==3.8.8
conda activate HD

cd src/utilities/main
pip install -e .

conda install numpy pandas scipy scikit-learn numba ipympl tqdm flammkuchen matplotlib seaborn colorspacious svgpath2mpl pooch ipynbname bg-atlasapi scikit-image trimesh statsmodels jupyterlab pingouin

pip install pingouin  
Note that conda will downgrade pandas when install pingouin, which causes errors when running certain codes. So please install pingouin via pip.

cd ../zebrafish
pip install -e .

## 3D plots
3D plots need a higher version of matplotlib and python, please use another environment to plot them.

conda create -n plot python matplotlib numpy jupyterlab scikit-learn tqdm

conda activate plot

cd src/utilities/main

pip install -e .


# Data
Data is publicly available from previous publications.  

Drosophila:   
https://github.com/DanTurner-Evans/AngularVelocityData/tree/main/data  
Only the "PEN1data.pkl" file is used. Put it to data/Drosophila/  

Zebrafish:   
https://zenodo.org/records/7715850  
Only the lightsheet data is used. Put it folders within the lightsheet folder to data/zebrafish/published/lightsheet/  


# Acknowledgements
utilities/zebrafish is copied from https://github.com/portugueslab/lotr/tree/v1.0.2  
It is used to read data and perform preprocessing.  


# Author
Siyuan Mei (mei@bio.lmu.de)  
2026-1-9
