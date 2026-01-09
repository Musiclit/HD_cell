'''
Functions dealing with color during ploting, scarcely used.

Siyuan Mei (mei@bio.lmu.de)
2024
'''
import numpy as np
from matplotlib import colors


def colorFader(c1,c2,mix=0.0): 
    '''gradient (linear interpolate) from color c1 to c2'''
    c1=np.array(colors.to_rgb(c1))
    c2=np.array(colors.to_rgb(c2))
    return colors.to_hex((1-mix)*c1 + mix*c2)


def colorFader2(c1,c2,c3,mix=0.0):
    '''gradient (linear interpolate) from color c1 to c2 to c3'''
    c1=np.array(colors.to_rgb(c1))
    c2=np.array(colors.to_rgb(c2))
    c3=np.array(colors.to_rgb(c3))
    if mix <= 0.5:
        color = (1-mix*2)*c1 + mix*2*c2
    else:
        color = (2-mix*2)*c2 + (mix*2-1)*c3
    return colors.to_hex(color)