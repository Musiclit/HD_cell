'''
plotting functions
'''
import numpy as np
import matplotlib.pyplot as plt
from HD_utils.comput_property import *
# from HD_utils.HD_functions import *
from HD_utils.comput_property import *
from matplotlib import colors
from HD_utils.IO import *
import pandas as pd
from HD_utils import network
import matplotlib

from HD_utils.defaults import *

color_dict = {'bad shape': 'black', 'cannot move': 'grey', 'equal flat': 'olive', 'unequal flat': 'olivedrab', 'explode': 'red', 'linear': 'blue', 'mid linear': 'cyan',
              'unlinear': 'green', 'unstable': 'orange', 'valid': 'blue', 'flat': 'olive', 'singularity': 'grey', 'unstable bumpy flat': 'forestgreen', 'bumpy flat': 'darkgreen',
              'linear moving': 'blue', 'mid-linear moving': 'deepskyblue', 'nonlinear moving': 'aquamarine', 'valid stationary shape': 'lawngreen',
              'shape&vel unstable': 'purple', 'shape unstable': 'funsia', 'vel unstable': 'deeppink',
              'Exploding': 'red', 'Flat': 'olive', 'Linearly integrating': 'blue', 'Unstable': 'orange', 'drift': 'purple'}

# IMAGE_PATH = 'C:/Users/15824/OneDrive/R-Direction_Cell/Presentation&Summary/BCCN/'
IMAGE_PATH = 'D:/OneDrive/R-Direction_Cell/Presentation&Summary/BCCN/'

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'): # make the x-axis label as fraction of pi
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))

def shift_2center(acv):
    theta_num = len(acv)
    shiftv = np.argmax(acv) - theta_num // 2
    acv = np.roll(acv, -shiftv)
    return acv

def oned2colormesh_s2ring(array1d, search_pars, network_pars, xlabel='J0', ylable='J1'):
    max_v_plot = np.zeros(( len(search_pars[ylable]), len(search_pars[xlabel])))
    for j, J1 in enumerate(search_pars[ylable]):
        for k, J0 in enumerate(search_pars[xlabel]):
            max_v_plot[j,k] = array1d[np.where((network_pars[:,0] == J0) & (network_pars[:,1] == J1))[0]]
    max_v_plot[max_v_plot == 0] = np.nan
    return max_v_plot

def oned2colormesh_general(array1d, search_pars, network_pars, parnames=None, zero2nan=False):
    if parnames is None:
        parnames = list(search_pars.keys())
    parnum = len(parnames)
    if parnum == 2:
        propplot = oned2colormesh(array1d, search_pars, network_pars, par1name=parnames[0], par2name=parnames[1], zero2nan=zero2nan)
    elif parnum == 3:
        propplot = oned2colormesh_3par(array1d, search_pars, network_pars, parnames=parnames, zero2nan=zero2nan)
    elif parnum == 4:
        propplot = oned2colormesh_4par(array1d, search_pars, network_pars, parnames=parnames, zero2nan=zero2nan)
    
    return propplot

def oned2colormesh(array1d, search_pars, network_pars, par1name='JI', par2name='JE', zero2nan=True):
    propplot = np.zeros(( len(search_pars[par2name]), len(search_pars[par1name])))
    for j, par1 in enumerate(search_pars[par1name]):
        for k, par2 in enumerate(search_pars[par2name]):
            index = np.where((network_pars[:,0] == par1) & (network_pars[:,1] == par2))[0][0]
            propplot[k,j] = array1d[np.where((network_pars[:,0] == par1) & (network_pars[:,1] == par2))[0]]
    if zero2nan:
        propplot[propplot == 0] = np.nan
    return propplot

def oned2colormesh_3par(array1d, search_pars, network_pars, parnames=['J0','J1','kappa'], zero2nan=True):
    max_v_plot = np.zeros(( len(search_pars[parnames[2]]), len(search_pars[parnames[1]]), len(search_pars[parnames[0]]) ))
    for i, kappa in enumerate(search_pars[parnames[2]]):
        for j, J1 in enumerate(search_pars[parnames[1]]):
            for k, J0 in enumerate(search_pars[parnames[0]]):
                # print(array1d[np.where((network_pars[:,0] == J0) & (network_pars[:,1] == J1) & (network_pars[:,2] == kappa))[0]])
                max_v_plot[i,j,k] = array1d[np.where((network_pars[:,0] == J0) & (network_pars[:,1] == J1) & (network_pars[:,2] == kappa))[0]]
    if zero2nan:
        max_v_plot[max_v_plot == 0] = np.nan
    return max_v_plot

def oned2colormesh_4par(array1d, search_pars, network_pars, parnames=['J0','J1','K0','kappa'], zero2nan=True):
    max_v_plot = np.zeros(( len(search_pars[parnames[3]]), len(search_pars[parnames[2]]), 
                           len(search_pars[parnames[1]]), len(search_pars[parnames[0]]) ))
    for m, k0 in enumerate(search_pars[parnames[2]][::-1]):
        for i, kappa in enumerate(search_pars[parnames[3]]):
            for j, j1 in enumerate(search_pars[parnames[1]]):
                for k, j0 in enumerate(search_pars[parnames[0]]):
                    max_v_plot[i,m,j,k] = array1d[np.where((network_pars[:,0] == j0) & (network_pars[:,1] == j1) &\
                                                            (network_pars[:,2] == k0) & (network_pars[:,3] == kappa))[0]]
    if zero2nan:
        max_v_plot[max_v_plot == 0] = np.nan
    return max_v_plot

def plot_2_par_on_type(network_evals, network_pars, figtitle='', xvar='JI', yvar='JE', figsize=(5,5), dotsize=40, alpha=1):
    '''plot how two parameters affect the network type (network_evals)'''
    xvari = 0
    yvari = 1
    fig = plt.figure(figsize=figsize)
    fig.suptitle(figtitle)
    for i, label in enumerate(np.unique(network_evals)):
        bol = network_evals == label
        xdata = (network_pars[bol,xvari])
        ydata = (network_pars[bol,yvari])
        plt.scatter(xdata, ydata, color=color_dict[label], marker='8', alpha=alpha, label=label, s=dotsize)
        plt.xlabel(xvar)
        plt.ylabel(yvar)
    plt.legend(frameon=True)
    plt.show()

# def plot_2_par_on_type2(network_evals, network_pars, figtitle='', xvar='JI', yvar='JE', figsize=(5,5), dotsize=40):
#     xvari = 0
#     yvari = 1
#     fig = plt.figure(figsize=figsize)
#     fig.suptitle(figtitle)
#     for i, label in enumerate(np.unique(network_evals)):
#         bol = network_evals == label
#         xdata = (network_pars[bol,xvari])
#         ydata = (network_pars[bol,yvari])
#         plt.scatter(xdata, ydata, color=color_dict[label], marker='8', alpha=1, label=label, s=dotsize)
#         plt.xlabel(xvar)
#         plt.ylabel(yvar)
#     plt.legend(frameon=True)
#     plt.show()

def plot_3par_on_type(search_pars, network_evals, network_pars, figtitle='', legend_subplot_loc=0, legend_loc=None, legend_fontsize=None,
                              xvar='JI', yvar='JE', plotvar='kappa', nrow=3, ncol=5, colsize=3, rowsize=3, dotsize=10):
    xvari, yvari, ploti = 0, 1, 2
    fig, axs = plt.subplots(nrow,ncol,figsize=(ncol*colsize,nrow*rowsize))
    fig.suptitle(figtitle)

    if legend_fontsize is None:
        legend_fontsize = 'small'

    for j, ax in enumerate(axs.flatten()):
        if j >= len(search_pars[plotvar]):
            ax.axis('off')
            continue
        bol0 = network_pars[:,ploti] == search_pars[plotvar][j]
        for i, label in enumerate(np.unique(network_evals)):
            bol = (bol0) & (network_evals == label)
            xdata = (network_pars[bol,xvari])
            ydata = (network_pars[bol,yvari])
            ax.scatter(xdata, ydata, color=color_dict[label], marker='8', alpha=1, label=label, s=dotsize)
            ax.set_xlabel(xvar)
            ax.set_ylabel(yvar)
            ax.set_title(f'{plotvar} = {search_pars[plotvar][j]:.2f}')
        if j == legend_subplot_loc:
            if legend_loc is None:
                ax.legend(fontsize=legend_fontsize)
            else:
                ax.legend(fontsize=legend_fontsize, loc=legend_loc)
    plt.tight_layout()
    plt.show()

def plot_4par_on_type(search_pars, network_evals, network_pars, figtitle='', legend_subplot_loc=0, legend_loc=None,
                              xvar='J0', yvar='J1', plotxvar='K0', plotyvar='kappa', 
                              colsize=3, rowsize=3):
    if legend_loc is None:
        legend_loc = 'best'
    xvari, yvari, plotxvari, plotyvari = 0, 1, 2, 3
    ncol = search_pars[plotxvar].size
    nrow = search_pars[plotyvar].size
    fig, axs = plt.subplots(nrow,ncol,figsize=(ncol*colsize,nrow*rowsize))
    fig.suptitle(figtitle)
    for j in range(nrow):
        for k in range(ncol):
            ax = axs[j,k]
            bol0 = (network_pars[:,plotxvari] == search_pars[plotxvar][k]) & (network_pars[:,plotyvari] == search_pars[plotyvar][j])
            for i, label in enumerate(np.unique(network_evals)):
                bol = (bol0) & (network_evals == label)
                xdata = (network_pars[bol,xvari])
                ydata = (network_pars[bol,yvari])
                ax.scatter(xdata, ydata, color=color_dict[label], marker='8', alpha=1, label=label)
                ax.set_xlabel(xvar)
                ax.set_ylabel(yvar)
                ax.set_title(f'{plotxvar} = {search_pars[plotxvar][k]:.2f}')
            if k == 0:
                ax.set_ylabel(f'{plotyvar} = {search_pars[plotyvar][j]:.2f}')
            if k == legend_subplot_loc:
                ax.legend(fontsize='small', loc=legend_loc)
    plt.tight_layout()
    plt.show()

def plot_2par_on_property(prop, search_pars, network_pars, linrange_plot, figsize, lin_rang_thres=0.075, title=None, xlabel='J0', ylabel='J1', cbarlabel=None, cmap='viridis'):
    max_v_plot = oned2colormesh_s2ring(prop, search_pars, network_pars, xlabel, ylabel)
    max_v_plot_cond = max_v_plot.copy()
    max_v_plot_cond[linrange_plot < lin_rang_thres] = np.nan
    data = [max_v_plot, max_v_plot_cond]
    fig, axs = plt.subplots(1,2,figsize=figsize)
    fig.suptitle(title)
    for i in range(2):
        ax = axs[i]
        im = ax.pcolormesh(search_pars[xlabel], search_pars[ylabel], data[i], cmap=cmap)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()
    
def plot_2par_on_property2(prop, search_pars, network_pars, figsize, title=None, xlabel='JI', ylabel='JE', cbarlabel=None, cmap='viridis'):
    propplot = oned2colormesh(prop, search_pars, network_pars)
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title)
    im = ax.pcolormesh(search_pars[xlabel], search_pars[ylabel], propplot, cmap=cmap)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

def plot_pars_on_property(prop, search_pars, parnames, titles='', ncol=5, nrow=None, figmag=2, par_present=None, 
                          vmin=None, vmax=None, target=False, norm=None, reverse_col_row_var=False):
    parnum = len(parnames)
    if par_present is None:
        par_present = parnames

    if parnum == 2:
        fig = plot_2par_on_property3(prop, search_pars, title=titles, xlabel=parnames[0], ylabel=parnames[1], 
                               figsize=(2*figmag,1.5*figmag), vmin=vmin, target=target, norm=norm)
    elif parnum == 3:
        if nrow == None:
            nrow = (len(search_pars[parnames[2]]) - 1) // ncol + 1
        fig = plot_3par_on_prop(search_pars, prop, nrow=nrow, ncol=ncol, parnames=parnames, figtitle=titles, parname_preset=par_present, 
                          figsizemag=figmag, vmin=vmin, target=target, norm=norm)
    elif parnum == 4:
        fig = plot_4par_on_prop(search_pars, prop, parnames=parnames, parname_preset=par_present, figtitle=titles, 
                          figsizemag=figmag, vmin=vmin, vmax=vmax, target=target, norm=norm, reverse_col_row_var=reverse_col_row_var)
        
    return fig

def plot_2par_on_property3(propplot, search_pars, network_pars=None, figsize=(4,3), title=None, xlabel='JI', ylabel='JE', 
                           cbarlabel=None, cmap='viridis', vmin=None, target=False, norm=None):
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_under(color='grey')
    cmap.set_over(color='pink')

    if vmin is None:
        vmin = np.nanmin(propplot)
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title)
    if norm == None:
        im = ax.pcolormesh(search_pars[xlabel], search_pars[ylabel], propplot, cmap=cmap, vmin=vmin)
    else:
        im = ax.pcolormesh(search_pars[xlabel], search_pars[ylabel], propplot, cmap=cmap, norm=norm)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if np.any(target):
        x = target[0]
        y = target[1]
        x_interval = search_pars[xlabel][1] - search_pars[xlabel][0]
        y_interval = search_pars[ylabel][1] - search_pars[ylabel][0]
        ax.add_patch(plt.Rectangle((x - x_interval/2, y - y_interval/2), x_interval, y_interval, fill=False, edgecolor='red', lw=2))
    plt.colorbar(im, ax=ax, extend='min')
    plt.tight_layout()
    plt.show()

    return fig

def plot_3par_on_prop(search_pars, propplot, nrow=3, ncol=5, figsizemag=2, parnames=['J0', 'J1', 'K0'], figtitle='', \
                      cmap='viridis', vmin=None, vmax=None, cbarlabel='', parname_preset=['$J_0$', '$J_1$', '$K_0$'],
                      figsize=None, norm=None, target=False):
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_under(color='grey')
    cmap.set_over(color='pink')

    if vmin is None:
        vmin = np.nanmin(propplot)
    if vmax is None:
        vmax = np.nanmax(propplot)
    if figsize is None:
        figsize = (ncol*1*figsizemag,nrow*0.7*figsizemag + 1)
    fig, axs = plt.subplots(nrow,ncol,figsize=figsize, sharex=True, sharey=True)
    fig.suptitle(figtitle, fontsize='x-large')
    axf = axs.flatten()
    for i, par2 in enumerate(search_pars[parnames[2]]):
        ax = axf[i]
        ax.set_title(f'{parname_preset[2]} = {par2:.1f}')
        if norm is None:
            im = ax.pcolormesh(search_pars[parnames[0]], search_pars[parnames[1]], propplot[i], 
                            cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            im = ax.pcolormesh(search_pars[parnames[0]], search_pars[parnames[1]], propplot[i], 
                            cmap=cmap, norm=norm)
        if i // ncol == nrow - 1:
            ax.set_xlabel(f'{parname_preset[0]}', fontsize=14)
            ax.set_xticks(np.linspace(search_pars[parnames[0]][0], search_pars[parnames[0]][-1], 3))
        if i % ncol == 0:
            ax.set_ylabel(f'{parname_preset[1]}', fontsize=14)
            ax.set_yticks(np.linspace(search_pars[parnames[1]][0], search_pars[parnames[1]][-1], 3))
        ax.tick_params(labelsize=12)

        if np.any(target):
            if par2 == target[2]:
                x = target[0]
                y = target[1]
                x_interval = search_pars[parnames[0]][1] - search_pars[parnames[0]][0]
                y_interval = search_pars[parnames[1]][1] - search_pars[parnames[1]][0]
                ax.add_patch(plt.Rectangle((x - x_interval/2, y - y_interval/2), x_interval, y_interval, fill=False, edgecolor='red', lw=2))
        
    for i in range(len(search_pars[parnames[2]]), len(axf)):
        axf[i].axis('off')


    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    cbar = fig.colorbar(im, ax=axs, extend='min', aspect= 20 * ( nrow / ncol ))
    cbar.set_label(cbarlabel)
    
    
    plt.show()

    return fig

def plot_4par_on_prop(search_pars, data_plot, figsizemag=1.5, parnames=['J0', 'J1', 'K0', 'kappa'], figtitle='', \
                      cmap='viridis', vmin=None, vmax=None, cbarlabel='', parname_preset=None,
                      figsize=None, norm=None, target=False, reverse_col_row_var=False):
    '''
    data_plot: 4D array, shape = (npar3, npar2, npar1, npar0)'''
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_under(color='grey')
    cmap.set_over(color='pink')

    if reverse_col_row_var:
        data_plot = np.swapaxes(data_plot, 0, 1)

        parnames_temp = parnames.copy()
        parnames = parnames_temp.copy()
        parnames[2] = parnames_temp[3]
        parnames[3] = parnames_temp[2]

        parname_preset_temp = parname_preset.copy()
        parname_preset = parname_preset_temp.copy()
        parname_preset[2] = parname_preset_temp[3]
        parname_preset[3] = parname_preset_temp[2]

        par2_list = search_pars[parnames[2]]
        par3_list = search_pars[parnames[3]][::-1]
    else:
        par2_list = search_pars[parnames[2]][::-1]
        par3_list = search_pars[parnames[3]]

    if parname_preset is None:
        parname_preset = parnames
    
    if vmin is None:
        vmin = np.nanmin(data_plot)
    if vmax is None:
        vmax = np.nanmax(data_plot)
    nrow = search_pars[parnames[2]].size
    ncol = search_pars[parnames[3]].size
    if figsize is None:
        figsize = (ncol*figsizemag,nrow*0.8*figsizemag + 1)
    fig, axs = plt.subplots(nrow,ncol,figsize=figsize, sharex=True, sharey=True)
    fig.suptitle(figtitle)
    for i, par2 in enumerate(par2_list): # row: k0
        for j, par3 in enumerate(par3_list): # col: kappa
            ax = axs[i,j]
            ax.tick_params(axis='both', which='major', labelsize='x-small')
            if i == 0:
                ax.set_title(f'{parname_preset[3]}\n{par3:.1f}\n', fontsize='small')
            if norm is None:
                im = ax.pcolormesh(search_pars[parnames[0]], search_pars[parnames[1]], data_plot[j, i], 
                            cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                im = ax.pcolormesh(search_pars[parnames[0]], search_pars[parnames[1]], data_plot[j, i], 
                            cmap=cmap, norm=norm)
            if i == nrow - 1:
                ax.set_xlabel(parname_preset[0])
                ax.set_xticks(np.linspace(search_pars[parnames[0]][0], search_pars[parnames[0]][-1], 3))
            if j == 0:
                ax.set_ylabel(f'{parname_preset[2]}\n{par2:.1f}\n\n{parname_preset[1]}')
                ax.set_yticks(np.linspace(search_pars[parnames[1]][0], search_pars[parnames[1]][-1], 3))
            if np.any(target):
                if (par2 == target[2]) and (par3 == target[3]):
                    x = target[0]
                    y = target[1]
                    x_interval = search_pars[parnames[0]][1] - search_pars[parnames[0]][0]
                    y_interval = search_pars[parnames[1]][1] - search_pars[parnames[1]][0]
                    ax.add_patch(plt.Rectangle((x - x_interval/2, y - y_interval/2), x_interval, y_interval, fill=False, edgecolor='red', lw=2))
    
    plt.tight_layout()
    cbar = fig.colorbar(im, ax=axs, extend='min')
    cbar.set_label(cbarlabel)
    plt.show()
    return fig

def plot_3par_on_type_a_prop(search_pars, network_evals, network_pars, color_dict, widths_mean, figtitle='', 
                              xvar='J0', yvar='J1', plotvar='K0', nrow=3, ncol=5, colsize=3, rowsize=3, dotsize=10, propsizediv=5):
    xvari, yvari, ploti = 0, 1, 2
    fig, axs = plt.subplots(nrow,ncol,figsize=(ncol*colsize,nrow*rowsize))
    fig.suptitle(figtitle)
    for j, ax in enumerate(axs.flatten()):
        if j >= len(search_pars[plotvar]):
            ax.axis('off')
            continue
        bol0 = network_pars[:,ploti] == search_pars[plotvar][j]
        for i, label in enumerate(np.unique(network_evals)):
            bol = (bol0) & (network_evals == label)
            xdata = (network_pars[bol,xvari])
            ydata = (network_pars[bol,yvari])
            if label == 'valid':
                ax.scatter(xdata, ydata, color=color_dict[label], marker='8', alpha=1, label=label, s=widths_mean[bol]//propsizediv)
            else:
                ax.scatter(xdata, ydata, color=color_dict[label], marker='8', alpha=1, label=label, s=dotsize)
            ax.set_xlabel(xvar)
            ax.set_ylabel(yvar)
            ax.set_title(f'{plotvar} = {search_pars[plotvar][j]:.2f}')
        if j == 0:
            ax.legend(fontsize='small')
    plt.tight_layout()
    plt.show()

def plot_4par_on_type_a_prop(search_pars, network_evals, network_pars, color_dict, widths_mean, figtitle='', 
                              xvar='J0', yvar='J1', plotxvar='K0', plotyvar='kappa', 
                              colsize=3, rowsize=3, dotsize=10, propsizediv=5):
    xvari, yvari, plotxvari, plotyvari = 0, 1, 2, 3
    ncol = search_pars[plotxvar].size
    nrow = search_pars[plotyvar].size
    fig, axs = plt.subplots(nrow,ncol,figsize=(ncol*colsize,nrow*rowsize))
    fig.suptitle(figtitle)
    for j in range(nrow):
        for k in range(ncol):
            ax = axs[j,k]
            bol0 = (network_pars[:,plotxvari] == search_pars[plotxvar][k]) & (network_pars[:,plotyvari] == search_pars[plotyvar][j])
            for i, label in enumerate(np.unique(network_evals)):
                bol = (bol0) & (network_evals == label)
                xdata = (network_pars[bol,xvari])
                ydata = (network_pars[bol,yvari])
                if label == 'valid':
                    ax.scatter(xdata, ydata, color=color_dict[label], marker='8', alpha=1, label=label, s=widths_mean[bol]//propsizediv)
                else:
                    ax.scatter(xdata, ydata, color=color_dict[label], marker='8', alpha=1, label=label, s=dotsize)
                ax.set_xlabel(xvar)
                ax.set_ylabel(yvar)
                ax.set_title(f'{plotxvar} = {search_pars[plotxvar][k]:.2f}')
            if k == 0:
                ax.legend(fontsize='small', loc='lower left')
                ax.set_ylabel(f'{plotyvar} = {search_pars[plotyvar][j]:.2f}')
    plt.tight_layout()
    plt.show()

def sample_stable_shape_3r(index_vis, network_evals, network_pars, network_acvs, network_ts, varnames, fig_per_row=10, times=1, maxnum=30, b0s=[0,1]):

    varnum = len(varnames)
    for num, neti in enumerate(index_vis):
        if num >= maxnum:
            break
        if num % fig_per_row == 0:
            fig = plt.figure(figsize=(30*times,1.5*times))
        ax = plt.subplot(1,fig_per_row,num % fig_per_row + 1)

        if varnum == 2:
            ax.set_title(f'{neti}, {network_evals[neti]} \n {varnames[0]}={network_pars[neti][0]:.1f}, {varnames[1]}={network_pars[neti][1]:.1f}')
        elif varnum == 3:
            ax.set_title(f'{neti}, {network_evals[neti]} \n {varnames[0]}={network_pars[neti][0]:.1f}, {varnames[1]}={network_pars[neti][1]:.1f} \
        \n {varnames[2]}={network_pars[neti][2]:.2f}')
        elif varnum == 4:
            ax.set_title(f'{neti}, {network_evals[neti]} \n {varnames[0]}={network_pars[neti][0]:.1f}, {varnames[1]}={network_pars[neti][1]:.1f} \
        \n {varnames[2]}={network_pars[neti][2]:.2f}, {varnames[3]}={network_pars[neti][3]:.1f}')
        elif varnum == 5:
            ax.set_title(f'{neti}, {network_evals[neti]} \n {varnames[0]}={network_pars[neti][0]:.1f}, {varnames[1]}={network_pars[neti][1]:.1f} \
        \n {varnames[2]}={network_pars[neti][2]:.2f}, {varnames[3]}={network_pars[neti][3]:.1f}, {varnames[3]}={network_pars[neti][4]:.1f}')
            
        ax.plot(network_acvs[neti][0,:,-1] + b0s[0], '-g', alpha=0.5, label='Central')
        ax.plot(network_acvs[neti][1,:,-1] + b0s[1], '-b', alpha=0.5, label='Left')
        ax.plot(network_acvs[neti][2,:,-1] + b0s[1], '-r', alpha=0.5, label='Right')
        # ax.legend()
        ax.plot(network_acvs[neti][0,:,-2] + b0s[0], '--g', alpha=0.5)
        ax.plot(network_acvs[neti][1,:,-2] + b0s[1], '--b', alpha=0.5)
        ax.plot(network_acvs[neti][2,:,-2] + b0s[1], '--r', alpha=0.5)

        ax.set_ylabel(f'{max(network_ts[neti])} ms')

        if num % fig_per_row == fig_per_row - 1:
            plt.show()
            
def sample_stable_shape_3r_unequal_theta(index_vis, network_evals, network_pars, network_acvs, network_ts, varnames, theta_range, fig_per_row=10, times=1, maxnum=30, b0s=[1,1]):

    for num, neti in enumerate(index_vis):
        if num >= maxnum:
            break
        if num % fig_per_row == 0:
            fig = plt.figure(figsize=(30*times,1.5*times))
        ax = plt.subplot(1,fig_per_row,num % fig_per_row + 1)
        ax.set_title(f'{neti}, {network_evals[neti]} \n {varnames[0]}={network_pars[neti][0]:.1f}, {varnames[1]}={network_pars[neti][1]:.1f} \
        \n {varnames[2]}={network_pars[neti][2]:.2f}, {varnames[3]}={network_pars[neti][3]:.1f}')
            
        ax.plot(theta_range[0], network_acvs[neti][0,:,-1] + b0s[0], '-g', alpha=0.5, label='Central')
        ax.plot(theta_range[1], network_acvs[neti][1,:,-1] + b0s[1], '-b', alpha=0.5, label='Left')
        ax.plot(theta_range[2], network_acvs[neti][2,:,-1] + b0s[1], '-r', alpha=0.5, label='Right')
        # ax.legend()
        ax.plot(theta_range[0], network_acvs[neti][0,:,-2] + b0s[0], '--g', alpha=0.5)
        ax.plot(theta_range[1], network_acvs[neti][1,:,-2] + b0s[1], '--b', alpha=0.5)
        ax.plot(theta_range[2], network_acvs[neti][2,:,-2] + b0s[1], '--r', alpha=0.5)

        ax.set_ylabel(f'{max(network_ts[neti])} ms')

        if num % fig_per_row == fig_per_row - 1:
            plt.show()

def sample_stable_shape(index_vis, network_evals, network_pars, network_acvs, network_ts, varnames, fig_per_row=10, times=1, maxnum=30, bothring=True):
    varnum = len(varnames)
    for num, neti in enumerate(index_vis):
        if num >= maxnum:
            break
        if num % fig_per_row == 0:
            fig = plt.figure(figsize=(30*times,1.5*times))
        ax = plt.subplot(1,fig_per_row,num % fig_per_row + 1)

        if varnum == 2:
            ax.set_title(f'{neti}, {network_evals[neti]} \n {varnames[0]}={network_pars[neti][0]:.1f}, {varnames[1]}={network_pars[neti][1]:.1f}')
        elif varnum == 3:
            ax.set_title(f'{neti}, {network_evals[neti]} \n {varnames[0]}={network_pars[neti][0]:.1f}, {varnames[1]}={network_pars[neti][1]:.1f} \
        \n {varnames[2]}={network_pars[neti][2]:.2f}')
        elif varnum == 4:
            ax.set_title(f'{neti}, {network_evals[neti]} \n {varnames[0]}={network_pars[neti][0]:.1f}, {varnames[1]}={network_pars[neti][1]:.1f} \
        \n {varnames[2]}={network_pars[neti][2]:.2f}, {varnames[3]}={network_pars[neti][3]:.1f}')
            
        ax.plot(network_acvs[neti][0,:,-1], '-b', alpha=0.5)
        if bothring:
            ax.plot(network_acvs[neti][1,:,-1], '-r', alpha=0.5)
        try:
            ax.plot(network_acvs[neti][0,:,-2], '--b', alpha=0.5)
            ax.plot(network_acvs[neti][0,:,-3], '-.b', alpha=0.5)
            ax.legend([max(network_ts[neti]), network_ts[neti][-2], network_ts[neti][-3]])
            if bothring:
                ax.plot(network_acvs[neti][1,:,-2], '--r', alpha=0.5)
                ax.plot(network_acvs[neti][1,:,-3], '-.r', alpha=0.5)
        except:
            pass
        if num % fig_per_row == fig_per_row - 1:
            plt.show()

def sample_stable_shape_1ring(index_vis, network_evals, network_pars, network_acvs, network_ts, varnames, fig_per_row=10, times=1, maxnum=30, plotpre=True):
    '''plot the activity of networks in its stable state for one ring'''
    varnum = len(varnames)
    for num, neti in enumerate(index_vis):
        if num >= maxnum:
            break
        if num % fig_per_row == 0:
            fig = plt.figure(figsize=(30*times,1.5*times))
        ax = plt.subplot(1,fig_per_row,num % fig_per_row + 1)
        if varnum == 2:
            ax.set_title(f'{neti}, {network_evals[neti]} \n {varnames[0]}={network_pars[neti][0]:.1f}, {varnames[1]}={network_pars[neti][1]:.1f}')
        elif varnum == 3:
            ax.set_title(f'{neti}, {network_evals[neti]} \n {varnames[0]}={network_pars[neti][0]:.1f}, {varnames[1]}={network_pars[neti][1]:.1f} \
        \n {varnames[2]}={network_pars[neti][2]:.2f}')
        elif varnum == 4:
            ax.set_title(f'{neti}, {network_evals[neti]} \n {varnames[0]}={network_pars[neti][0]:.1f}, {varnames[1]}={network_pars[neti][1]:.1f} \
        \n {varnames[2]}={network_pars[neti][2]:.2f}, {varnames[3]}={network_pars[neti][3]:.1f}')
            
        ax.plot(network_acvs[neti][0,:,-1], '-b', alpha=0.5)
        if plotpre:
            try:
                ax.plot(network_acvs[neti][0,:,-2], '--b', alpha=0.5)
                ax.plot(network_acvs[neti][0,:,-3], '-.b', alpha=0.5)
                ax.legend([max(network_ts[neti]), network_ts[neti][-2], network_ts[neti][-3]])
            except:
                pass
        if num % fig_per_row == fig_per_row - 1:
            plt.show()

def plot_sample_vv_relation(index_vv, network_pars, network_eval_moving, inputs, Vels, fig_per_row=12, parnames=['J0', 'J1']):
    for num, neti in enumerate(index_vv):
        if num % fig_per_row == 0:
            fig = plt.figure(figsize=(30,1.5))
        ax = plt.subplot(1,fig_per_row,num % fig_per_row + 1)
        if len(parnames) == 2:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}')
        elif len(parnames) == 3:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}')
        elif len(parnames) == 4:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}, {parnames[3]} = {network_pars[neti,3]:.1f}')
        points = np.concatenate( [ [0], np.where(network_eval_moving[neti] == 'stable moving')[0] ] )
        ax.plot(inputs[points], Vels[neti,points,0], '-bo', label='left')
        ax.plot(inputs[points], Vels[neti,points,1], '-ro', label='right')
        ax.set_xlim([0,1])
        ax.legend()
        if num % fig_per_row == fig_per_row - 1:
            plt.show()

def plot_overview_25_s2ring(inputs, index_vv, network_acvs_moving, network_pars, network_eval_moving, Vels, bump_height,
                            max_firate, max_amplitude, skewness_mean, theta_range, plot_height=False, parnames=['J0', 'J1'],
                            plot_lf=True, fir_from_zero=True, b0=1, addb=1, subb=1, title3='Mean Activity'):
    num_pre_fig = 4
    pre_fig_mag = 2
    fig_per_row = len(inputs) + num_pre_fig * pre_fig_mag + 1
    for num, neti in enumerate(index_vv):
        fig = plt.figure(figsize=(25,1.5))
        # input - velocity
        ax = plt.subplot(1,fig_per_row//pre_fig_mag,1)

        if len(parnames) == 2:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}')
        elif len(parnames) == 3:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}')
        elif len(parnames) == 4:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}, {parnames[3]} = {network_pars[neti,3]:.1f}')

        points = np.concatenate( [ [0], np.where(network_eval_moving[neti] == 'stable moving')[0] ] )
        ax.plot(inputs[points], Vels[neti,points,0], '-bo', label='left')
        ax.plot(inputs[points], Vels[neti,points,1], '-ro', label='right')
        ax.set_xlim([0,max(inputs)])
        ax.set_ylabel('Turn/s')
        ax.set_xlabel('Inputs')
        ax.legend()
        # input - Firrate
        ax = plt.subplot(1,fig_per_row//pre_fig_mag,2)
        if plot_lf:
            ax.plot(inputs, max_firate[neti,:,0], '-bo')
            ax.plot(inputs, max_firate[neti,:,1], '-ro')
        ax.plot(inputs, np.mean(max_firate[neti], axis=1), '-go', label='mean')
        ax.set_xlim([0,max(inputs)])
        ax.set_title('Peak Firing Rate')
        ax.set_xlabel('Inputs')
        if plot_height:
            ax.plot(inputs, bump_height[neti,:,0], '--bo', label='left')
            ax.plot(inputs, bump_height[neti,:,1], '--ro', label='right')
            ax.legend()
        if fir_from_zero:
            ax.set_ylim([0, np.max(max_firate[neti] * 1.1)])
        # input - amplitude
        ax = plt.subplot(1,fig_per_row//pre_fig_mag,3)
        
        ax.plot(inputs, max_amplitude[neti,:,0], '-bo')
        ax.plot(inputs, max_amplitude[neti,:,1], '-ro')
        ax.plot(inputs, np.mean(max_amplitude[neti], axis=1), '-go', label='mean')
        ax.legend()
        ax.set_xlim([0,max(inputs)])
        ax.set_title(title3)
        ax.set_xlabel('Inputs')
        # input - skweness
        ax = plt.subplot(1,fig_per_row//pre_fig_mag,4)
        ax.plot(inputs, skewness_mean[neti], '-bo')
        ax.set_xlim([0,max(inputs)])
        ax.set_title('Mean Skewness')
        ax.set_xlabel('Inputs')
        # for ratioi in range(len(inputs)):
        #     ax = plt.subplot(1,fig_per_row,num_pre_fig * pre_fig_mag+1+ratioi)
        #     ax.plot(theta_range, network_acvs_moving[neti,ratioi][0,:,-1] + b0 + inputs[ratioi], '-b', label='left end', alpha=0.5)
        #     ax.plot(theta_range, network_acvs_moving[neti,ratioi][0,:,-2] + b0 + inputs[ratioi], '--b', label='left -20ms', alpha=0.5)
        #     ax.plot(theta_range, network_acvs_moving[neti,ratioi][1,:,-1] + b0 - inputs[ratioi], '-r', label='right end', alpha=0.5)
        #     ax.plot(theta_range, network_acvs_moving[neti,ratioi][1,:,-2] + b0 - inputs[ratioi], '--r', label='right -20ms', alpha=0.5)
        #     ax.set_title(f'Input: {inputs[ratioi]}')
        

        minv, maxv = 100, -100
        for ratioi in range(len(inputs)):
            mint = np.min(network_acvs_moving[neti,ratioi][1,:,-1] + b0 - subb*inputs[ratioi])
            minv = min(minv, mint)
            maxt = np.max(network_acvs_moving[neti,ratioi][0,:,-1] + b0 + addb*inputs[ratioi])
            maxv = max(maxv, maxt)
        for ratioi in range(len(inputs)):
            ax = plt.subplot(1,fig_per_row,num_pre_fig * pre_fig_mag+1+ratioi)
            # acv_avg1 = np.mean(network_acvs_moving[neti,ratioi][:,:,-1] + b0, axis=0)
            # acv_avg2 = np.mean(network_acvs_moving[neti,ratioi][:,:,-2] + b0, axis=0)
            # ax.plot(theta_range, acv_avg1, '-g', label='current', alpha=0.5)
            # ax.plot(theta_range, acv_avg2, '--g', label='-20ms', alpha=0.5)
            ax.plot(theta_range, network_acvs_moving[neti,ratioi][0,:,-1] + b0 + addb*inputs[ratioi], '-b', label='left end', alpha=0.5)
            ax.plot(theta_range, network_acvs_moving[neti,ratioi][0,:,-2] + b0 + addb*inputs[ratioi], '--b', label='left -20ms', alpha=0.5)
            ax.plot(theta_range, network_acvs_moving[neti,ratioi][1,:,-1] + b0 - subb*inputs[ratioi], '-r', label='right end', alpha=0.5)
            ax.plot(theta_range, network_acvs_moving[neti,ratioi][1,:,-2] + b0 - subb*inputs[ratioi], '--r', label='right -20ms', alpha=0.5)
            ax.set_title(f'Input: {inputs[ratioi]}')
            ax.set_ylim([minv, maxv])

        plt.show()

def plot_overview_25_3ring(inputs, index_vv, network_acvs_moving, network_pars, network_eval_moving, Vels,
                            max_firate, mean_firate, acv_mean, theta_range, bE, bI, parnames=['K', 'H', 'L'],
                            plot_lf=True, fir_from_zero=True, addb=1, subb=1, plot_fir=True, title3='Mean Activity',
                            FWHM_from_zero=False):
    Vels = Vels * 360
    theta_range = theta_range / np.pi * 180
    num_pre_fig = 4
    pre_fig_mag = 1
    fig_per_row = len(inputs) + num_pre_fig * pre_fig_mag + 1
    for num, neti in enumerate(index_vv):
        fig = plt.figure(figsize=(25,1.5))
        # input - velocity
        ax = plt.subplot(1,fig_per_row//pre_fig_mag,1)

        if len(parnames) == 2:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}')
        elif len(parnames) == 3:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}')
        elif len(parnames) == 4:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}, {parnames[3]} = {network_pars[neti,3]:.1f}')
        elif len(parnames) == 1:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}')

        points = np.concatenate( [ [0], np.where(network_eval_moving[neti] == 'stable moving')[0] ] )
        ax.plot(inputs[points], Vels[neti,points,1], '-bo', label='left')
        ax.plot(inputs[points], Vels[neti,points,2], '-ro', label='right')
        ax.plot(inputs[points], Vels[neti,points,0], '-go', label='center')
        ax.set_xlim([min(inputs),max(inputs)])
        ax.set_ylabel('Degree/s')
        ax.set_xlabel('Inputs')
        ax.legend()
        # input - Max Firrate
        ax = plt.subplot(1,fig_per_row//pre_fig_mag,2)
        if plot_lf:
            ax.plot(inputs, max_firate[neti,:,1], '-bo')
            ax.plot(inputs, max_firate[neti,:,2], '-ro')
            ax.plot(inputs, np.mean(max_firate[neti], axis=1), '-ko', label='mean')
        ax.plot(inputs, max_firate[neti,:,0], '-go')
        ax.set_xlim([min(inputs),max(inputs)])
        ax.set_title('Peak Firing Rate')
        ax.set_xlabel('Inputs')
        if fir_from_zero & plot_lf:
            ax.set_ylim([0, np.max(max_firate[neti] * 1.1)])
        # input - mean firrate
        ax = plt.subplot(1,fig_per_row//pre_fig_mag,3)
        
        ax.plot(inputs, mean_firate[neti,:,1], '-bo')
        ax.plot(inputs, mean_firate[neti,:,2], '-ro')
        ax.plot(inputs, mean_firate[neti,:,0], '-go')
        ax.plot(inputs, np.mean(mean_firate[neti], axis=1), '-ko', label='mean')
        ax.legend()
        ax.set_xlim([min(inputs),max(inputs)])
        ax.set_title('Mean Firing Rate')
        ax.set_xlabel('Inputs')
        if fir_from_zero:
            ax.set_ylim([0, np.max(mean_firate[neti] * 1.1)])
        # input - mean acv / FWHM
        ax = plt.subplot(1,fig_per_row//pre_fig_mag,4)
        ax.plot(inputs, acv_mean[neti,:,1], '-bo')
        ax.plot(inputs, acv_mean[neti,:,2], '-ro')
        ax.plot(inputs, acv_mean[neti,:,0], '-go')
        ax.plot(inputs, np.mean(acv_mean[neti], axis=1), '-ko', label='mean')
        ax.set_xlim([min(inputs),max(inputs)])
        ax.set_title(title3)
        ax.set_xlabel('Inputs')
        if FWHM_from_zero:
            ax.set_ylim([0, np.max(acv_mean[neti] * 1.1)])

        # Shape dependence on input
        bE = scalr2vec(bE, inputs)
        bI = scalr2vec(bI, inputs)
        minv, maxv = 100, -100
        for ratioi, ratiov in enumerate(inputs):
            mintc = np.min(network_acvs_moving[neti,ratioi][0,:,-1] + bE[ratioi])
            mint = np.min(network_acvs_moving[neti,ratioi][2,:,-1] + bI[ratioi] - subb*inputs[ratioi])
            minv = min(mintc, mint, minv)
            maxtc = np.max(network_acvs_moving[neti,ratioi][0,:,-1] + bE[ratioi])
            maxt = np.max(network_acvs_moving[neti,ratioi][1,:,-1] + bI[ratioi] + addb*inputs[ratioi])
            maxv = max(maxtc, maxt, maxv)
        if plot_fir:
            for ratioi, ratiov in enumerate(inputs):
                ax = plt.subplot(1,fig_per_row,num_pre_fig * pre_fig_mag+1+ratioi)
                ax.plot(theta_range, max0x(network_acvs_moving[neti,ratioi][1,:,-1] + bI[ratioi] + addb*inputs[ratioi]), '-b', label='left end', alpha=0.5)
                ax.plot(theta_range, max0x(network_acvs_moving[neti,ratioi][1,:,-2] + bI[ratioi] + addb*inputs[ratioi]), '--b', label='left -20ms', alpha=0.5)
                ax.plot(theta_range, max0x(network_acvs_moving[neti,ratioi][2,:,-1] + bI[ratioi] - subb*inputs[ratioi]), '-r', label='right end', alpha=0.5)
                ax.plot(theta_range, max0x(network_acvs_moving[neti,ratioi][2,:,-2] + bI[ratioi] - subb*inputs[ratioi]), '--r', label='right -20ms', alpha=0.5)
                ax.plot(theta_range, max0x(network_acvs_moving[neti,ratioi][0,:,-1] + bE[ratioi]), '-g', label='center end', alpha=0.5)
                ax.plot(theta_range, max0x(network_acvs_moving[neti,ratioi][0,:,-2] + bE[ratioi]), '--g', label='center -20ms', alpha=0.5)
                ax.set_title(f'Input: {inputs[ratioi]}')
                ax.set_ylim([0, maxv])
        else:
            for ratioi, ratiov in enumerate(inputs):
                ax = plt.subplot(1,fig_per_row,num_pre_fig * pre_fig_mag+1+ratioi)
                ax.plot(theta_range, network_acvs_moving[neti,ratioi][1,:,-1] + bI[ratioi] + addb*inputs[ratioi], '-b', label='left end', alpha=0.5)
                ax.plot(theta_range, network_acvs_moving[neti,ratioi][1,:,-2] + bI[ratioi] + addb*inputs[ratioi], '--b', label='left -20ms', alpha=0.5)
                ax.plot(theta_range, network_acvs_moving[neti,ratioi][2,:,-1] + bI[ratioi] - subb*inputs[ratioi], '-r', label='right end', alpha=0.5)
                ax.plot(theta_range, network_acvs_moving[neti,ratioi][2,:,-2] + bI[ratioi] - subb*inputs[ratioi], '--r', label='right -20ms', alpha=0.5)
                ax.plot(theta_range, network_acvs_moving[neti,ratioi][0,:,-1] + bE[ratioi], '-g', label='center end', alpha=0.5)
                ax.plot(theta_range, network_acvs_moving[neti,ratioi][0,:,-2] + bE[ratioi], '--g', label='center -20ms', alpha=0.5)
                ax.set_title(f'Input: {inputs[ratioi]}')
                ax.set_ylim([minv, maxv])

        plt.show()

def plot_overview_25_3ring_asyb(inputs, index_vv, network_acvs_moving, network_pars, network_eval_moving, Vels,
                            max_firate, mean_firate, acv_mean, theta_range, bE, bl, br, parnames=['K', 'H', 'L'],
                            plot_lf=True, fir_from_zero=True, addb=1, subb=1, plot_fir=True, title3='Mean Activity'):
    Vels = Vels * 360
    theta_range = theta_range / np.pi * 180
    num_pre_fig = 4
    pre_fig_mag = 1
    fig_per_row = len(inputs) + num_pre_fig * pre_fig_mag + 1
    for num, neti in enumerate(index_vv):
        fig = plt.figure(figsize=(25,1.5))
        # input - velocity
        ax = plt.subplot(1,fig_per_row//pre_fig_mag,1)

        if len(parnames) == 2:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}')
        elif len(parnames) == 3:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}')
        elif len(parnames) == 4:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}, {parnames[3]} = {network_pars[neti,3]:.1f}')
        elif len(parnames) == 1:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}')

        points = np.concatenate( [ [0], np.where(network_eval_moving[neti] == 'stable moving')[0] ] )
        ax.plot(inputs[points], Vels[neti,points,1], '-bo', label='left')
        ax.plot(inputs[points], Vels[neti,points,2], '-ro', label='right')
        ax.plot(inputs[points], Vels[neti,points,0], '-go', label='center')
        ax.set_xlim([min(inputs),max(inputs)])
        ax.set_ylabel('Degree/s')
        ax.set_xlabel('Inputs')
        ax.legend()
        # input - Max Firrate
        ax = plt.subplot(1,fig_per_row//pre_fig_mag,2)
        if plot_lf:
            ax.plot(inputs, max_firate[neti,:,1], '-bo')
            ax.plot(inputs, max_firate[neti,:,2], '-ro')
        ax.plot(inputs, max_firate[neti,:,0], '-go')
        ax.plot(inputs, np.mean(max_firate[neti], axis=1), '-ko', label='mean')
        ax.set_xlim([min(inputs),max(inputs)])
        ax.set_title('Peak Firing Rate')
        ax.set_xlabel('Inputs')
        if fir_from_zero:
            ax.set_ylim([0, np.max(max_firate[neti] * 1.1)])
        # input - mean firrate
        ax = plt.subplot(1,fig_per_row//pre_fig_mag,3)
        
        ax.plot(inputs, mean_firate[neti,:,1], '-bo')
        ax.plot(inputs, mean_firate[neti,:,2], '-ro')
        ax.plot(inputs, mean_firate[neti,:,0], '-go')
        ax.plot(inputs, np.mean(mean_firate[neti], axis=1), '-ko', label='mean')
        ax.legend()
        ax.set_xlim([min(inputs),max(inputs)])
        ax.set_title('Mean Firing Rate')
        ax.set_xlabel('Inputs')
        if fir_from_zero:
            ax.set_ylim([0, np.max(mean_firate[neti] * 1.1)])
        # input - mean acv
        ax = plt.subplot(1,fig_per_row//pre_fig_mag,4)
        ax.plot(inputs, acv_mean[neti,:,1], '-bo')
        ax.plot(inputs, acv_mean[neti,:,2], '-ro')
        ax.plot(inputs, acv_mean[neti,:,0], '-go')
        ax.plot(inputs, np.mean(acv_mean[neti], axis=1), '-ko', label='mean')
        ax.set_xlim([min(inputs),max(inputs)])
        ax.set_title(title3)
        ax.set_xlabel('Inputs')
        

        # Shape dependence on input
        bE = scalr2vec(bE, inputs)
        bl = scalr2vec(bl, inputs)
        br = scalr2vec(br, inputs)
        minv, maxv = 100, -100
        for ratioi, ratiov in enumerate(inputs):
            mintc = np.min(network_acvs_moving[neti,ratioi][0,:,-1] + bE[ratioi])
            mint = np.min(network_acvs_moving[neti,ratioi][2,:,-1] + bl[ratioi])
            minv = min(mintc, mint, minv)
            maxtc = np.max(network_acvs_moving[neti,ratioi][0,:,-1] + bE[ratioi])
            maxt = np.max(network_acvs_moving[neti,ratioi][1,:,-1] + bl[ratioi])
            maxv = max(maxtc, maxt, maxv)
        if plot_fir:
            for ratioi, ratiov in enumerate(inputs):
                ax = plt.subplot(1,fig_per_row,num_pre_fig * pre_fig_mag+1+ratioi)
                ax.plot(theta_range, max0x(network_acvs_moving[neti,ratioi][1,:,-1] + bl[ratioi]), '-b', label='left end', alpha=0.5)
                ax.plot(theta_range, max0x(network_acvs_moving[neti,ratioi][1,:,-2] + bl[ratioi]), '--b', label='left -20ms', alpha=0.5)
                ax.plot(theta_range, max0x(network_acvs_moving[neti,ratioi][2,:,-1] + br[ratioi]), '-r', label='right end', alpha=0.5)
                ax.plot(theta_range, max0x(network_acvs_moving[neti,ratioi][2,:,-2] + br[ratioi]), '--r', label='right -20ms', alpha=0.5)
                ax.plot(theta_range, max0x(network_acvs_moving[neti,ratioi][0,:,-1] + bE[ratioi]), '-g', label='center end', alpha=0.5)
                ax.plot(theta_range, max0x(network_acvs_moving[neti,ratioi][0,:,-2] + bE[ratioi]), '--g', label='center -20ms', alpha=0.5)
                ax.set_title(f'Input: {inputs[ratioi]}')
                ax.set_ylim([0, maxv])
        else:
            for ratioi, ratiov in enumerate(inputs):
                ax = plt.subplot(1,fig_per_row,num_pre_fig * pre_fig_mag+1+ratioi)
                ax.plot(theta_range, network_acvs_moving[neti,ratioi][1,:,-1] + bl[ratioi], '-b', label='left end', alpha=0.5)
                ax.plot(theta_range, network_acvs_moving[neti,ratioi][1,:,-2] + bl[ratioi], '--b', label='left -20ms', alpha=0.5)
                ax.plot(theta_range, network_acvs_moving[neti,ratioi][2,:,-1] + br[ratioi], '-r', label='right end', alpha=0.5)
                ax.plot(theta_range, network_acvs_moving[neti,ratioi][2,:,-2] + br[ratioi], '--r', label='right -20ms', alpha=0.5)
                ax.plot(theta_range, network_acvs_moving[neti,ratioi][0,:,-1] + bE[ratioi], '-g', label='center end', alpha=0.5)
                ax.plot(theta_range, network_acvs_moving[neti,ratioi][0,:,-2] + bE[ratioi], '--g', label='center -20ms', alpha=0.5)
                ax.set_title(f'Input: {inputs[ratioi]}')
                ax.set_ylim([minv, maxv])

        plt.show()
        
def plot_overview_25_s2ring2(inputs, index_vv, network_acvs_moving, network_pars, network_eval_moving, Vels, bump_height,
                            max_firate, max_amplitude, skewness_mean, theta_range, b0, parnames=['J0', 'J1', 'K0', 'kappa'],
                            plot_lf=True, fir_from_zero=True, figsize=(40,1.5)):
    num_pre_fig = 4
    fig_per_row = 3 + num_pre_fig
    for num, neti in enumerate(index_vv):
        fig = plt.figure(figsize=figsize)

        # input - velocity
        ax = plt.subplot(1,fig_per_row,1)

        if len(parnames) == 2:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}')
        elif len(parnames) == 3:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}')
        elif len(parnames) == 4:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}, {parnames[3]} = {network_pars[neti,3]:.1f}')

        points = np.concatenate( [ [0], np.where(network_eval_moving[neti] == 'stable moving')[0] ] )
        ax.plot(inputs[points], Vels[neti,points,0], '-bo', label='left')
        ax.plot(inputs[points], Vels[neti,points,1], '-ro', label='right')
        ax.set_xlim([min(inputs),max(inputs)])
        ax.set_ylabel('Turn/s')
        ax.set_xlabel('Inputs')
        ax.legend()

        # input - Firrate
        ax = plt.subplot(1,fig_per_row,2)
        if plot_lf:
            ax.plot(inputs, max_firate[neti,:,0], '-bo')
            ax.plot(inputs, max_firate[neti,:,1], '-ro')
        ax.plot(inputs, np.mean(max_firate[neti], axis=1), '-go', label='mean')
        ax.set_xlim([min(inputs),max(inputs)])
        ax.set_title('Firing Rate')
        ax.set_xlabel('Inputs')
        if fir_from_zero:
            ax.set_ylim([0, np.max(max_firate[neti] * 1.1)])

        # input - amplitude
        ax = plt.subplot(1,fig_per_row,3)
        ax.plot(inputs, np.mean(max_amplitude[neti], axis=1), '-go')
        if plot_lf:
            ax.plot(inputs, max_amplitude[neti,:,0], '-bo')
            ax.plot(inputs, max_amplitude[neti,:,1], '-ro')
        ax.set_xlim([min(inputs),max(inputs)])
        ax.set_title('Amplitude')
        ax.set_xlabel('Inputs')

        # input - skweness
        ax = plt.subplot(1,fig_per_row,4)
        ax.plot(inputs, skewness_mean[neti], '-go')
        ax.set_xlim([0,max(inputs)])
        ax.set_title('Mean Skewness')
        ax.set_xlabel('Inputs')

        # Bump at different speed
        input_midi = len(inputs) // 4
        zeroid = np.where(inputs == 0)[0][0]
        ids = [0, input_midi, zeroid, len(inputs)-input_midi-1, -1]
        colors = ['darkred', 'tomato', 'darkgreen', 'royalblue', 'darkblue']
        ax = plt.subplot(1,fig_per_row,5)
        for i,ratioi in enumerate(ids):
            # print(ratioi)
            acv_avg = np.mean(network_acvs_moving[neti,ratioi][:,:,-1] + b0, axis=0)
            shiftv = np.argmax(acv_avg) - len(theta_range) // 2
            acv_avg = np.roll(acv_avg, -shiftv)
            ax.plot(theta_range, acv_avg, color=colors[i], label=f'{inputs[ratioi]}')
        ax.legend(loc='upper right', fontsize='small')
        ax.set_title(f'Bump Shape at Different Input')

        plt.show()

def plot_lrbump_s2ring(inputs, index_vv, network_acvs_moving, network_pars, network_eval_moving, Vels, bump_height,
                            max_firate, max_amplitude, skewness_mean, theta_range, plot_height=True, parnames=['J0', 'J1'],
                            plot_lf=True, fir_from_zero=True, b0=1):
    fig_per_row = len(inputs)
    for num, neti in enumerate(index_vv):
        for ratioi in range(len(inputs)):
            ax = plt.subplot(1, fig_per_row, 1 + ratioi)
            ax.plot(theta_range, network_acvs_moving[neti,ratioi][0,:,-1] + b0 + inputs[ratioi], '-b', label='left end', alpha=0.5)
            ax.plot(theta_range, network_acvs_moving[neti,ratioi][0,:,-2] + b0 + inputs[ratioi], '--b', label='left -20ms', alpha=0.5)
            ax.plot(theta_range, network_acvs_moving[neti,ratioi][1,:,-1] + b0 - inputs[ratioi], '-r', label='right end', alpha=0.5)
            ax.plot(theta_range, network_acvs_moving[neti,ratioi][1,:,-2] + b0 - inputs[ratioi], '--r', label='right -20ms', alpha=0.5)
            ax.set_title(f'Input: {inputs[ratioi]}')
        plt.show()

def plot_overview_25_s2ring_onebump(inputs, neti, network_acvs_moving, network_pars, network_eval_moving, Vels, bump_height,
                            max_firate, max_amplitude, skewness_mean, theta_range, b0, pretitle='', parnames=['J0', 'J1', 'K0', 'kappa'],
                            plot_lf=True, fir_from_zero=True, figsize=(20, 3), parname_preset=['$J_0$', '$J_1$', '$K_0$', '$\kappa$'],
                            plot_ids=None, plot_colors=None):
    fig, axs = plt.subplots(1, 4, figsize=figsize, width_ratios=[2,1,1,1])

    if len(parnames) == 2:
        fig.suptitle(pretitle+f'{parname_preset[0]} $= {network_pars[neti,0]:.1f}$, {parname_preset[1]} $= {network_pars[neti,1]:.1f}$', fontsize=14)
    elif len(parnames) == 3:
        fig.suptitle(pretitle+f'{parname_preset[0]} $= {network_pars[neti,0]:.1f}$, {parname_preset[1]} $= {network_pars[neti,1]:.1f}$, \
{parname_preset[2]} $= {network_pars[neti,2]:.2f}$', fontsize=14)
    elif len(parnames) == 4:
        fig.suptitle(pretitle+f'{parname_preset[0]} $= {network_pars[neti,0]:.1f}$, {parname_preset[1]} $= {network_pars[neti,1]:.1f}$, \
{parname_preset[2]} $= {network_pars[neti,2]:.2f}$, {parname_preset[3]} $= {network_pars[neti,3]:.1f}$', fontsize=14)

    # Bump Shape at different speed
    ax = axs[0]
    input_midi = len(inputs) // 4
    zeroid = np.where(inputs == 0)[0][0]
    if plot_ids is None:
        plot_ids = [0, input_midi, zeroid, len(inputs)-input_midi-1, -1]
    if plot_colors is None:
        plot_colors = ['darkred', 'tomato', 'darkgreen', 'royalblue', 'darkblue']
    for i,ratioi in enumerate(plot_ids):
        acv_avg = np.mean(network_acvs_moving[neti,ratioi][:,:,-1] + b0, axis=0)
        shiftv = np.argmax(acv_avg) - len(theta_range) // 2
        acv_avg = np.roll(acv_avg, -shiftv)
        ax.plot(theta_range, acv_avg, color=plot_colors[i], label=f'$\Delta b: ${inputs[ratioi]}', alpha=0.7)
    ax.set_xlabel('$\\theta$ - $\\theta_0$ (rad)', fontsize=14)
    ax.legend(fontsize=10)
    # ax.set_title(f'$\Delta b$ and Bump Shape', fontsize=14)
    ax.set_ylabel('Activity (A.U.)', fontsize=14)
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi/2))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax.tick_params(labelsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # input - velocity
    ax = axs[1]
    # ax.set_title(f'$\Delta b$ and Bump Moving Speed', fontsize=14)
    points = np.concatenate( [ [0], np.where(network_eval_moving[neti] == 'stable moving')[0] ] )
    ax.plot(inputs[points], np.mean(Vels[neti,points] * 2 * np.pi, axis=1), '-go')
    # ax.set_xlim([min(inputs),max(inputs)])
    ax.set_ylabel('Angular Velocity (rad/s)', fontsize=14)
    ax.set_xlabel('$\Delta b$', fontsize=14)
    ax.set_xlim([0, max(inputs)])
    ax.set_ylim([0, None])
    ax.tick_params(labelsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # input - peak activity
    ax = axs[2]
    ax.plot(inputs, np.mean(max_firate[neti], axis=1), '-go', label='mean')
    if plot_lf:
        ax.plot(inputs, max_firate[neti,:,0], '-bo', label='left')
        ax.plot(inputs, max_firate[neti,:,1], '-ro', label='right')
        ax.legend()
    # ax.set_xlim([min(inputs),max(inputs)])
    # ax.set_title('$\Delta b$ and Peak Activity', fontsize=14)
    ax.set_xlabel('$\Delta b$', fontsize=14)
    if fir_from_zero:
        ax.set_ylim([0, np.max(max_firate[neti] * 1.1)])
    ax.set_ylabel('Peak Activity (A.U.)', fontsize=14)
    ax.set_xlim([0, max(inputs)])
    ax.tick_params(labelsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # input - skweness
    ax = axs[3]
    ax.plot(inputs, skewness_mean[neti], '-go')
    ax.set_xlim([0,max(inputs)])
    # ax.set_title('$\Delta b$ and Skewness', fontsize=14)
    ax.set_xlabel('$\Delta b$', fontsize=14)
    ax.set_ylim([-0.2,0.2])
    ax.set_ylabel('Skewness', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout(w_pad=2)
    plt.show()
    return fig

def plot_overview(search_num, inputs, valid_index_linear_move, network_acvs, network_acvs_moving, 
                  network_eval_moving, network_pars, Vels, theta_range, amplitude_plot=True, skewness_plot=True,
                  compare_ids = [0, 2, 4, 6, 8]):
    if amplitude_plot:
        max_amplitude = compute_amplitude(search_num, inputs, valid_index_linear_move, network_acvs, network_acvs_moving)
        # print(max_amplitude[valid_index_linear_move])
    if skewness_plot:
        skewness = compute_skewness(search_num, inputs, valid_index_linear_move, network_acvs, network_acvs_moving, theta_range)

    add_col = 1 + int(amplitude_plot) + int(skewness_plot)
    fig_per_row = len(inputs) + add_col
    index_vv = valid_index_linear_move
    for num, neti in enumerate(index_vv):
        fig = plt.figure(figsize=(40,1.5))
        # input - velocity
        ax = plt.subplot(1,fig_per_row,1)
        ax.set_title(f'({neti})\nJ0={network_pars[neti,0]:.1f}, J1={network_pars[neti,1]:.0f}\n')
        points = np.concatenate( [ [0], np.where(network_eval_moving[neti] == 'stable moving')[0] ] )
        ax.plot(inputs[points], Vels[neti,points,0], '-bo', label='left')
        ax.plot(inputs[points], Vels[neti,points,1], '-ro', label='right')
        ax.set_xlim([0,1])
        ax.set_ylabel('Turn/s')
        ax.set_xlabel('Inputs')
        ax.legend()
        if amplitude_plot:
            ax = plt.subplot(1,fig_per_row,2)
            ax.plot(inputs, max_amplitude[neti,:,0], '-bo', label='left')
            ax.plot(inputs, max_amplitude[neti,:,1], '-ro', label='right')
            ax.set_xlim([0,1])
            ax.set_title('Amplitude')
            ax.set_xlabel('Inputs')
        if skewness_plot:
            ax = plt.subplot(1,fig_per_row,add_col)
            ax.plot(inputs, skewness[neti,:,0], '-bo')
            ax.plot(inputs, skewness[neti,:,1], '-ro')
            ax.set_xlim([0,1])
            ax.set_title('Skewness')
            ax.set_xlabel('Inputs')
        for ratioi in range(len(inputs)):
            ax = plt.subplot(1,fig_per_row,add_col+ratioi+1)
            if ratioi == 0:
                ax.plot(theta_range, network_acvs[neti][0,:,-1], '-b', label='left')
                ax.plot(theta_range, network_acvs[neti][1,:,-1], '-r', label='right')
            else:
                ax.plot(theta_range, network_acvs_moving[neti,ratioi][0,:,-1], '-b', label='left end')
                ax.plot(theta_range, network_acvs_moving[neti,ratioi][0,:,-2], '--b', label='left -25ms')
                ax.plot(theta_range, network_acvs_moving[neti,ratioi][1,:,-1], '-r', label='right end')
                ax.plot(theta_range, network_acvs_moving[neti,ratioi][1,:,-2], '--r', label='right -25ms')
            ax.set_title(f'Input: {inputs[ratioi]}')
        plt.show()

def plot_overview_25_s1ring(inputs, index_vv, network_acvs_moving, network_pars, network_eval_moving, Vels, height,
                            bottom_acv, theta_range, b0, parnames=['JI', 'JE'], actfun=max0x,
                            figsize=(40,1.5), compare_ids = [0, 2, 4, 6, 8]):
    num_pre_fig = 3
    fig_per_row = 2 + num_pre_fig
    for num, neti in enumerate(index_vv):
        fig = plt.figure(figsize=figsize)

        # input - velocity
        ax = plt.subplot(1,fig_per_row,1)

        if len(parnames) == 2:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}')
        elif len(parnames) == 3:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}')
        elif len(parnames) == 4:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}, {parnames[3]} = {network_pars[neti,3]:.1f}')

        points = np.concatenate( [ [0], np.where(network_eval_moving[neti] == 'stable moving')[0] ] )
        ax.plot(inputs[points], Vels[neti,points], '-go')
        ax.set_xlim([min(inputs),max(inputs)])
        ax.set_ylabel('Turn/s')
        ax.set_xlabel('Inputs')
        # ax.legend()

        # input - Height
        ax = plt.subplot(1,fig_per_row,2)
        ax.plot(inputs, height[neti] + b0, '-go')
        ax.set_xlim([min(inputs),max(inputs)])
        ax.set_title('Height')
        ax.set_xlabel('Inputs')
        # if height_from_zero:
        #     ax.set_ylim([0, np.max(height[neti] * 1.1)])

        # input - mean_acv
        ax = plt.subplot(1,fig_per_row,3)
        ax.plot(inputs, bottom_acv[neti] + b0, '-go')
        ax.set_xlim([min(inputs),max(inputs)])
        ax.set_title('Bottom Activity')
        ax.set_xlabel('Inputs')

        # Bump S at different speed
        ids = compare_ids
        colors = ['darkred', 'tomato', 'darkgreen', 'royalblue', 'darkblue']
        ax = plt.subplot(1,fig_per_row,4)
        for i,ratioi in enumerate(ids):
            # print(ratioi)
            membrane = network_acvs_moving[neti,ratioi][:,-1] + b0
            # shiftv = int(cstat.mean(theta_range, membrane + min(membrane)) / (2*np.pi) * len(theta_range)) - len(theta_range) // 2
            shiftv = np.argmax(membrane) - len(theta_range) // 2
            membrane = np.roll(membrane, -shiftv)
            ax.plot(theta_range, membrane, color=colors[i], label=f'{inputs[ratioi]}')
        ax.legend(loc='upper right', fontsize='small')
        ax.set_title(f'Membrane Potential Shape at Different Input')

        # Bump f at different speed
        ids = compare_ids
        colors = ['darkred', 'tomato', 'darkgreen', 'royalblue', 'darkblue']
        ax = plt.subplot(1,fig_per_row,5)
        for i,ratioi in enumerate(ids):
            # print(ratioi)
            membrane = network_acvs_moving[neti,ratioi][:,-1] + b0
            shiftv = np.argmax(membrane) - len(theta_range) // 2
            membrane = np.roll(membrane, -shiftv)
            f = actfun(membrane)
            ax.plot(theta_range, f, color=colors[i], label=f'{inputs[ratioi]}')
        ax.legend(loc='upper right', fontsize='small')
        ax.set_title(f'Firing rate Shape at Different Input')

        plt.show()

def plot_acv_overview_3r_unequal_HD(network_acvs_moving, index_plot, theta_range, inputs, network_pars, 
                         mid_id=-3, fin_id=-6, figsize=(20,5), parnames=('CI', 'CE', 'LI'), title='', 
                         nrow=2, ncol=4):
    '''
    When arranging the subplots, assume the lenght of input is 9, from negative to positive 
    plot the activity and emphasize the phase difference
    '''
    
    input_num = len(network_acvs_moving[index_plot[0]])
    zeroid = np.where(inputs == 0)[0][0]
    input_ids = np.array([i for i in range(zeroid)] + [i for i in range(zeroid+1, input_num)])
    for neti in index_plot:
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
        axs = axes.flatten()
        fig.suptitle(f'{title}({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}, {parnames[3]} = {network_pars[neti,3]:.1f}')

        acv = network_acvs_moving[neti]

        for input_idi in range(input_num-1):
            input_id = input_ids[input_idi]
            ax = axs[input_idi]
            ax.set_title(f'Input: {inputs[input_id]}')

            ring_colors = ['g', 'b', 'r']
            time_ls = ['-', ':']
            t_ids = [-1, fin_id]
        
            # Central ring
            for ringi in range(3):
                for t_counter, tid in enumerate(t_ids):
                    ## Activity
                    ax.plot(theta_range[ringi], acv[input_id][ringi,:,tid], c=ring_colors[ringi], ls=time_ls[t_counter])
                    ## Vector average
                    a_acv = acv[input_id][ringi,:,tid]
                    se = a_acv - min(a_acv)
                    circular_mean = cstat.mean(theta_range[ringi], se, axis=0)
                    ax.axvline(circular_mean, 0, 1, c=ring_colors[ringi], ls=time_ls[t_counter])

        plt.tight_layout()
        plt.show()

def plot_acv_overview_3r(network_acvs_moving, index_plot, theta_range, inputs, network_pars, 
                         mid_id=-3, fin_id=-6, figsize=(20,5), parnames=('CI', 'CE', 'LI'), title='', plotmaxloc=False,
                         nrow=2, ncol=4):
    '''
    When arranging the subplots, assume the lenght of input is 9, from negative to positive 
    plot the activity and emphasize the phase difference
    '''
    
    input_num = len(network_acvs_moving[index_plot[0]])
    zeroid = np.where(inputs == 0)[0][0]
    input_ids = np.array([i for i in range(zeroid)] + [i for i in range(zeroid+1, input_num)])
    for neti in index_plot:
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
        axs = axes.flatten()
        if len(parnames) == 2:
            fig.suptitle(f'{title}({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}')
        elif len(parnames) == 3:
            fig.suptitle(f'{title}({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}')
        elif len(parnames) == 4:
            fig.suptitle(f'{title}({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}, {parnames[3]} = {network_pars[neti,3]:.1f}')


        acv = network_acvs_moving[neti]

        for input_idi in range(input_num-1):
            input_id = input_ids[input_idi]
            ax = axs[input_idi]
            ax.set_title(f'Input: {inputs[input_id]}')

            ring_colors = ['g', 'b', 'r']
            time_ls = ['-', ':']
            t_ids = [-1, fin_id]
        
            if not plotmaxloc:
                # Central ring
                ## Activity
                ax.plot(acv[input_id][0,:,-1], 'g-')
                ax.plot(acv[input_id][0,:,fin_id], 'g:')

                ## Vector average
                peakid = cal_peak_loc(acv[input_id][0,:,-1], theta_range)
                ax.axvline(peakid, 0, 1, c='g')
                peakid = cal_peak_loc(acv[input_id][0,:,fin_id], theta_range)
                ax.axvline(peakid, 0, 1, c='g', ls=':')

                # Left ring
                ## Activity
                ax.plot(acv[input_id][1,:,-1], 'b-', alpha=0.5)
                ax.plot(acv[input_id][1,:,fin_id], 'b:', alpha=0.5)

                ## Vector average
                peakid = cal_peak_loc(acv[input_id][1,:,-1], theta_range)
                ax.axvline(peakid, 0, 1, c='b', alpha=0.5)
                peakid = cal_peak_loc(acv[input_id][1,:,fin_id], theta_range)
                ax.axvline(peakid, 0, 1, c='b', ls=':')

                # Right ring
                ## Activity
                ax.plot(acv[input_id][2,:,-1], 'r-', alpha=0.5)
                ax.plot(acv[input_id][2,:,fin_id], 'r:', alpha=0.5)

                ## Vector average
                peakid = cal_peak_loc(acv[input_id][2,:,-1], theta_range)
                ax.axvline(peakid, 0, 1, c='r', alpha=0.5)
                peakid = cal_peak_loc(acv[input_id][2,:,fin_id], theta_range)
                ax.axvline(peakid, 0, 1, c='r', ls=':')
            else:
                for i in range(3):
                    for t in range(2):
                        acv_tpr = acv[input_id][i,:,t_ids[t]]
                        ax.plot(theta_range, acv_tpr, c=ring_colors[i], ls=time_ls[t], alpha=0.5)
                        ax.axvline(theta_range[np.argmax(acv_tpr)], 0, 1, c=ring_colors[i], ls=time_ls[t], alpha=0.5)

        plt.tight_layout()
        plt.show()

def plot_acv_overview_3r_cen(network_acvs_moving, index_plot, theta_range, inputs, network_pars, 
                         mid_id=-3, fin_id=-6, figsize=(20,5), parnames=('CI', 'CE', 'LI'), title='', plotmaxloc=False,
                         nrow=2, ncol=4):
    '''
    When arranging the subplots, assume the lenght of input is 9, from negative to positive 
    plot the activity and emphasize the phase difference
    '''
    
    input_num = len(network_acvs_moving[index_plot[0]])
    zeroid = np.where(inputs == 0)[0][0]
    input_ids = np.array([i for i in range(zeroid)] + [i for i in range(zeroid+1, input_num)])
    for neti in index_plot:
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
        axs = axes.flatten()
        if len(parnames) == 2:
            fig.suptitle(f'{title}({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}')
        elif len(parnames) == 3:
            fig.suptitle(f'{title}({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}')
        elif len(parnames) == 4:
            fig.suptitle(f'{title}({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}, {parnames[3]} = {network_pars[neti,3]:.1f}')


        acv = network_acvs_moving[neti]

        for input_idi in range(input_num-1):
            input_id = input_ids[input_idi]
            ax = axs[input_idi]
            ax.set_title(f'Input: {inputs[input_id]}')

            ring_colors = ['g', 'b', 'r']
            time_ls = ['-', ':']
            t_ids = [-1, fin_id]
        
            if not plotmaxloc:
                # Central ring
                ## Activity
                ax.plot(acv[input_id][0,:,-1], 'g-')
                ax.plot(acv[input_id][0,:,fin_id], 'g:')

                ## Vector average
                if isinstance(theta_range, list):
                    theta_range = theta_range[0]
                peakid = cal_peak_loc(acv[input_id][0,:,-1], theta_range)
                ax.axvline(peakid, 0, 1, c='g')
                peakid = cal_peak_loc(acv[input_id][0,:,fin_id], theta_range)
                ax.axvline(peakid, 0, 1, c='g', ls=':')

        plt.tight_layout()
        plt.show()

def plot_acv_overview_2r(network_acvs_moving, index_plot, theta_range, inputs, network_pars, 
                         mid_id=-3, fin_id=-5, figsize=(20,5), parnames=('JI', 'JE'), title='', plot_r=True):
    '''
    When arranging the subplots, assume the lenght of input is 9, from negative to positive 
    plot the activity and emphasize the phase difference
    '''
    
    input_num = len(network_acvs_moving[index_plot[0]])
    zeroid = np.where(inputs == 0)[0][0]
    input_ids = np.array([i for i in range(zeroid)] + [i for i in range(zeroid+1, input_num)])
    for neti in index_plot:
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axs = axes.flatten()
        if len(parnames) == 2:
            fig.suptitle(f'{title}({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}')
        elif len(parnames) == 3:
            fig.suptitle(f'{title}({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}')
        elif len(parnames) == 4:
            fig.suptitle(f'{title}({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}, {parnames[3]} = {network_pars[neti,3]:.1f}')


        acv = network_acvs_moving[neti]

        for input_idi in range(input_num-1):
            input_id = input_ids[input_idi]
            ax = axs[input_idi]
            ax.set_title(f'Input: {inputs[input_id]}')

            # Left ring
            ## Activity
            ax.plot(acv[input_id][0,:,-1], 'b-', alpha=0.5)
            ax.plot(acv[input_id][0,:,mid_id], 'b--', alpha=0.5)
            ax.plot(acv[input_id][0,:,fin_id], 'b:', alpha=0.5)

            ## Vector average
            peakid = cal_peak_loc(acv[input_id][0,:,-1], theta_range)
            ax.axvline(peakid, 0, 1, c='b', alpha=0.5)
            peakid = cal_peak_loc(acv[input_id][0,:,mid_id], theta_range)
            ax.axvline(peakid, 0, 1, c='b', ls='--')
            peakid = cal_peak_loc(acv[input_id][0,:,fin_id], theta_range)
            ax.axvline(peakid, 0, 1, c='b', ls=':')
            
            if plot_r:
                # Right ring
                ## Activity
                ax.plot(acv[input_id][1,:,-1], 'r-', alpha=0.5)
                ax.plot(acv[input_id][1,:,mid_id], 'r--', alpha=0.5)
                ax.plot(acv[input_id][1,:,fin_id], 'r:', alpha=0.5)

                ## Vector average
                peakid = cal_peak_loc(acv[input_id][1,:,-1], theta_range)
                ax.axvline(peakid, 0, 1, c='r', alpha=0.5)
                peakid = cal_peak_loc(acv[input_id][1,:,mid_id], theta_range)
                ax.axvline(peakid, 0, 1, c='r', ls='--')
                peakid = cal_peak_loc(acv[input_id][0,:,fin_id], theta_range)
                ax.axvline(peakid, 0, 1, c='r', ls=':')

        plt.tight_layout()
        plt.show()

def plot_acv_overview_1r(network_acvs_moving, index_plot, theta_range, inputs, network_pars, 
                         mid_id=-6, ini_id = 0, figsize=(20,5), parnames=('JI', 'JE'), title=''):
    '''
    When arranging the subplots, assume the lenght of input is 9, from negative to positive 
    plot the activity and emphasize the phase difference
    '''
    
    input_num = len(network_acvs_moving[index_plot[0]])
    zeroid = np.where(inputs == 0)[0][0]
    input_ids = np.array([i for i in range(zeroid)] + [i for i in range(zeroid+1, input_num)])
    for neti in index_plot:
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axs = axes.flatten()
        if len(parnames) == 2:
            fig.suptitle(f'{title}({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}')
        elif len(parnames) == 3:
            fig.suptitle(f'{title}({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}')
        elif len(parnames) == 4:
            fig.suptitle(f'{title}({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}, {parnames[3]} = {network_pars[neti,3]:.1f}')


        acv = network_acvs_moving[neti]

        for input_idi in range(input_num-1):
            input_id = input_ids[input_idi]
            ax = axs[input_idi]
            ax.set_title(f'Input: {inputs[input_id]}')

            # Central ring
            ## Activity
            ax.plot(acv[input_id][:,-1], 'g-')
            ax.plot(acv[input_id][:,mid_id], 'g--')
            ax.plot(acv[input_id][:,ini_id], 'g:')

            ## Vector average
            peakid = cal_peak_loc(acv[input_id][:,-1], theta_range)
            ax.axvline(peakid, 0, 1, c='g')
            peakid = cal_peak_loc(acv[input_id][:,mid_id], theta_range)
            ax.axvline(peakid, 0, 1, c='g', ls='--')
            peakid = cal_peak_loc(acv[input_id][:,ini_id], theta_range)
            ax.axvline(peakid, 0, 1, c='g', ls='--')

        plt.tight_layout()
        plt.show()

def plot_overview_25_s3ring_simple(inputs, index_vv, network_acvs_moving, network_pars, Vels, 
                            theta_range, b0, bump_amplitudes, parnames=['CI', 'CE', 'LI'], actfun=max0x,
                            figsize=(40,1.5), compare_ids = [0, 2, 4, 6, 8], title='', kind='normal', normalize=False):
    
    '''
    Plot 8 figures: v-input, mean & peak acv-input (2 plots), peak & mean firate-input (2 plots), 
    Mean&Max_acv&firate-input (1 plot), acv shape, firrate shape
    '''
    ring_num = 3
    total_inputs = cal_total_inputs(b0, inputs, ring_num, kind)

    for num, neti in enumerate(index_vv):
        fig, axs = plt.subplots(2, 4, figsize=figsize)
        
        if len(parnames) == 2:
            fig.suptitle(f'{title}({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}')
        elif len(parnames) == 3:
            fig.suptitle(f'{title}({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}')
        elif len(parnames) == 4:
            fig.suptitle(f'{title}({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}, {parnames[3]} = {network_pars[neti,3]:.1f}')
        elif len(parnames) == 5:
            fig.suptitle(f'{title}({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}, {parnames[3]} = {network_pars[neti,3]:.1f}, {parnames[4]} = {network_pars[neti,3]:.1f}')

        # input - velocity
        ax = axs[1,0]
        ax.plot(inputs, Vels[neti,:,1], '-bo', label='Left')
        ax.plot(inputs, Vels[neti,:,2], '--r*', label='Right')
        ax.plot(inputs, Vels[neti,:,0], ':g+', label='Central')
        ax.legend()
        ax.set_title('Velocity')
        ax.set_ylabel('Turn/s')
        ax.set_xlabel('Inputs')

        # input - mean & peak acv-input (2 plots), peak & mean firate-input (2 plots)
        titles4 = ['Peak ACV', 'Mean ACV', 'Peak Firing rate', 'Mean Firing rate']
        for i in range(4):
            ax = axs[0,i]
            yvar = bump_amplitudes[i]
            yvar = yvar/yvar[:,4:5,:] if normalize else yvar
            ax.plot(inputs, yvar[neti,:,0], '-ko', label='Central')
            ax.plot(inputs, yvar[neti,:,1], '-bo', label='Left')
            ax.plot(inputs, yvar[neti,:,2], '-ro', label='Right')
            ax.set_title(titles4[i])
            ax.set_xlabel('Inputs')

        # input - (right - left) difference
        ax = axs[1,1]
        colors4 = ['gold', 'darkorange', 'purple', 'violet']
        linetypes4 = ['-', '-', '--', '--']
        for i in range(4,8):
            yvar = bump_amplitudes[i]
            ax.plot(inputs, yvar[neti,:], color=colors4[i-4], label=titles4[i-4], linestyle=linetypes4[i-4])
        ax.legend()
        ax.set_title('$Right - Left$')
        ax.set_xlabel('Inputs')

        # Bump S at different speed
        ids = compare_ids
        colors = [['lawngreen', 'forestgreen', 'darkgreen'], ['cornflowerblue', 'royalblue', 'navy'], ['lightcoral', 'brown', 'darkred']]
        linetypes = ['-', (0, (2,2)), (2,(2,2))]
        rings = ['Centeral', 'Left', 'Right']
        ax = axs[1,2]
        for i,ratioi in enumerate(ids):
            for ringi in range(ring_num):
                membrane = network_acvs_moving[neti,ratioi][ringi,:,-1] + total_inputs[ringi][ratioi]

                shiftv = np.rint(cstat.mean(theta_range, membrane-min(membrane)) / (2*np.pi) * len(theta_range)).astype(int)
                membrane = np.roll(membrane, -shiftv)
                ax.plot(theta_range, membrane, color=colors[ringi][i], label=f'{inputs[ratioi]},{rings[ringi]}', linestyle=linetypes[ringi])
        ax.legend(loc='best', fontsize='small')
        ax.set_title(f' ACV')

        # Bump f at different speed
        ax = axs[1,3]
        for i,ratioi in enumerate(ids):
            for ringi in range(ring_num):
                membrane = network_acvs_moving[neti,ratioi][ringi,:,-1] + total_inputs[ringi][ratioi]
                f = actfun(membrane)
                
                shiftv = np.rint(cstat.mean(theta_range, f) / (2*np.pi) * len(theta_range)).astype(int)
                f = np.roll(f, -shiftv)
                ax.plot(theta_range, f, color=colors[ringi][i], label=f'{inputs[ratioi]},{rings[ringi]}', linestyle=linetypes[ringi])
        ax.set_title(f'Firing rate')

        plt.tight_layout()
        plt.show()

def plot_overview_25_s2ring_simple(inputs, index_vv, network_acvs_moving, network_pars, network_eval_moving, Vels, 
                            theta_range, b0, bump_amplitudes, parnames=['JI', 'JE'], actfun=max0x,
                            figsize=(40,1.5), compare_ids = [0, 2, 4, 6, 8], kind='normal'):
    
    '''
    Plot 8 figures: v-input, mean & peak acv-input (2 plots), peak & mean firate-input (2 plots), 
    Mean&Max_acv&firate-input (1 plot), acv shape, firrate shape
    '''
    ring_num = 2
    total_inputs = cal_total_inputs(b0, inputs, ring_num, kind)

    for num, neti in enumerate(index_vv):
        fig, axs = plt.subplots(2, 4, figsize=figsize)
        
        if len(parnames) == 2:
            fig.suptitle(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}')
        elif len(parnames) == 3:
            fig.suptitle(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}')
        elif len(parnames) == 4:
            fig.suptitle(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}, {parnames[3]} = {network_pars[neti,3]:.1f}')

        # input - velocity
        ax = axs[1,0]
        ax.plot(inputs, Vels[neti,:,0], '-bo', label='Left')
        ax.plot(inputs, Vels[neti,:,1], ':r*', label='Right')
        ax.legend()
        ax.set_title('Velocity')
        ax.set_ylabel('Turn/s')
        ax.set_xlabel('Inputs')

        # input - mean & peak acv-input (2 plots), peak & mean firate-input (2 plots)
        titles4 = ['Peak ACV', 'Mean ACV', 'Peak Firing rate', 'Mean Firing rate']
        for i in range(4):
            ax = axs[0,i]
            yvar = bump_amplitudes[i]
            ax.plot(inputs, yvar[neti,:,0], '-bo', label='Left')
            ax.plot(inputs, yvar[neti,:,1], '-ro', label='Right')
            ax.set_title(titles4[i])
            ax.set_xlabel('Inputs')

        # input - (right - left) difference
        ax = axs[1,1]
        colors4 = ['gold', 'darkorange', 'purple', 'violet']
        linetypes4 = ['-', '-', '--', '--']
        for i in range(4,8):
            yvar = bump_amplitudes[i]
            ax.plot(inputs, yvar[neti,:], color=colors4[i-4], label=titles4[i-4], linestyle=linetypes4[i-4])
        ax.legend()
        ax.set_title('$Right - Left$')
        ax.set_xlabel('Inputs')

        # Bump S at different speed
        ids = compare_ids
        colors = [['cornflowerblue', 'royalblue', 'navy'], ['lightcoral', 'brown', 'darkred']]
        linetypes = ['-', '--']
        rings = ['Left', 'Right']
        ax = axs[1,2]
        for i,ratioi in enumerate(ids):
            for ringi in range(ring_num):
                membrane = network_acvs_moving[neti,ratioi][ringi,:,-1] + total_inputs[ringi][ratioi]

                shiftv = np.rint(cstat.mean(theta_range, membrane-min(membrane)) / (2*np.pi) * len(theta_range)).astype(int)
                membrane = np.roll(membrane, -shiftv)
                ax.plot(theta_range, membrane, color=colors[ringi][i], label=f'{inputs[ratioi]},{rings[ringi]}', linestyle=linetypes[ringi])
        ax.legend(loc='best', fontsize='small')
        ax.set_title(f' ACV')

        # Bump f at different speed
        ax = axs[1,3]
        for i,ratioi in enumerate(ids):
            for ringi in range(ring_num):
                membrane = network_acvs_moving[neti,ratioi][ringi,:,-1] + total_inputs[ringi][ratioi]
                f = actfun(membrane)
                
                shiftv = np.rint(cstat.mean(theta_range, f) / (2*np.pi) * len(theta_range)).astype(int)
                f = np.roll(f, -shiftv)
                ax.plot(theta_range, f, color=colors[ringi][i], label=f'{inputs[ratioi]},{rings[ringi]}', linestyle=linetypes[ringi])
        ax.set_title(f'Firing rate')

        plt.tight_layout()
        plt.show()

def plot_overview_25_s1ring_simple(inputs, index_vv, network_acvs_moving, network_pars, network_eval_moving, Vels, 
                            theta_range, b0, parnames=['JI', 'JE'], actfun=max0x,
                            figsize=(40,1.5), compare_ids = [0, 2, 4, 6, 8]):
    
    theta_num = len(theta_range)
    for num, neti in enumerate(index_vv):
        fig, axs = plt.subplots(1, 3, figsize=figsize, width_ratios=[1,2,2])
        ax = axs[0]

        if len(parnames) == 2:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}')
        elif len(parnames) == 3:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}')
        elif len(parnames) == 4:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}, {parnames[3]} = {network_pars[neti,3]:.1f}')

        # input - velocity
        points = np.concatenate( [ [0], np.where(network_eval_moving[neti] == 'stable moving')[0] ] )
        ax.plot(inputs[points], Vels[neti,points], '-go')
        ax.set_xlim([min(inputs),max(inputs)])
        ax.set_ylabel('Turn/s')
        ax.set_xlabel('Inputs')

        # Bump S at different speed
        ids = compare_ids
        colors = ['darkred', 'tomato', 'darkgreen', 'royalblue', 'darkblue']
        linetypes = ['-', '--']
        ax = axs[1]
        for i,ratioi in enumerate(ids):
            for time in range(2):
                membrane = network_acvs_moving[neti,ratioi][:,-(time+1)]
                if time == 0:
                    shiftv = np.rint(cstat.mean(theta_range, membrane-min(membrane), axis=0) / (2*np.pi) * theta_num - 0.5).astype(int)
                membrane = np.roll(membrane, -shiftv)
                ax.plot(theta_range, membrane, color=colors[i], label=f'{inputs[ratioi]}', linestyle=linetypes[time])
        ax.legend(loc='upper right', fontsize='small')
        ax.set_title(f'Membrane Potential Shape at Different Input')

        # Bump f at different speed
        ids = compare_ids
        colors = ['darkred', 'tomato', 'darkgreen', 'royalblue', 'darkblue']
        ax = axs[2]
        for i,ratioi in enumerate(ids):
            for time in range(1):
                membrane = network_acvs_moving[neti,ratioi][:,-(time+1)] + b0
                if time == 0:
                    shiftv = np.argmax(membrane) - len(theta_range) // 2
                membrane = np.roll(membrane, -shiftv)
                f = actfun(membrane)
                ax.plot(theta_range, f, color=colors[i], label=f'{inputs[ratioi]}')
        ax.legend(loc='upper right', fontsize='small')
        ax.set_title(f'Firing rate Shape at Different Input')

        plt.show()

def cor_stat_print_sep(input_diff_cors, valid_index_linear_move):
    show_value = input_diff_cors[valid_index_linear_move][:,0]
    print('peak_acv')
    print(f'{np.nanmin(show_value):.3f} = Min[Cor: input - RL diff]')
    print(f'{np.nanmean(show_value):.3f} = Mean[Cor: input - RL diff]')
    print(f'{np.nanstd(show_value):.3f} = SD[Cor: input - RL diff]')

    show_value = input_diff_cors[valid_index_linear_move][:,1]
    print('\nmean acv')
    print(f'{np.nanmin(show_value):.3f} = Min[Cor: input - RL diff]')
    print(f'{np.nanmean(show_value):.3f} = Mean[Cor: input - RL diff]')
    print(f'{np.nanstd(show_value):.3f} = SD[Cor: input - RL diff]')

    show_value = input_diff_cors[valid_index_linear_move][:,2]
    print('\npeak_firate')
    print(f'{np.nanmin(show_value):.3f} = Min[Cor: input - RL diff]')
    print(f'{np.nanmean(show_value):.3f} = Mean[Cor: input - RL diff]')
    print(f'{np.nanstd(show_value):.3f} = SD[Cor: input - RL diff]')

    show_value = input_diff_cors[valid_index_linear_move][:,3]
    print('\nmean_firate')
    print(f'{np.nanmin(show_value):.3f} = Min[Cor: input - RL diff]')
    print(f'{np.nanmean(show_value):.3f} = Mean[Cor: input - RL diff]')
    print(f'{np.nanstd(show_value):.3f} = SD[Cor: input - RL diff]')

def cor_stat_print_all(input_diff_cors, valid_index_linear_move):
    show_value = input_diff_cors[valid_index_linear_move]
    print('peak_acv, mean_acv, peak_firate, mean_firate')
    
    print(f'{np.nanmean(show_value):.3f} = All Mean[Cor: input - RL diff]')
    print(f'{np.nanstd(show_value):.3f} = All SD[Cor: input - RL diff]')
    print(f'{np.nanmin(show_value):.3f} = All Min[Cor: input - RL diff]')
    
    print(f'{np.nanmean(show_value[:,1]):.3f} = Mean ACV Mean[Cor: input - RL diff]')
    print(f'{np.nanstd(show_value[:,1]):.3f} = Mean ACV SD[Cor: input - RL diff]')
    print(f'{np.nanmin(show_value[:,1]):.3f} = Mean ACV Min[Cor: input - RL diff]')

def plot_cor(input_diff_cors, valid_index_linear_move, figsize=(10,2)):
    fig = plt.figure(figsize=figsize)
    labels = ['peak_acv', 'mean_acv', 'peak_firate', 'mean_firate']
    linetypes = [(0, (1, 3)), (1, (1, 3)), (2, (1, 3)), (3, (1, 3))]
    for i in range(4):
        plt.plot(input_diff_cors[valid_index_linear_move][:,i], label=labels[i], linestyle=linetypes[i], linewidth=3)
    plt.legend()
    plt.xlabel('Net Index')
    plt.show()

def plot_mean_dif_cor(input_diff_cors, valid_index_linear_move, figsize=(10,2)):
    fig = plt.figure(figsize=figsize)
    plt.plot(input_diff_cors[valid_index_linear_move][:,1], label='mean_acv', c='orange')
    plt.legend()
    plt.xlabel('Net Index')
    plt.show()

# Plot functions for 66_master_thesis_picture
def plot_2_par_on_type2(network_evals, network_pars, figi, figtitle='', xvar='JI', yvar='JE', figsize=(5,5), dotsize=30, fontsize=11):
    xvari = 0
    yvari = 1
    dpis = [300, 100]
    for round in range(2):
        fig = plt.figure(figsize=figsize, dpi=dpis[round])
        fig.suptitle(figtitle)
        for i, label in enumerate(np.unique(network_evals)):
            bol = network_evals == label
            xdata = (network_pars[bol,xvari])
            ydata = (network_pars[bol,yvari])
            plt.scatter(xdata, ydata, color=color_dict[label], marker='8', alpha=1, label=label, s=dotsize)
            plt.xlabel(xvar, fontsize=fontsize)
            plt.ylabel(yvar, fontsize=fontsize)
            plt.tick_params(labelsize=fontsize-2)
        plt.legend(frameon=True, fontsize=fontsize, loc='upper right')
        if round == 0:
            plt.savefig(f'C:/Users/15824/OneDrive/文档/00-Master Thesis/figures/sup_fig{figi}.png')
            plt.close()
        else:
            plt.show()

def plot_3par_on_type2(search_pars, network_evals, network_pars, figi, figtitle='', legend_subplot_loc=0, 
                       legend_loc='best', legend_fontsize=11, xvar='$J_I$', yvar='$J_E$', plotvar='kappa', plotvar_show='$\kappa$',
                       nrow=3, ncol=5, colsize=3, rowsize=3, dotsize=10, fontsize=11):
    
    xvari, yvari, ploti = 0, 1, 2
    dpis = [100, 300]
    for round in range(2):
        fig, axs = plt.subplots(nrow,ncol,figsize=(ncol*colsize,nrow*rowsize), dpi=dpis[round])
        fig.suptitle(figtitle)

        for j, ax in enumerate(axs.flatten()):
            if j >= len(search_pars[plotvar]):
                ax.axis('off')
                continue
            bol0 = network_pars[:,ploti] == search_pars[plotvar][j]
            for i, label in enumerate(np.unique(network_evals)):
                bol = (bol0) & (network_evals == label)
                xdata = (network_pars[bol,xvari])
                ydata = (network_pars[bol,yvari])
                ax.scatter(xdata, ydata, color=color_dict[label], marker='8', alpha=1, label=label, s=dotsize)
                ax.set_xlabel(xvar, fontsize=fontsize)
                ax.set_ylabel(yvar, fontsize=fontsize)
                ax.set_title(f'{plotvar_show} = {search_pars[plotvar][j]:.2f}')
                ax.tick_params(labelsize=fontsize-2)
            if j == legend_subplot_loc:
                ax.legend(fontsize=legend_fontsize, loc=legend_loc)
        plt.tight_layout()

        if round == 1:
            plt.savefig(f'C:/Users/15824/OneDrive/文档/00-Master Thesis/figures/sup_fig{figi}.png')
            plt.close()
        else:
            plt.show()

def plot_2par_on_property4(propplot, search_pars, figsize, figi, title=None, xlabel='JI', ylabel='JE', 
                           xshow = '$J_I$', yshow = '$J_E$', cbarlabel=None, cmap='viridis', fontsize=11):
    dpis = [100, 300]
    for round in range(2):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpis[round])
        fig.suptitle(title)
        im = ax.pcolormesh(search_pars[xlabel], search_pars[ylabel], propplot, cmap=cmap)
        ax.set_xlabel(xshow, fontsize=fontsize)
        ax.set_ylabel(yshow, fontsize=fontsize)
        cb = plt.colorbar(im, ax=ax)
        ax.tick_params(labelsize=fontsize-2)
        cb.ax.tick_params(labelsize=fontsize-2)
        plt.tight_layout()
        if round == 0:
            plt.show()
        else:
            plt.savefig(f'C:/Users/15824/OneDrive/文档/00-Master Thesis/figures/sup_fig{figi}.png')
            plt.close()

def plot_3par_on_prop2(search_pars, linrange_plot, figi, nrow=3, ncol=5, parnames=['JI', 'JE', 'kappa'], figtitle='', \
                      cmap='viridis', vmin=None, vmax=None, cbarlabel='', parname_preset=['$J_I$', '$J_E$', '$\kappa$'],
                      figsize=None, fontsize=11, colmag=2, rowmag=2, hspace=0.3):
    
    dpis = [100, 300]
    cbaspect = 20 * ( nrow / ncol )

    fig_num = len(search_pars[parnames[2]])
        
    for round in range(2):
        if vmin is None:
            vmin = np.nanmin(linrange_plot)
        if vmax is None:
            vmax = np.nanmax(linrange_plot)
        if figsize is None:
            figsize = (ncol*colmag,nrow*rowmag)
        fig, axs = plt.subplots(nrow,ncol,figsize=figsize, sharex=True, sharey=True, dpi=dpis[round])
        # fig.suptitle(figtitle, fontsize='x-large')
        axf = axs.flatten()
        for i, par2 in enumerate(search_pars[parnames[2]]):
            ax = axf[i]
            ax.set_title(f'{parname_preset[2]} = {par2:.1f}', fontsize=fontsize)
            im = ax.pcolormesh(search_pars[parnames[0]], search_pars[parnames[1]], linrange_plot[i], 
                            cmap=cmap, vmin=vmin, vmax=vmax)
            if i > fig_num - ncol - 1:
                ax.set_xlabel(f'{parname_preset[0]}', fontsize=fontsize)
                ax.set_xticks(np.linspace(search_pars[parnames[0]][0], search_pars[parnames[0]][-1], 3))
            if i % ncol == 0:
                ax.set_ylabel(f'{parname_preset[1]}', fontsize=fontsize)
                ax.set_yticks(np.linspace(search_pars[parnames[1]][0], search_pars[parnames[1]][-1], 3))
            ax.tick_params(labelsize=fontsize-2)

        plt.tight_layout()
        plt.subplots_adjust(hspace=hspace)
        cb = fig.colorbar(im, ax=axs, aspect=cbaspect)
        cb.ax.tick_params(labelsize=fontsize-2)

        for i in range(fig_num, len(axf)):
            axf[i].axis('off')
            
        if round == 0:
            plt.show()
        else:
            plt.savefig(f'C:/Users/15824/OneDrive/文档/00-Master Thesis/figures/sup_fig{figi}.png')
            plt.close()

    return fig

def plot_4par_on_prop2(search_pars, data_plot, figi, parnames=['J0', 'J1', 'K0', 'kappa'], \
                      cmap='viridis', vmin=None, vmax=None, parname_preset=['$J_0$', '$J_1$', '$K_0$', '$\kappa$'],
                      figsize=None, colmag=2, rowmag=2, fontsize=11):
    '''
    data_plot: 4D array, shape = (npar3, npar2, npar1, npar0)'''
    
    if vmin is None:
        vmin = np.nanmin(data_plot)
    if vmax is None:
        vmax = np.nanmax(data_plot)
    nrow = search_pars[parnames[2]].size
    ncol = search_pars[parnames[3]].size
    if figsize is None:
        figsize = (ncol*colmag,nrow*rowmag)

    aspect = 20 * (nrow / ncol)

    dpis = [100, 300]
    for round in range(2):
        fig, axs = plt.subplots(nrow,ncol,figsize=figsize, sharex=True, sharey=True, dpi=dpis[round])
        for i, par2 in enumerate(search_pars[parnames[2]][::-1]): # row: k0
            for j, par3 in enumerate(search_pars[parnames[3]]): # col: kappa
                ax = axs[i,j]
                if i == 0:
                    ax.set_title(f'{parname_preset[3]} $= {par3:.1f}$\n', fontsize=fontsize)
                im = ax.pcolormesh(search_pars[parnames[0]], search_pars[parnames[1]], data_plot[j, i], 
                                cmap=cmap, vmin=vmin, vmax=vmax)
                if i == nrow - 1:
                    ax.set_xlabel(parname_preset[0], fontsize=fontsize)
                    ax.set_xticks(np.linspace(search_pars[parnames[0]][0], search_pars[parnames[0]][-1], 3))
                if j == 0:
                    ax.set_ylabel(f'{parname_preset[2]} $= {par2:.0f}$\n \n{parname_preset[1]}', fontsize=fontsize)
                    ax.set_yticks(np.linspace(search_pars[parnames[1]][0], search_pars[parnames[1]][-1], 3))
                ax.tick_params(labelsize=fontsize-2)

        plt.tight_layout()
        cbar = fig.colorbar(im, ax=axs, aspect=aspect)
        cbar.ax.tick_params(labelsize=fontsize-2)

        if round == 0:
            plt.show()
        else:
            plt.savefig(f'C:/Users/15824/OneDrive/文档/00-Master Thesis/figures/sup_fig{figi}.png')
            plt.close()
    return fig

def plot_acv_example_66(figi, network_acvs_moving, Vels, bump_amplitudes, theta_range,
                        neti, figsize, compare_ids, ring_num, actfun, b0, inputs, dpis, fontsize, legendsize, kind='normal',
                        legendcloc=(1.15, 0.6)):
    
    '''
    The order of original ring axis is central, left, right
    '''

    total_inputs = cal_total_inputs(b0, inputs, ring_num, kind)
    ringadd = ring_num - 2
    labels3r = ['Right', 'Left', 'Central']
    clmtypes1 = ['-ro', '--b*', ':k+']
    clmtypes2 = ['-r.', '-b.', '-k.']
    orders = [ringadd+1, ringadd, 0]
    colorsvr = [['lightcoral', 'brown', 'darkred'], ['cornflowerblue', 'royalblue', 'navy'], ['darkgrey', 'dimgrey', 'black']]
    linetypes = [(0, (2,2)), (2,(2,2)), '-']

    for round in range(2):
        fig, axs = plt.subplots(4, 2, figsize=figsize, dpi=dpis[round], height_ratios=[1,1,0,1.1])

        for ax in axs.flatten():
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=fontsize-2)
            ax.title.set_fontsize(fontsize)
            ax.xaxis.label.set_fontsize(fontsize)
            ax.yaxis.label.set_fontsize(fontsize)

        # input - velocity
        ax = axs[0,0]
        for i in range(ring_num):
            ax.plot(inputs, Vels[neti,:,orders[i]], clmtypes1[i], label=labels3r[i])
            ax.legend(fontsize=legendsize, loc=(0.7,0.5))

        ax.set_ylabel('Turn/s')
        ax.set_xticks([-1, -0.5, 0, 0.5, 1], ['','','','',''])
        ax.text(-0.15, 1.2, 'A', fontsize=fontsize, fontweight='bold', va='top', ha='right', transform=ax.transAxes)

        # input - (right - left) difference
        ax = axs[0,1]
        colors4 = ['gold', 'darkorange', 'purple', 'violet']
        linetypes4 = ['-', (0,(5,3)), (3,(5,3)), (6,(5,3))]
        labels4c = ['Peak $u$', 'Mean $u$', 'Peak $f$', 'Mean $f$']
        for i in range(4,8):
            yvar = bump_amplitudes[i]
            ax.plot(inputs, yvar[neti,:], color=colors4[i-4], label=labels4c[i-4], linestyle=linetypes4[i-4])
        handles, labels = ax.get_legend_handles_labels()
        order = [0,2,1,3]
        ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=legendsize, ncol=2, loc=(0.03,0.7), columnspacing=0.5)
        ax.set_ylabel('Right $-$ Left')
        ax.set_xticks([-1, -0.5, 0, 0.5, 1], ['','','','',''])
        ax.text(-0.15, 1.2, 'B', fontsize=fontsize, fontweight='bold', va='top', ha='right', transform=ax.transAxes)

        ax = axs[1,0]
        yvar = bump_amplitudes[0]
        ax.plot(inputs, yvar[neti,:,orders[0]], '-k', label='$u$')
        ax.plot(inputs, yvar[neti,:,orders[1]], '--k', label='$f$')
        # align the legend to the lower center
        ax.legend(fontsize=legendsize, loc='center right', bbox_to_anchor=legendcloc)

        labels4c3 = ['Peak $u\:/\:f$', 'Mean $u$', 'Peak $u\:/\:f$', 'Mean $f$']
        clmtypes2 = [['-r.', '-b.', '-k.'], ['--r.', '--b.', '--k.']]
        cd = ['C', 'D']
        for i in range(4):
            ax = axs[1,i%2]

            if i == 3:
                ax = ax.twinx()
                ax.spines['top'].set_visible(False)
            if i < 2:
                ax.text(-0.15, 1.2, cd[i], fontsize=fontsize, fontweight='bold', va='top', ha='right', transform=ax.transAxes)
                ax.set_xlabel('$\Delta b(v)$', fontsize=fontsize-1)

            ax.set_ylabel(labels4c3[i], fontsize=fontsize)
            ax.set_xticks([-1, -0.5, 0, 0.5, 1])

            yvar = bump_amplitudes[i]
            for ringi in range(ring_num):
                ax.plot(inputs, yvar[neti,:,orders[ringi]], clmtypes2[i//2][ringi], label=labels3r[ringi])

        for i in range(2):
            ax = axs[2,i]
            ax.set_visible(False)

        # Bump u at different speed
        ids = compare_ids
        ax = axs[3,0]
        for i,ratioi in enumerate(ids):
            for ringi in range(ring_num):
                membrane = network_acvs_moving[neti,ratioi][orders[ringi],:,-1] + total_inputs[orders[ringi]][ratioi]
                shiftv = np.rint(cstat.mean(theta_range, membrane-min(membrane)) / (2*np.pi) * len(theta_range)).astype(int)
                membrane = np.roll(membrane, -shiftv)
                ax.plot(theta_range, membrane, color=colorsvr[ringi][i], label=f'{inputs[ratioi]}, {labels3r[ringi]}', linestyle=linetypes[ringi])
        ax.set_ylabel('$u$', rotation=0)
        ax.set_xlabel('                                                                             HD cells indexed by their preferred HDs')
        ax.set_xticks([-np.pi/3*2, 0, np.pi/3*2], ['$-120°$', '$0°$', '$120°$'])
        ax.text(-0.15, 1.2, 'E', fontsize=fontsize, fontweight='bold', va='top', ha='right', transform=ax.transAxes)


        # Bump f at different speed
        ax = axs[3,1]
        for i,ratioi in enumerate(ids):
            for ringi in range(ring_num):
                membrane = network_acvs_moving[neti,ratioi][orders[ringi],:,-1] + total_inputs[orders[ringi]][ratioi]
                f = actfun(membrane)
                shiftv = np.rint(cstat.mean(theta_range, f) / (2*np.pi) * len(theta_range)).astype(int)
                f = np.roll(f, -shiftv)
                ax.plot(theta_range, f, color=colorsvr[ringi][i], label=f'{inputs[ratioi]:.0f}, {labels3r[ringi]}', linestyle=linetypes[ringi])
        ax.set_ylabel(f'$f$', rotation=0)
        ax.set_xticks([-np.pi/3*2, 0, np.pi/3*2], ['$-120°$', '$0°$', '$120°$'])
        handles, labels = ax.get_legend_handles_labels()
        ordersvr = [0, ring_num, 1, 1+ring_num, 2, 5]
        ax.legend([handles[idx] for idx in ordersvr[:ring_num*2]], [labels[idx] for idx in ordersvr[:ring_num*2]], fontsize=legendsize, loc=(-0.65,0.5))
        ax.text(-0.15, 1.2, 'F', fontsize=fontsize, fontweight='bold', va='top', ha='right', transform=ax.transAxes)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.45, hspace=0.5)
        if round == 0:
            plt.show()
        else:
            plt.savefig(f'C:/Users/15824/OneDrive/文档/00-Master Thesis/figures/fig{figi}.png')
            plt.close()

def plot_acv_example_66_fig7(figi, network_acvs_moving, Vels, peaku, meanu, meanf, theta_range,
                            neti, figsize, compare_ids, actfun, b0, inputs, dpis, fontsize, legendsize):
    
    colorsb = ['greenyellow', 'limegreen', 'darkgreen']
    for round in range(2):
        fig, axs = plt.subplots(2, 2, figsize=figsize, dpi=dpis[round])

        for ax in axs.flatten():
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=fontsize-2)
            ax.title.set_fontsize(fontsize)
            ax.xaxis.label.set_fontsize(fontsize)
            ax.yaxis.label.set_fontsize(fontsize)

        # input - velocity
        ax = axs[0,0]
        ax.plot(inputs, Vels[neti], '-g')
        ax.set_ylabel('Turn/s')
        ax.text(-0.15, 1.2, 'A', fontsize=fontsize, fontweight='bold', va='top', ha='right', transform=ax.transAxes)

        # input - mean acv & peak acv
        ax = axs[0,1]
        ax.plot(inputs, peaku[neti,:], label='Peak $f$', color=colorsb[0], linestyle='-')
        ax.plot(inputs, meanf[neti,:], label='Mean $f$', color=colorsb[1], linestyle='--')
        ax.set_ylabel('Peak / mean $f$')
        ax.text(-0.15, 1.2, 'B', fontsize=fontsize, fontweight='bold', va='top', ha='right', transform=ax.transAxes)
        handles1, labels1 = ax.get_legend_handles_labels()
        ax2 = ax.twinx()
        ax2.spines['top'].set_visible(False)
        ax2.plot(inputs, meanu[neti,:], label='Mean $u$', color=colorsb[2], linestyle=':')
        ax2.set_ylabel('Mean $u$')
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1+handles2, labels1+labels2, fontsize=legendsize, loc=(0, 1), ncol=3, columnspacing=0.2)

        for i in range(2):
            ax = axs[0,i]
            ax.set_xticks([-1, -0.5, 0, 0.5, 1])
            ax.set_xlabel('$\Delta b(v)$', fontsize=fontsize-1)

        # Bump u & f at different speed
        labelstype = ['$u$', '$f$']
        cd = ['C', 'D']
        for uorf in range(2):
            ax = axs[1,uorf]
            for i,ratioi in enumerate(compare_ids):
                membrane = network_acvs_moving[neti,ratioi][:,-1] + b0
                if uorf == 1:
                    membrane = actfun(membrane)
                shiftv = np.rint(cstat.mean(theta_range, membrane-min(membrane)) / (2*np.pi) * len(theta_range)).astype(int)
                membrane = np.roll(membrane, -shiftv)
                ax.plot(theta_range, membrane, color=colorsb[i], label=f'{inputs[ratioi]}')
            ax.set_ylabel(labelstype[uorf], rotation=0)
            ax.set_xticks([-np.pi/3*2, 0, np.pi/3*2], ['$-120°$', '$0°$', '$120°$'])
            ax.text(-0.15, 1.2, cd[uorf], fontsize=fontsize, fontweight='bold', va='top', ha='right', transform=ax.transAxes)
            if uorf == 0:
                ax.set_xlabel('                                                                      HD cells indexed by their preferred HDs')
            if uorf == 1:
                ax.legend(fontsize=legendsize, loc=(-0.6,0.3))

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.6, hspace=1)
        if round == 0:
            plt.show()
        else:
            plt.savefig(f'C:/Users/15824/OneDrive/文档/00-Master Thesis/figures/fig{figi}.png')
            plt.close()

def par_titles_ax(ax, parnames, network_pars, neti):
        if len(parnames) == 2:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}')
        elif len(parnames) == 3:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}')
        elif len(parnames) == 4:
            ax.set_title(f'({neti})\n {parnames[0]}={network_pars[neti,0]:.1f}, {parnames[1]}={network_pars[neti,1]:.1f}\n \
{parnames[2]} = {network_pars[neti,2]:.2f}, {parnames[3]} = {network_pars[neti,3]:.1f}')

def plot_reverse_rotation(Vels, valid_index_linear_move, inputs, par_names, network_pars):
    vel_slopes = cal_vel_slope(Vels, inputs, valid_index_linear_move)
    index_neg_slope = np.where(vel_slopes < 0)[0]
    num_neg_slope = len(index_neg_slope)
    index_pos_slope = np.where(vel_slopes > 0)[0]
    num_pos_slope = len(index_pos_slope)
    print(f'NUM: Reverse rotation: {num_neg_slope}, Forward rotation: {num_pos_slope}')
    mean, std, maxv, minv = cal_statistics(vel_slopes[index_neg_slope])
    print(f'Reverse rotation slope: M={mean:6.3f}, SD={std:.3f}, Range=[{minv:6.3f}, {maxv:6.3f}]')
    if num_pos_slope > 0:
        mean, std, maxv, minv = cal_statistics(vel_slopes[index_pos_slope])
        print(f'Forward rotation slope: M={mean:6.3f}, SD={std:.3f}, Range=[{minv:6.3f}, {maxv:6.3f}]')
        ncol = min(5, num_pos_slope)
        fig, axs = plt.subplots(1, ncol, figsize=(3*ncol,2))
        for i in range(ncol):
            ax = axs[i] if ncol > 1 else axs
            ax.plot(inputs, Vels[index_pos_slope[i]].mean(axis=1), 'o-')
            par_titles_ax(ax, par_names, network_pars, index_pos_slope[i])
            ax.set_xlabel('Inputs')
            if i == 0:
                ax.set_ylabel('Vels')
        plt.show()

def print_match_result(network_acvs_moving, valid_index_linear_move, zeroid, match_fun):
    index_shape_mismatch, dev_shape_ratios, if_match = match_fun(network_acvs_moving, valid_index_linear_move, zeroid)
    print(f'{len(index_shape_mismatch)/len(valid_index_linear_move) * 100:.2f} % = Percent of shape not match')
    print(f'{np.mean(dev_shape_ratios):.2e} = Mean[(acv dif) / (max one neighboring acv dif)]')
    print(f'{np.std(dev_shape_ratios):.2e} = SD[(acv dif) / (max one neighboring acv dif)]')
    print(f'{np.max(dev_shape_ratios):.2e} = Max[(acv dif) / (max one neighboring acv dif)]')

def plot_phase_dif(cors, index_c, slopes, search_pars, network_pars, par_names, \
                   Phase_dif_names = ['Vel - Phase lag: Right - Left', 'Vel - Phase lag: Center - (Left + Right) / 2'], 
                   nowords='No phase difference'):
    figmag = [1.6, 1]
    for i in range(2):
        if np.isnan(cors[:,i]).all():
            print(f'{Phase_dif_names[i]:22}: {nowords}')
        else:
            print(f'{Phase_dif_names[i]}:')
            mean, std, maxv, minv = cal_statistics(cors[index_c, i])
            print(f'cor  : M={mean:6.3f}, SD={std:.3f}, Range=[{minv:6.3f}, {maxv:6.3f}]')
            mean, std, maxv, minv = cal_statistics(slopes[index_c, i])
            print(f'slope: M={mean:6.3f}, SD={std:.3f}, Range=[{minv:6.3f}, {maxv:6.3f}]')
            titles = [f'SLOPE: {Phase_dif_names[i]}', f'COR: {Phase_dif_names[i]}']
            vars_plot = [slopes[:,i], cors[:,i]]
            for j in range(2):
                phase_plot = oned2colormesh_general(vars_plot[j], search_pars, network_pars, parnames=par_names)
                plot_pars_on_property(phase_plot, search_pars, par_names, titles=titles[j], figmag=figmag[j])

def plot_net_vars_distribution(cors, index_c, netvars, Vels, inputs, \
                   Phase_dif_names = ['Right - Left', 'Central - rotational'], 
                   nowords='No phase difference', ylabel='Phase lag', bins=100):
    '''
    Plot the slope and correlation when using bump moving speed to predict phase difference / skewness

    cors/vars: 2D array, shape = (num networks, 2)
        2 dim: 
            phase difference: 0 is Right - Left, 1 is Center - (Left + Right) / 2
            skewness: 0 is right, 1 is central
    '''

    for i in range(2):
        if np.isnan(cors[:,i]).all():
            print(f'{Phase_dif_names[i]:22}{nowords}')
        else:
            fig, axs = plt.subplots(1, 2, figsize=(15,1.5))
            mean, std, maxv, minv = cal_statistics(cors[index_c, i])       
            print(f'{Phase_dif_names[i]:30}cor: M={mean:6.3f},  SD={std:.3f},  Range=[{minv:6.3f}, {maxv:6.3f}]')
            legends = [f'Velocity - {ylabel}', f'input - {ylabel}']
            vel_slope = cal_vel_slope(Vels, inputs, index_c)
            vars_plot = [netvars[index_c,i], netvars[index_c,i] * vel_slope[index_c]]
            for j in range(2):
                ax = axs[j]
                ax.hist(vars_plot[j], bins=bins, label=legends[j])
                mean, std, maxv, minv = cal_statistics(vars_plot[j])
                ax.set_title(f'M={mean:.3f} ({std:.3f})')
                ax.legend()
            plt.show()

def pars_for_plot_par_on_prop_with_phi(net_filei, include_phi=True):
    network_settings = pd.read_pickle(SIM_RESULT_PATH / 'network_v1_settings.pkl')
    ring_num = network_settings.loc[net_filei, 'Ring num']
    actfun_name = network_settings.loc[net_filei, 'Act Fun']
    weight_name = network_settings.loc[net_filei, 'Weight Fun']
    weight_if_same = network_settings.loc[net_filei, 'LR Weight']
    search_pars = network_settings.loc[net_filei, 'search_pars']
    if include_phi:
        search_pars['phi'] = np.array([i/9*np.pi for i in range(1, 9)])

    file_pre_name = network_settings.loc[net_filei, 'file_pre_name']
    weight_fun = getattr(network, network_settings.loc[net_filei, 'weightfun_'])
    actfun = getattr(network, network_settings.loc[net_filei, 'actfun_'])
    ## Generated parameters
    par_num = len(search_pars)
    par_names = list(search_pars.keys())

    [network_pars] = load_pickle(['pars'], weight_fun, actfun, file_pre_name)
    network_pars = np.array(network_pars) # to remove error
    if include_phi:
        network_pars2 = np.zeros((8, network_pars.shape[0], par_num))
        for i in range(1, 9):
            network_pars2[i-1, :, :-1] = network_pars
            network_pars2[i-1, :, -1] = i/9 * np.pi
        network_pars2 = network_pars2.reshape(-1, par_num)
    else:
        network_pars2 = network_pars

    return search_pars, network_pars2, par_names, ring_num, actfun_name, weight_name, weight_if_same

def plot_prop_on_par_with_phi(net_filei, refrac_v_nets_all, vv_slope_nets_all, figmag=2,
                              target1 = [0,0,0,0], target2 = [0,0,0,0], norm1=None, norm2=None, reverse_col_row_var=False):
    search_pars, network_pars2, par_names, ring_num, actfun_name, weight_name, weight_if_same = \
        pars_for_plot_par_on_prop_with_phi(net_filei)

    par_plot = oned2colormesh_general(refrac_v_nets_all[net_filei-4].flatten(), search_pars, network_pars2, zero2nan=False)
    valid_m = np.where(par_plot > 0)
    zero_ids = np.where(par_plot == 0)
    vmax = np.nanmax(par_plot)
    vmin = np.nanmin(par_plot[valid_m])
    pre_title = f'Ring No. {ring_num}; w: {weight_name} ({weight_if_same}); $f$: {actfun_name} \n'
    if norm1 is None:
        fig1 = plot_pars_on_property(par_plot, search_pars, par_names, titles=pre_title + 'Max velocity', 
                            ncol=8, nrow=None, figmag=figmag, par_present=None, vmin=vmin, target=target1, reverse_col_row_var=reverse_col_row_var)
    elif norm1 == 'log':
        norm1 = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        fig1 = plot_pars_on_property(par_plot, search_pars, par_names, titles=pre_title + 'Max velocity', 
                            ncol=8, nrow=None, figmag=figmag, par_present=None, norm=norm1, target=target1, reverse_col_row_var=reverse_col_row_var)

    par_plot = oned2colormesh_general(-vv_slope_nets_all[net_filei-4].flatten(), search_pars, network_pars2, zero2nan=False)
    par_plot[zero_ids] = 1e-10
    vmax = np.nanmax(par_plot)
    vmin = np.nanmin(par_plot[valid_m])
    # print(vmin)
    if norm2 is None:
        fig2 = plot_pars_on_property(par_plot, search_pars, par_names, titles=pre_title + 'Gain', 
                            ncol=8, nrow=None, figmag=figmag, par_present=None, vmin=vmin, target=target2, reverse_col_row_var=reverse_col_row_var)
    elif norm2 == 'log':
        norm2 = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        fig2 = plot_pars_on_property(par_plot, search_pars, par_names, titles=pre_title + 'Gain', 
                                ncol=8, nrow=None, figmag=figmag, par_present=None, target=target2, norm=norm2, reverse_col_row_var=reverse_col_row_var)
        
    return fig1, fig2
        
def plot_prop_on_5par_with_phi(net_filei, refrac_v_nets_all, vv_slope_nets_all, figmag=2,
                              target1 = [0,0,0,0], target2 = [0,0,0,0], norm1=None, norm2=None, reverse_col_row_var=False):

    network_settings = pd.read_pickle(SIM_RESULT_PATH + '/network_v1_settings.pkl')
    ring_num = network_settings.loc[net_filei, 'Ring num']
    actfun_name = network_settings.loc[net_filei, 'Act Fun']
    weight_name = network_settings.loc[net_filei, 'Weight Fun']
    weight_if_same = network_settings.loc[net_filei, 'LR Weight']
    search_pars = network_settings.loc[net_filei, 'search_pars']

    file_pre_name = network_settings.loc[net_filei, 'file_pre_name']
    weight_fun = getattr(network, network_settings.loc[net_filei, 'weightfun_'])
    actfun = getattr(network, network_settings.loc[net_filei, 'actfun_'])
    ## Generated parameters
    par_num = len(search_pars)
    par_names = list(search_pars.keys())

    [network_pars] = load_pickle(['pars'], weight_fun, actfun, file_pre_name)

    for i_phi in range(1, 9):
        par_plot = oned2colormesh_general(refrac_v_nets_all[net_filei-4][i_phi-1], search_pars, network_pars, zero2nan=False)
        valid_m = np.where(par_plot > 0)
        if len(valid_m[0]) == 0:
            continue
        zero_ids = np.where(par_plot == 0)
        vmax = np.nanmax(par_plot)
        vmin = np.nanmin(par_plot[valid_m])
        if i_phi == 1:
            title = f'Net {net_filei}, Ring {ring_num}| W: {weight_name}({weight_if_same})| f: {actfun_name} \n phi={i_phi/9*np.pi:.1f} Refraction V'
        else:
            title = f'phi={i_phi/9*np.pi:.1f} Refraction V'
        if norm1 is None:
            plot_pars_on_property(par_plot, search_pars, par_names, titles=title,
                                ncol=8, nrow=None, figmag=figmag, par_present=None, vmin=vmin, target=target1, reverse_col_row_var=reverse_col_row_var)
        elif norm1 == 'log':
            norm1 = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            plot_pars_on_property(par_plot, search_pars, par_names, titles=title,
                                ncol=8, nrow=None, figmag=figmag, par_present=None, norm=norm1, target=target1, reverse_col_row_var=reverse_col_row_var)
        
        par_plot = oned2colormesh_general(-vv_slope_nets_all[net_filei-4][i_phi-1], search_pars, network_pars, zero2nan=False)
        par_plot[zero_ids] = 1e-10
        vmax = np.nanmax(par_plot)
        vmin = np.nanmin(par_plot[valid_m])
        if norm2 is None:
            plot_pars_on_property(par_plot, search_pars, par_names, titles=f'b2v Slope', 
                                ncol=8, nrow=None, figmag=figmag, par_present=None, vmin=vmin, target=target2, reverse_col_row_var=reverse_col_row_var)
        elif norm2 == 'log':
            norm2_temp = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            plot_pars_on_property(par_plot, search_pars, par_names, titles=f'b2v Slope', 
                                    ncol=8, nrow=None, figmag=figmag, par_present=None, target=target2, norm=norm2_temp, reverse_col_row_var=reverse_col_row_var)