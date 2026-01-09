import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from PIL import Image

import HD_utils.plot as myplot

figtitle_fontsize = 14
label_fontsize = 12
tick_fontsize = 10

def plot_3par_on_prop1(net_filei, par_plot, par_present, ncol, nrow, figsize, norm, cmap='viridis', figtitle_appendix=''):
    
    search_pars, network_pars2, par_names, ring_num, actfun_name, weight_name, weight_if_same = myplot.pars_for_plot_par_on_prop_with_phi(net_filei)
    weight_if_same = 'same' if weight_if_same == 'Same' else 'different'

    valid_m = np.where(par_plot > 0)
    vmax = np.nanmax(par_plot)
    vmin = np.nanmin(par_plot[valid_m])

    pre_title = f'Ring No. {ring_num}; w: {weight_name} ({weight_if_same}); $f$: {actfun_name} \n'
    figtitle = pre_title + figtitle_appendix

    cmap = plt.get_cmap(cmap).copy()
    cmap.set_under(color='grey')
    cmap.set_over(color='pink')

    fig, axs = plt.subplots(nrow, ncol, figsize=figsize, sharex=True, sharey=True)
    fig.suptitle(figtitle, fontsize=figtitle_fontsize)
    axf = axs.flatten()
    for i, par2 in enumerate(search_pars[par_names[2]]):
        if i >= ncol:
            continue
        ax = axf[i]
        ax.set_title(f'{par_present[2]} = {par2*180/np.pi:.0f}°')
        ax.set_aspect('equal')
        if norm is None:
            im = ax.pcolormesh(search_pars[par_names[0]], search_pars[par_names[1]], par_plot[i], 
                            cmap=cmap, vmin=vmin, vmax=vmax)
        elif norm == 'log':
            norm_object = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            im = ax.pcolormesh(search_pars[par_names[0]], search_pars[par_names[1]], par_plot[i], 
                            cmap=cmap, norm=norm_object)
        if i // ncol == nrow - 1:
            ax.set_xlabel(f'{par_present[0]}', fontsize=label_fontsize)
            ax.set_xticks(np.linspace(search_pars[par_names[0]][0], search_pars[par_names[0]][-1], 3))
        if i % ncol == 0:
            ax.set_ylabel(f'{par_present[1]}', fontsize=label_fontsize)
            ax.set_yticks(np.linspace(search_pars[par_names[1]][0], search_pars[par_names[1]][-1], 3))
        ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    cbar = fig.colorbar(im, ax=axs, extend='min', aspect= 20 * ( nrow / ncol ), shrink=1)
    
    return fig


def plot_4par_on_prop1(net_filei, par_plot, par_present, ncol, nrow, figsize, norm, 
                       cmap='viridis', figtitle_appendix='', reverse_col_row_var=False, vmax=None, vmin=None,
                       row_var_format=None, col_var_format=None, include_phi=True, plot_fig=False):
    
    search_pars, network_pars2, par_names, ring_num, actfun_name, weight_name, weight_if_same = myplot.pars_for_plot_par_on_prop_with_phi(net_filei, include_phi=include_phi)
    weight_if_same = 'same' if weight_if_same == 'Same' else 'different'

    valid_m = np.where(par_plot > 0)
    vmax = np.nanmax(par_plot) if vmax == None else vmax
    vmin = np.nanmin(par_plot[valid_m]) if vmin == None else vmin

    pre_title = f'Ring Number. {ring_num}; w: {weight_name} ({weight_if_same}); $f$: {actfun_name} \n'
    figtitle = pre_title + figtitle_appendix

    cmap = plt.get_cmap(cmap).copy()
    cmap.set_under(color='grey')
    cmap.set_over(color='pink')

    if reverse_col_row_var:
        par_plot = np.swapaxes(par_plot, 0, 1)

        par_names_temp = par_names.copy()
        par_names = par_names_temp.copy()
        par_names[2] = par_names_temp[3]
        par_names[3] = par_names_temp[2]

        par_present_temp = par_present.copy()
        par_present = par_present_temp.copy()
        par_present[2] = par_present_temp[3]
        par_present[3] = par_present_temp[2]

        par2_list = search_pars[par_names[2]]
        par3_list = search_pars[par_names[3]][::-1]
    else:
        par2_list = search_pars[par_names[2]][::-1]
        par3_list = search_pars[par_names[3]]


    fig, axs = plt.subplots(nrow, ncol, figsize=figsize, sharex=True, sharey=True)
    fig.suptitle(figtitle, fontsize=figtitle_fontsize)
    for i, par2 in enumerate(par2_list): 
        if i >= nrow:
            break
        for j, par3 in enumerate(par3_list): 
            if j >= ncol:
                break
            ax = axs[i,j]
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
            ax.set_aspect('equal')
            if i == 0:
                if col_var_format == 'f':
                    ax.set_title(f'{par_present[3]}\n{par3:.1f}\n', fontsize=label_fontsize)
                elif col_var_format == '-180':
                    ax.set_title(f'{par_present[3]}\n{par3*180/np.pi-180:.0f}°\n', fontsize=label_fontsize)
                else:
                    ax.set_title(f'{par_present[3]}\n{par3*180/np.pi:.0f}°\n', fontsize=label_fontsize)
            if norm is None:
                im = ax.pcolormesh(search_pars[par_names[0]], search_pars[par_names[1]], par_plot[j, i], 
                            cmap=cmap, vmin=vmin, vmax=vmax)
            elif norm == 'log':
                norm_object = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
                im = ax.pcolormesh(search_pars[par_names[0]], search_pars[par_names[1]], par_plot[j, i], 
                            cmap=cmap, norm=norm_object)
            if i == nrow - 1:
                ax.set_xlabel(par_present[0], fontsize=label_fontsize)
                ax.set_xticks(np.linspace(search_pars[par_names[0]][0], search_pars[par_names[0]][-1], 3))
            if j == 0:
                if row_var_format == 'd':
                    ax.set_ylabel(f'{par_present[2]}\n{par2:.0f}\n\n{par_present[1]}', fontsize=label_fontsize)
                else:
                    ax.set_ylabel(f'{par_present[2]}\n{par2:.1f}\n\n{par_present[1]}', fontsize=label_fontsize)
                ax.set_yticks(np.linspace(search_pars[par_names[1]][0], search_pars[par_names[1]][-1], 3))
    
    plt.tight_layout()
    cbar = fig.colorbar(im, ax=axs, extend='min')
    cbar.ax.tick_params(labelsize=16)
    if plot_fig:
        plt.show()
    
    return fig

def combine_four_pngs(file1s, output_file):
    # Open the images
    imgs = []
    for i in range(len(file1s)):
        img_temp = Image.open(file1s[i])
        imgs.append(img_temp)
    
    # Get dimensions (use the maximum dimensions of the images)
    width, height = imgs[0].size
    for img in imgs[1:]:
        width = max(width, img.size[0])
        height = max(height, img.size[1])
    
    # Create a new image with double width and height
    combined = Image.new('RGBA', (width * 2, height * 2))
    
    # Paste images in 2x2 grid
    combined.paste(imgs[0], (0, 0))
    combined.paste(imgs[1], (width, 0))
    combined.paste(imgs[2], (0, height))
    try:
        combined.paste(imgs[3], (width, height))
    except:
        pass # if there is only 3 images, leave the right bottom empty
    # Save the result
    combined.save(output_file)
    
    
def combine_eight_pngs(file_list, output_file):
    # Open the images
    imgs = []
    for i in range(8):
        img_temp = Image.open(file_list[i])
        imgs.append(img_temp)
    
    # Get dimensions (use the maximum dimensions of the images)
    width, height = imgs[0].size
    for img in imgs[1:]:
        width = max(width, img.size[0])
        height = max(height, img.size[1])
    
    # Create a new image with 3x3 grid dimensions
    combined = Image.new('RGBA', (width * 3, height * 3))
    
    # Paste images in 3x3 grid (leaving bottom-right empty)
    # Row 1
    combined.paste(imgs[0], (0, 0))
    combined.paste(imgs[1], (width, 0))
    combined.paste(imgs[2], (width * 2, 0))
    
    # Row 2
    combined.paste(imgs[3], (0, height))
    combined.paste(imgs[4], (width, height))
    combined.paste(imgs[5], (width * 2, height))
    
    # Row 3 (bottom-right position remains empty)
    combined.paste(imgs[6], (0, height * 2))
    combined.paste(imgs[7], (width, height * 2))
    # Position (width * 2, height * 2) remains empty
    
    # Save the result
    combined.save(output_file)