#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
from lyanna.utils import *

# plt.rcParams['font.size']       = 16
# plt.rcParams['lines.linewidth'] = 1.2
# plt.rcParams['text.usetex']     = False



class CornerTriangularPlot:
    """
    Plotting a scatter of hyperparameter trials in a triangular corner plot. May be modified to extended to other corner plots as well.
    """
    def __init__(
            self,
            param_keys:list, # list of strings representing the parameters
            param_labels:list, # list of strings representing the parameter labels
            figsize:tuple = (15,18), # figure size
            font_kwargs:dict = {},
    ): 
        self.param_keys   = param_keys
        self.param_labels = param_labels
        # self.dataframe    = dataframe
        self.N_params     = len(param_keys)
        self.figsize      = figsize
        self.N_rows       = self.N_params - 1
        self.N_cols       = self.N_params - 1
        self.fig = plt.figure(figsize = self.figsize)
        self.axs = self.fig.subplots(self.N_rows, self.N_cols, sharex='col', sharey='row')
        self.font_kwargs  = font_kwargs

        self.pointer_matrix = np.zeros((self.N_rows, self.N_cols), dtype = object)*np.NaN
        for row_num in range(self.N_rows):
            for col_num in range(self.N_cols):
                param_x = col_num
                param_y = row_num + 1
                if param_y<=param_x:
                    self.axs[row_num, col_num].set_axis_off()
                else:
                    self.pointer_matrix[row_num, col_num] = (param_x, param_y)
                    if param_x == 0:
                        self.axs[row_num, col_num].set_ylabel(self.param_labels[param_y], **font_kwargs)
                    if param_y == self.N_params - 1:
                        self.axs[row_num, col_num].set_xlabel(self.param_labels[param_x], **font_kwargs)
        plt.subplots_adjust(wspace = 0.05, hspace = 0.05)  
        # return self.fig, self.axs, self.pointer_matrix
    
    def plot(
        self,
        dataframe, # dataframe or dictionary of the data to plot, with param_keys as the keys 
        plot_type = 'scatter', # type of plot to make
        **kwargs, # keyword arguments to pass onto pyplot
    ):
        plots = np.zeros((self.N_rows, self.N_cols), dtype = object)
        for row_num in range(self.N_rows):
            for col_num in range(self.N_cols):
                if type(self.pointer_matrix[row_num, col_num]) == tuple:
                    param_x = col_num
                    param_y = row_num + 1
                    if plot_type == 'scatter':
                        p = self.axs[row_num, col_num].scatter(
                            dataframe[self.param_keys[param_x]], 
                            dataframe[self.param_keys[param_y]],  
                            **kwargs,
                        )
                    plots[row_num, col_num] = p
        return plots 
    
    def configure_plot(
        self,
        search_space,
        xtick_rotate = 45, # rotation of the x ticks
        ytick_rotate = 45, # rotation of the y ticks
        show_priors  = True, # whether to show the priors
        prior_border_kwargs = {'color': 'darkgreen', 'linewidth': 0.5, 'alpha': 0.3}, # keyword arguments for the prior border
        filled_priors = True, # whether to fill the priors
        prior_fill_kwargs = {'color': 'yellowgreen', 'alpha': 0.1}, # keyword arguments for the prior fill
    ):
        for i in range(self.N_rows):
            for j in range(self.N_cols):
                if type(self.pointer_matrix[i,j])==tuple:
                    param_x, param_y = self.pointer_matrix[i,j]
                    if show_priors:
                        self.axs[i,j].axhline(search_space[self.param_keys[param_y]]['low'],  **prior_border_kwargs)
                        self.axs[i,j].axhline(search_space[self.param_keys[param_y]]['high'], **prior_border_kwargs)
                        self.axs[i,j].axvline(search_space[self.param_keys[param_x]]['low'],  **prior_border_kwargs)
                        self.axs[i,j].axvline(search_space[self.param_keys[param_x]]['high'], **prior_border_kwargs)
                        if filled_priors:
                            self.axs[i,j].fill_between( 
                                [search_space[self.param_keys[param_x]]['low'], search_space[self.param_keys[param_x]]['high']],
                                search_space[self.param_keys[param_y]]['low'],
                                search_space[self.param_keys[param_y]]['high'],
                                **prior_fill_kwargs,
                            )
                    self.axs[i,j].tick_params("x", rotation = xtick_rotate)
                    self.axs[i,j].tick_params("y", rotation = ytick_rotate)
                    if search_space[self.param_keys[param_x]]['log']:
                        self.axs[i,j].set_xscale('log')
                    if search_space[self.param_keys[param_y]]['log']:
                        self.axs[i,j].set_yscale('log')

    def savefig(self, filepath:str, **kwargs):
        plt.savefig(filepath, **kwargs)

    def show(self):  
        plt.show()



class SmoothingLearningCurves:

    def __init__(self, LOG, metrics, kernel_size):
        self.LOG         = LOG
        self.metrics     = metrics
        self.kernel_size = kernel_size  # for the gaussian case, kernel_size is sigma in epochs

    def top_hat_smooth(self):
        epochs       = self.LOG['epoch']
        len_smooth   = len(epochs) - self.kernel_size + 1
        SMOOTH       = {}
        for m in self.metrics:
            values   = self.LOG[m]
            smooth   = np.zeros(len_smooth)
            for i in range(len(smooth)):
                smooth[i]  = np.mean(values[i : i+self.kernel_size])
            SMOOTH[m] = smooth
        SMOOTH['epoch']    = epochs[self.kernel_size//2 : self.kernel_size//2 + len(smooth)]
        return SMOOTH
    
    def top_hat_minimum(self):
        epochs       = self.LOG['epoch']
        len_smooth   = len(epochs) - self.kernel_size + 1
        SMOOTH       = {}
        for m in self.metrics:
            values   = self.LOG[m]
            smooth   = np.zeros(len_smooth)
            for i in range(len(smooth)):
                smooth[i]  = np.min(values[i : i+self.kernel_size])
            SMOOTH[m] = smooth
        SMOOTH['epoch']    = epochs[self.kernel_size//2 : self.kernel_size//2 + len(smooth)]
        return SMOOTH

    def gaussian_smooth(self): 
        epochs       = self.LOG['epoch']
        len_kernel   = 6 * self.kernel_size
        x = np.linspace(-3,3,len_kernel)
        kernel       = np.exp(-x**2/2)
        len_smooth   = len(epochs) - len_kernel + 1
        SMOOTH       = {}
        for m in self.metrics:
            values   = self.LOG[m]
            smooth   = np.zeros(len_smooth)
            for i in range(len(smooth)):
                smooth[i]  = np.average(values[i : i+len_kernel], weights = kernel)
            SMOOTH[m] = smooth
        SMOOTH['epoch']    = epochs[len_kernel//2 : len_kernel//2 + len(smooth)]
        return SMOOTH
    
    def __call__(self, kernel_type = 2):
        if kernel_type == 1:
            return self.top_hat_smooth()
        elif kernel_type ==2:
            return self.gaussian_smooth()
        else:
            raise ValueError("Invalid kernel type")




def plot_posterior_contours_from_chains(
        chains:dict, 
        param_labels:tuple = (rel_T0_string, rel_gamma_string),
        truth:list    = [0.0, 0.0],
        figsize:tuple = (6.5, 6.5),
        usetex:bool   = True,
        summary:bool  = False,
        kde:bool|float  = False, 
        flip:bool     = False,
        legend:bool   = True,
        extents:list  = None,
        colors:list   = ['red', 'green', 'blue', 'orange',],
        font_sizes:list   = [19,19],
        bar_shades:list   = True,
        shade_alphas:list = 0.2,
        plot_means:bool   = True,
        plot_hists:bool   = True,
        legend_kwargs     = None,
    ):
    c = ChainConsumer()
    for chain in chains:
        c.add_chain(
            chains[chain],
            parameters = [*param_labels],
            name       = chain,
        )
    c.configure(
        usetex   = usetex,
        summary  = summary,
        kde      = kde,
        colors   = colors,
        tick_font_size  = font_sizes[0],
        label_font_size = font_sizes[1],
        bar_shade = bar_shades,
        shade_alpha     = shade_alphas,
        plot_hists      = plot_hists,
        legend_kwargs   = legend_kwargs,
        flip            = flip,
    )
    fig = c.plotter.plot(
        figsize = figsize,
        truth   = truth,
        legend  = legend,
        extents = extents,
    )
    if plot_means:
        if plot_hists:
            ax = fig.axes[2]
        else:
            ax = fig.axes[0]
        for i, chain in enumerate(chains):
            ax.scatter(
                *np.mean(chains[chain], axis = 0),
                s = 100, color = colors[i], marker = 'x',
            )
    c.plotter.restore_rc_params()
    return fig