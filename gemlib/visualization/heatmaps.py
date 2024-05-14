import seaborn as sns
from gemlib.abstarct.basefunctionality import BasePlotter
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import gemlib.validation.utilities as utils
import os
from pathlib import Path


class HeatMaps(BasePlotter):

    def run(self, dpi=None):
        if self.columns is not None:
            self.df = self.df[self.columns]
            self.remove_na()
        h = 100
        if len(self.df) < 50:
            h = 6
        if self.fontsize:
            font_size = self.fontsize

        self.plt = plt
        fig = self.plt.figure(figsize=(6, 6))

        font_size = 12
        font_scale = 0.8
        sns.set(font_scale=font_scale)
        ax = sns.heatmap(self.df,
                         vmin=self.min_v,
                         vmax=self.max_v,
                         annot=True,
                         fmt=".1f",
                         cmap="Blues",
                         cbar=False)

        ax.tick_params(axis='x')
        ax.tick_params(axis='y')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=font_size * font_scale)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=font_size * font_scale)
        ax.set_xlabel(ax.get_xlabel(), fontsize=font_size)
        ax.set_ylabel(ax.get_ylabel(), fontsize=font_size)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.set_title(self.title,
                     loc='center',
                     fontsize=font_size + 1,
                     y=1.12)
        fig.tight_layout()

        filename = self.dirpath / Path(self.title + '.png')

        self.save(filename, dpi=dpi)
        self.plt.close("all")


class CustomHeatMaps(BasePlotter):

    def std_y_bounds(self):
        y_bins = np.arange(self.ticks_min_y, self.ticks_max_y, self.bins)
        y_indices = np.arange(len(y_bins))
        return (y_bins, y_indices)

    def std_x_bounds(self):
        x_bins = np.arange(self.ticks_min_x, self.ticks_max_x, self.bins)
        x_indices = np.arange(len(x_bins) - 1)
        return (x_bins, x_indices)

    def get_matrix(self, df):
        (y_bins, y_indices) = self.std_y_bounds()
        (x_bins, x_indices) = self.std_x_bounds()

        subset = df[df[self.z] < self.max_v]

        X_index = np.digitize(subset[self.x], x_bins)
        Y_index = np.digitize(subset[self.y], y_bins)

        U = pd.pivot_table(subset, values=self.z, index=Y_index,
                           columns=X_index, aggfunc=np.mean)
        U = U.reindex(index=y_indices, columns=x_indices, fill_value=None)
        return np.flipud(np.array(U, dtype=float))

    def plot_matrix(self, matrix, filename=''):
        (y_bins, y_indices) = self.std_y_bounds()
        (x_bins, x_indices) = self.std_x_bounds()

        matplotlib.rc("figure", figsize=self.figsize)
        self.plt = plt
        self.plt.matshow(matrix, vmin=self.min_v, vmax=self.max_v)

        if self.annot:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    text = self.plt.text(j, i, np.round(matrix[i, j], 1),
                                         ha="center", va="center", color="w", fontsize=8)

        self.plt.title(self.title)
        self.plt.xticks(x_indices, x_bins, rotation=90)
        self.plt.xlabel(self.x_label, labelpad=10)
        self.plt.yticks(y_indices, y_bins * -1)
        self.plt.ylabel(self.y_label, labelpad=10)
        self.plt.colorbar(shrink=0.83, pad=0.02)
        self.plt.tick_params(axis='both', which='major',
                             labelsize=10, labelbottom=True,
                             labeltop=False)
        filename = self.dirpath / Path(self.title + '_' + filename + '.png')
        self.save(filename)
        plt.close("all")

    def run(self):
        if self.groupby:
            for g, df in self.df.groupby(self.groupby):
                matrix = self.get_matrix(df)
                g = utils.resolve_string_tuple(g)
                self.plot_matrix(matrix=matrix, filename=g)
        else:
            matrix = self.get_matrix(self.df)
            self.plot_matrix(matrix)

