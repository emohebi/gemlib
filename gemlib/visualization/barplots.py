import numpy as np
import matplotlib.pyplot as plt
from gemlib.abstarct.basefunctionality import BasePlotter
import matplotlib
import sys
from pathlib import Path


class BarPlotting(BasePlotter):

    def run(self):
        if self.columns:
            self.df = self.df[self.columns]
        if self.trans:
            self.df = self.df.T.rename({0: ' '}, axis='columns')
            ax = self.df.plot(kind='bar', figsize=self.figsize, colormap='Fovio')
        else:
            self.df = self.df.rename({'0': ' '}, axis='columns')
            ax = self.df.plot(kind='bar', figsize=self.figsize, colormap='Fovio')

        for p in ax.patches:
            ax.annotate(str(p.get_height()),
                        (p.get_x() + p.get_width() / 2.,
                         p.get_height()),
                        ha='center',
                        va='center',
                        xytext=(0, 5),
                        textcoords='offset points', fontsize=4)
        ax.set_xlabel(self.x_label, labelpad=10)
        ax.set_ylabel(self.y_label, labelpad=10)
        ax.set_title(self.title)

        if self.figsize:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(self.fontsize)
            ax.legend(prop={'size': self.fontsize - 2})
        self.plt = ax.get_figure()
        dirpath = self.dirpath / Path(self.title + '.png')
        csv_path = self.dirpath / Path(self.title + '.csv')
        self.df.to_csv(csv_path)
        self.save(dirpath)
        plt.close('all')

    def bar(self, x_labels, y_values, filename, dirpath, title=None,
            rot_angle=None, xlabel=None, ylabel=None):

        fig = plt.figure(figsize=(2, 1))
        self.fontsize = 3.5
        pad_size = 1
        color = "blue"
        plt.tick_params(axis='both', pad=pad_size)
        index = np.arange(len(x_labels))
        plt.bar(index, y_values, color=color)
        plt.xticks(index, x_labels, rotation=90, fontsize=self.fontsize)
        y_ticks = np.arange(self.ticks_min_y, self.ticks_max_y, 10)
        plt.yticks(y_ticks, fontsize=self.fontsize)
        plt.ylabel(ylabel, fontsize=self.fontsize, labelpad=pad_size)
        plt.xlabel(xlabel, fontsize=self.fontsize, labelpad=pad_size)

        filename = dirpath / Path(filename + '.png')
        fig.savefig(filename, dpi=500, bbox_inches='tight')
        print('file {0} saved.'.format(filename))
        plt.close('all')



def bar():
    pass
