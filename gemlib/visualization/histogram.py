import numpy as np
import matplotlib.pyplot as plt
from gemlib.abstarct.basefunctionality import BasePlotter
import gemlib.validation.utilities as utils
import os
from matplotlib.ticker import  MultipleLocator, FormatStrFormatter
from pathlib import Path


class HistPlot(BasePlotter):

    def run(self):
        if self.groupby:
            for g, df in self.df.groupby(self.groupby):
                g = utils.resolve_string_tuple(g)
                self.single_plot(df=df, filename=g)
        else:
            self.single_plot(self.df, '')

    def single_plot(self, df, filename):
        fig = plt.figure(figsize=self.figsize)
        b =self.bins
        if self.binsize:
            b1 = np.arange(0, np.max(df[self.columns].values) + self.binsize, self.binsize)
            b2 = np.arange(-self.binsize, np.min(df[self.columns].values) - self.binsize, -self.binsize)
            b = np.concatenate((np.flip(b2, axis=0),b1))
            ax = df[self.columns].plot(kind='hist',
                                       bins=b,
                                       alpha=self.alpha,
                                       histtype='bar', rwidth=0.8)
        else:
            ax = df[self.columns].plot(kind='hist',
                                       bins=b,
                                       alpha=self.alpha)
        ax.set_title(self.title)
        ax.set_ylabel(self.y_label)
        ax.set_xlabel(self.x_label)
        ax.set_xlim(self.ticks_min_x,self.ticks_max_x)
        ax.set_ylim(self.ticks_min_y,self.ticks_max_y)
        if self.binsize:
            # ax.set_xticks(b, minor=True)
            minorLocator = MultipleLocator(self.binsize)
            ax.xaxis.set_minor_locator(minorLocator)
            majorLocator = MultipleLocator(self.binsize * 5)
            majorFormatter = FormatStrFormatter('%d')
            ax.xaxis.set_major_locator(majorLocator)
            ax.xaxis.set_major_formatter(majorFormatter)

        if self.fontsize:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(self.fontsize)
            ax.legend(prop={'size': self.fontsize - 2})

        if self.legend_names:
            ax.legend(self.legend_names)
        self.plt = plt
        filename = self.dirpath / Path(self.title + '_' + filename + '.png')
        self.save(filename)
        plt.close('all')
