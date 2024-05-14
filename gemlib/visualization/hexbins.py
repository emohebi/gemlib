import matplotlib.pyplot as plt
import numpy as np
from gemlib.abstarct.basefunctionality import BasePlotter
from pathlib import Path


class HexbinPlot(BasePlotter):
    def __init__(self, columns, target):
        BasePlotter.__init__(self, columns=columns, target=target)

    def run(self):
        self.remove_na()
        if self.target is not None:
            classes = self.df[self.target].dropna().unique().tolist()
            fig, axes = plt.subplots(ncols=len(classes), nrows=(len(self.columns) * (len(self.columns) - 1) / 2),
                                     figsize=(len(classes) * 9, (len(self.columns) * (len(self.columns) - 1) / 2) * 7)) #
        n = -1
        for i in range(0, len(self.columns)):
            for j in range(i + 1, len(self.columns)):
                x_min = np.min(self.df[self.columns[i]])
                x_max = np.max(self.df[self.columns[i]])
                y_min = np.min(self.df[self.columns[j]])
                y_max = np.max(self.df[self.columns[j]])
                if self.target is None:
                    plt.figure()
                    xt = self.df[self.columns[i]].values
                    yt = self.df[self.columns[j]].values
                    hb = plt.hexbin(xt, yt, gridsize=30, cmap=self.colors[0], extent=[x_min, x_max, y_min, y_max],
                                    mincnt=1)
                    cb = plt.colorbar(hb)
                    cb.set_label('counts')
                    plt.xlabel(self.columns[i])
                    plt.ylabel(self.columns[j])
                    path = self.dirpath + 'hexbins_' + self.columns[i] + '_vs_' + self.columns[j] + '.png'
                    plt.savefig(path, dpi=600)
                    print('file {0} saved.'.format(path))
                    continue
                max_hexbinbar = 0
                for c in classes:
                    xt = self.df.loc[self.df[self.target] == c, self.columns[i]]
                    yt = self.df.loc[self.df[self.target] == c, self.columns[j]]
                    h, xedge, yedge = np.histogram2d(xt, yt, bins=30)
                    max_hexbinbar = max(np.max(h), max_hexbinbar)
                for c in classes:
                    n = n + 1
                    row = n // len(classes)
                    col = n % len(classes)
                    if len(self.columns) > 2:
                        ax = axes[row, col]
                    else:
                        ax = axes[n]
                    xt = self.df.loc[self.df[self.target] == c, self.columns[i]]
                    yt = self.df.loc[self.df[self.target] == c, self.columns[j]]
                    hb = ax.hexbin(xt, yt, gridsize=30, cmap=self.colors[0], extent=[x_min, x_max, y_min, y_max],
                                   mincnt=1)
                    cb = fig.colorbar(hb, ax=ax)
                    cb.set_label(c)
                    cb.set_clim(vmin=1, vmax=max_hexbinbar)
                    ax.set_xlabel(self.columns[i])
                    ax.set_ylabel(self.columns[j])
        if self.target is not None:
            path = self.dirpath / Path('hexbins_' + self.target + '.png')
            fig.savefig(path, dpi=600)
            print('file {0} saved.'.format(path))
