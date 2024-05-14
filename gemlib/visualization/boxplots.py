import seaborn as sns
from gemlib.abstarct.basefunctionality import BasePlotter
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class BoxPlotting(BasePlotter):
    def run(self):
        if self.columns is not None and self.x is None:
            self.df = self.df[self.columns]
            self.remove_na()
        if not isinstance(self.y, list):
            self.y = [self.y]
        for y in self.y:
            plt.figure(figsize=(6, 6))
            ax = sns.boxplot(x=self.x, y=y, hue=self.z, data=self.df, fliersize=1,
                             linewidth=0.35, order=self.order)
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=5)

            self.add_meta_and_save(ax, y)


    def add_meta_and_save(self, ax, y):
        medians = self.df.groupby([self.x])[y].median().values
        median_labels = [str(np.round(s, 2)) for s in medians]

        pos = range(len(medians))
        for tick, label in zip(pos, ax.get_xticklabels()):
            ax.text(pos[tick], medians[tick] - 0.0, median_labels[tick],
                    horizontalalignment='center', size='5', color='b')

        self.plt = ax.get_figure()
        dirpath = self.dirpath / Path(self.x + '_vs_' + y + '_vs_' + str(self.z) + '_boxplot.png')
        self.save(dirpath)
