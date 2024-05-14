from gemlib.abstarct.basefunctionality import BasePlotter
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

class Drilldownplot(BasePlotter):
    def run(self):
        sns.set(font_scale=0.5)
        kws = dict(s=5, linewidth=.5, edgecolor="w")
        g = sns.FacetGrid(self.df, col=self.d1, row=self.d2, hue=self.d3)
        ext = str(self.d1) + '_' + str(self.d2) + '_' + str(self.d3)
        if self.fun == 'hist':
            g = (g.map(plt.hist, self.x).add_legend())
            self.plt = g
            dirpath = self.dirpath + 'drilldown_hist_' + self.x + '_' + ext + '.png'
            self.save(dirpath)
        elif self.fun == 'scatter':
            g = (g.map(plt.scatter, self.x, self.y, **kws).add_legend())
            self.plt = g
            dirpath = self.dirpath / Path('drilldown_scatter_' + self.x + '_vs_' + self.y + '_' + ext + '.png')
            self.save(dirpath)
        return
