import seaborn as sns
from gemlib.abstarct.basefunctionality import BasePlotter
from pathlib import Path


class ScatterPlotMatrix(BasePlotter):
    def __init__(self, columns, target, s):
        BasePlotter.__init__(self, columns=columns, target=target, s=s)

    def run(self):
        self.remove_na()
        sns.set()
        self.plt = sns.pairplot(data=self.df, vars=self.columns, hue=self.target,
                                plot_kws=dict(s=self.s), dropna=True)
        dirpath = self.dirpath / 'scattermatrix.png'
        if self.target is not None:
            dirpath = self.dirpath / Path('scattermatrix_' + self.target + '.png')
        self.save(dirpath)
