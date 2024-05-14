import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
from gemlib.abstarct.basefunctionality import BasePlotter
from pathlib import Path

class ScatterPlot(BasePlotter):
    def __init__(self, columns, target):
        BasePlotter.__init__(self, columns=columns, target=target)

    def run(self):
        df = self.df
        columns = self.columns
        dirpath = self.dirpath
        filename = self.target
        target = self.target
        decisionfun = None
        predictproba = None

        X = df[columns].values

        x_min = X[:, 0].min()
        x_max = X[:, 0].max()
        y_min = X[:, 1].min()
        y_max = X[:, 1].max()
        colors = ['red', 'green', 'blue', 'purple', 'sienna', 'gold', 'deeppink', 'coral',
                  'orange', 'lime', 'deepskyblue', 'lightsteelblue', 'salmon', 'grey', 'brown',
                  'navy', 'blueviolet', 'orchid', 'dodgerblue', 'teal', 'chocolate', 'cyan', 'y', 'orangered']

        if target is not None:
            Y = df[target].values
            classes = df[target].unique().tolist()
            labels = [classes.index(x) for x in Y]
        else:
            classes = [' ']
            labels = [1 for x in X[:, 0]]

        fig, axes = plt.subplots(figsize=(15, 15))
        scatter = axes.scatter(X[:, 0], X[:, 1], c=labels, cmap=matplotlib.colors.ListedColormap(colors[0:len(classes)]),
                               s=10)
        try:
            if (decisionfun is not None) or (predictproba is not None):
                # plot the line, the points, and the nearest vectors to the plane
                xx = np.linspace(x_min, x_max, 10)
                yy = np.linspace(y_min, y_max, 10)

                X1, X2 = np.meshgrid(xx, yy)
                if len(classes) > 2:
                    for c in range(0, len(classes)):
                        Z = np.empty(X1.shape)
                        for (i, j), val in np.ndenumerate(X1):
                            x1 = val
                            x2 = X2[i, j]
                            if decisionfun is not None:
                                p = decisionfun([[x1, x2]])[:, c]
                                Z[i, j] = p[0]
                            else:
                                p = predictproba([[x1, x2]])[:, c]
                                Z[i, j] = p[0]
                        levels = [-1, 0.0, 1]
                        linestyles = ['dashed', 'solid', 'dashed']
                        axes.contour(X1, X2, Z, levels, colors=colors[c], linestyles=linestyles)
                else:
                    Z = np.empty(X1.shape)
                    for (i, j), val in np.ndenumerate(X1):
                        x1 = val
                        x2 = X2[i, j]
                        if decisionfun is not None:
                            p = decisionfun([[x1, x2]])
                            Z[i, j] = p[0]
                        else:
                            p = predictproba([[x1, x2]])[:, 1]
                            Z[i, j] = p[0]
                    levels = [-1, 0.0, 1]
                    linestyles = ['dashed', 'solid', 'dashed']
                    axes.contour(X1, X2, Z, levels, colors='k', linestyles=linestyles)
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            print("countor plotting failed!!!\n {0}".format(exceptionValue))

        axes.set_xlim(x_min, x_max)
        axes.set_ylim(y_min, y_max)
        axes.set_xlabel(columns[0])
        axes.set_ylabel(columns[1])
        cb = fig.colorbar(scatter, ax=axes)
        loc = np.arange(0, max(labels), max(labels) / float(len(colors[0:len(classes)])))
        cb.set_ticks(loc)
        cb.set_ticklabels(classes)
        path = dirpath / Path('_vs_'.join(columns) + '_' + filename + '.png')
        fig.savefig(path, dpi=600)
        print('file {0} saved.'.format(path))


def scatter(df, columns, filename, dirpath, target=None, decisionfun=None, predictproba=None):

    X = df[columns].values

    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()
    colors = ['red', 'green', 'blue', 'purple', 'sienna', 'gold', 'deeppink', 'coral',
              'orange', 'lime', 'deepskyblue', 'lightsteelblue', 'salmon', 'grey', 'brown',
              'navy', 'blueviolet', 'orchid', 'dodgerblue', 'teal', 'chocolate', 'cyan', 'y', 'orangered']

    if target is not None:
        Y = df[target].values
        classes = df[target].unique().tolist()
        labels = [classes.index(x) for x in Y]
    else:
        classes = [' ']
        labels = [1 for x in X[:, 0]]

    fig, axes = plt.subplots(figsize=(15, 15))
    scatter = axes.scatter(X[:, 0], X[:, 1], c=labels, cmap=matplotlib.colors.ListedColormap(colors[0:len(classes)]),
                           s=5)
    try:
        if (decisionfun is not None) or (predictproba is not None):
            # plot the line, the points, and the nearest vectors to the plane
            xx = np.linspace(x_min, x_max, 10)
            yy = np.linspace(y_min, y_max, 10)

            X1, X2 = np.meshgrid(xx, yy)
            if len(classes) > 2:
                for c in range(0, len(classes)):
                    Z = np.empty(X1.shape)
                    for (i, j), val in np.ndenumerate(X1):
                        x1 = val
                        x2 = X2[i, j]
                        if decisionfun is not None:
                            p = decisionfun([[x1, x2]])[:, c]
                            Z[i, j] = p[0]
                        else:
                            p = predictproba([[x1, x2]])[:, c]
                            Z[i, j] = p[0]
                    levels = [-1, 0.0, 1]
                    linestyles = ['dashed', 'solid', 'dashed']
                    axes.contour(X1, X2, Z, levels, colors=colors[c], linestyles=linestyles)
            else:
                Z = np.empty(X1.shape)
                for (i, j), val in np.ndenumerate(X1):
                    x1 = val
                    x2 = X2[i, j]
                    if decisionfun is not None:
                        p = decisionfun([[x1, x2]])
                        Z[i, j] = p[0]
                    else:
                        p = predictproba([[x1, x2]])[:, 1]
                        Z[i, j] = p[0]
                levels = [-1, 0.0, 1]
                linestyles = ['dashed', 'solid', 'dashed']
                axes.contour(X1, X2, Z, levels, colors='k', linestyles=linestyles)
    except:
        exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
        print("countor plotting failed!!!\n {0}".format(exceptionValue))

    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_min, y_max)
    axes.set_xlabel(columns[0])
    axes.set_ylabel(columns[1])
    cb = fig.colorbar(scatter, ax=axes)
    loc = np.arange(0, max(labels), max(labels) / float(len(colors[0:len(classes)])))
    cb.set_ticks(loc)
    cb.set_ticklabels(classes)
    path = dirpath / Path('_vs_'.join(columns) + '_' + filename + '.png')
    fig.savefig(path, dpi=600)
    print('file {0} saved.'.format(path))
