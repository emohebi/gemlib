import numpy as np
import matplotlib.pyplot as plt
from gemlib.abstarct.basefunctionality import BasePlotter
import gemlib.validation.utilities as utils
from matplotlib.ticker import MaxNLocator
import os
from pathlib import Path


class LinePlot(BasePlotter):
    def run(self):
        if self.multiaxis:
            self.multi_plot()
            return
        if self.groupby:
            for g, df in self.df.groupby(self.groupby):
                g = utils.resolve_string_tuple(g)
                if self.stacked:
                    self.stacked_plot(df=df, filename=g)
                else:
                    self.single_plot(df, g)
        else:
            self.single_plot(self.df, '')

    def single_plot(self, df, filename):
        with plt.style.context('classic'):
            df.reset_index(inplace=True, drop=True)
            for columns in self.columns:
                fig = plt.figure(figsize=self.figsize)
                if self.add_error:
                    ax = fig.add_axes((.1, .3, .8, .6))
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                    if self.x_col:
                        df.plot(x=self.x_col, y=columns, ax=ax)
                    else:
                        df[columns].plot(ax=ax)
                    ax2 = fig.add_axes((.1, .1, .8, .2))
                    df['zero'] = 0
                    plt.plot(df['zero'], color='green')
                    df['Absolute Error'] = np.abs(df[columns[0]] - df[columns[1]])
                    plt.plot(df['Absolute Error'], color='red')
                    ax2.set_ylabel('Absolute Error')
                    ax2.set_ylim(ymin=self.ticks_min_y, ymax=self.ticks_max_y)
                    ax2.set_xlabel(self.x_label)
                    if self.fontsize:
                        for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
                                         ax2.get_xticklabels() + ax2.get_yticklabels()):
                            item.set_fontsize(self.fontsize)
                        ax2.legend(prop={'size': self.fontsize - 2})
                else:
                    print(df)
                    if self.x_col:
                        ax = df.plot(x=self.x_col, y=columns,figsize=self.figsize)
                    else:
                        ax = df[columns].plot(figsize=self.figsize)
                    ax.set_xlabel(self.x_label)
                    ax.set_ylim(ymin=self.ticks_min_y, ymax=self.ticks_max_y)
                ax.set_title(self.title)
                ax.set_ylabel(self.y_label)
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                if self.fontsize:
                    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                     ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(self.fontsize)
                    ax.legend(prop={'size': self.fontsize - 2})
                    # ax.get_legend().remove()
                self.plt = plt
                filename_ = self.dirpath / Path(self.title + '_' + filename +'_'.join(columns)+ '.png')
                self.save(filename_)
                plt.close("all")

    def set_lims(self, ax, df, columns):
        if abs(df[columns].max() - self.ticks_max_y) < 10 and abs(df[columns].min() - self.ticks_min_y) < 10:
            ax.set_ylim(ymin=self.ticks_min_y, ymax=self.ticks_max_y)
        elif df[columns].max() == 0.0 and df[columns].min() == 0.0:
            ax.set_ylim(ymin=0.0 - 0.1, ymax=1.1)
        elif abs(df[columns].max() - self.ticks_max_y) >= 10 and abs(df[columns].min() - self.ticks_min_y) < 10:
            ax.set_ylim(ymin=self.ticks_min_y - 0.1, ymax=df[columns].max() + 0.1)
        else:
            ax.set_ylim(ymin=df[columns].min() - 0.1, ymax=df[columns].max() + 0.1)

    def stacked_plot(self, df, filename):
        with plt.style.context('classic'):
            linewidth = 0.5
            df.reset_index(inplace=True, drop=True)
            constant_gap = {2: .06, 3: .07, 4: .075, 5: .065}
            num_plots_in_stack = len(self.columns)
            step = 1/num_plots_in_stack
            gap = 1/num_plots_in_stack - constant_gap[num_plots_in_stack]
            fig = plt.figure(figsize=self.figsize)
            axes = []
            for ind, item in enumerate(zip(self.columns, np.arange(.1, 1, step))):
                columns, offset = item
                ind = ind//2
                ax = fig.add_axes((.1, offset, .8, gap))
                axes.append(ax)
                if self.x_col:
                    df.plot(x=self.x_col, y=columns, ax=ax,
                            colormap=self.colors[ind], linewidth=linewidth)
                else:
                    df[columns].plot(ax=ax, colormap=self.colors[ind], linewidth=linewidth)
                self.set_lims(ax, df, columns)
                ax.set_ylabel(self.y_label if isinstance(columns, list) else columns)
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            for i in range(len(axes)):
                if i==0:
                    axes[i].set_xlabel(self.x_label if self.x_label else '')
                else:
                    axes[i].set_xlabel('')
            axes[-1].set_title(self.title)
            if self.fontsize:
                for ax in axes:
                    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                     ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(self.fontsize)
                    ax.legend(prop={'size': 2})
                    ax.get_legend().remove()
            self.plt = plt
            filename = self.dirpath / Path(self.title + '_' + filename + '.png')
            self.save(filename)
            plt.close("all")

    def multi_plot(self):
        num_sub_plots_per_plot = 4
        counter = 0
        plt_counter = 0
        x = 0
        y = 0
        font_size = self.fontsize
        pad_size = 1

        fig, axs = plt.subplots(2, 2, figsize=self.figsize)
        grouped_df = self.df.groupby(self.groupby)
        linewidth = 0.3
        marker_size = 0.15

        for g, df in grouped_df:
            df = df.dropna()
            if counter < num_sub_plots_per_plot:
                col1 = self.columns[0]
                p1 = axs[x][y].plot(df[self.x_col], df[col1], label=col1, color='green', linewidth=linewidth)
                axs[x][y].tick_params(axis='both', grid_linewidth=marker_size, width=marker_size, length=marker_size, pad=pad_size)
                y_ticks = np.arange(self.ticks_min_y, self.ticks_max_y, 1)
                axs[x][y].set_yticks(y_ticks)
                axs[x][y].set_yticklabels(y_ticks, fontsize=font_size)
                axs[x][y].set_ylabel(col1, fontsize=font_size, labelpad=pad_size)

                col2 = self.columns[1]
                if self.multiaxis:
                    twinx = axs[x][y].twinx()
                    p2 = twinx.plot(df[self.x_col], df[col2], label=col2, color='black', linewidth=linewidth)
                    twinx.tick_params(axis='both', grid_linewidth=marker_size, width=marker_size, length=marker_size, pad=pad_size)
                    y_ticks = np.arange(df[col2].min(), df[col2].max()+0.2)
                    y_ticks = [np.round(x, decimals=1) for x in y_ticks]
                    twinx.set_yticks(y_ticks)
                    twinx.set_yticklabels(y_ticks, fontsize=font_size)
                    twinx.set_ylabel(col2, fontsize=font_size, labelpad=pad_size)
                    if 'RegressorScoreThreshold' in df:
                        p3 = twinx.plot(df[self.x_col], df['RegressorScoreThreshold'], color='grey', linestyle="dashed",
                                        linewidth=linewidth)
                else:
                    p2 = axs[x][y].plot(df[self.x_col], df[col2], label=col2, color='k', linewidth=linewidth)

                lns = p1 + p2 + p3
                labs = [l.get_label() for l in lns]
                axs[x][y].legend(lns, labs, loc=2, fontsize=font_size, frameon=False)
                axs[x][y].set_title(g, fontsize=font_size)
                axs[x][y].tick_params(axis='x')
                axs[x][y].set_xticklabels(df[self.x_col], fontsize=font_size)
                axs[x][y].set_xlabel(self.x_label, fontsize=font_size, labelpad=pad_size)

                counter += 1
                y = 1 if y == 0 else 0
                x = 1 if (counter % 2 == 0) else x

            if counter >= num_sub_plots_per_plot:
                plt.tight_layout()
                filename = self.dirpath + self.title + "_" + str(plt_counter) + ".png"
                self.save_fig(fig, filename)
                x = 0
                y = 0
                fig, axs = plt.subplots(2, 2, figsize=self.figsize)
                counter = 0
                plt_counter += 1

        filename = self.dirpath / Path(self.title + "_" + str(plt_counter) + ".png")
        self.save_fig(fig, filename)
        fig.clf()
        plt.close("all")

    @staticmethod
    def save_fig(fig, filename):
        fig.savefig(filename, dpi=500, bbox_inches='tight', pad_inches=0.1)
        print('file {0} saved.'.format(filename))


class LinePlotting(BasePlotter):
    def __init__(self, x, y, legend, x_label, y_label, dirpath, figsize, fontsize, scatter_point):
        super().__init__(x=x, y=y, legend=legend, x_label=x_label, y_label=y_label, dirpath=dirpath,
                         figsize=figsize, fontsize=fontsize, scatter_point=scatter_point)

    def run(self):
        self.single_plot(self.title)

    @staticmethod
    def exclude_missing_values(x_values, y_values):
        not_nan_mask = ~np.isnan(y_values)
        return x_values[not_nan_mask], y_values[not_nan_mask]

    def single_plot(self, title):
        fig = plt.figure(50, figsize=self.figsize)
        for x_val, y_val, label in zip(self.x, self.y, self.legend):
            if np.all(np.isnan(y_val)):
                return
            x_val, y_val = self.exclude_missing_values(x_val, y_val)
            plt.plot(x_val, y_val, label=label, linewidth=1)
        plt.legend(loc=4, borderaxespad=0.8, fontsize=self.fontsize)
        plt.xlabel(self.x_label, fontsize=self.fontsize)
        plt.ylabel(self.y_label, fontsize=self.fontsize)
        plt.title(title, fontsize=self.fontsize)
        if self.scatter_point:
            plt.plot(self.scatter_point[0], self.scatter_point[1], 'o')
        fig.savefig(self.dirpath, dpi=500, bbox_inches='tight')
        print('file {0} saved.'.format(self.dirpath))
        plt.close("all")

    def multi_plots(self, titles, legends):
        fig, axs = plt.subplots(2, 2)
        x = 0
        y = 0
        counter = 0
        for fpr, tpr in zip(self.x, self.y):
            fpr, tpr = self.exclude_missing_values(fpr, tpr)
            axs[x][y].plot(fpr, tpr, label=legends[counter])
            axs[x][y].set_xlabel(self.x_label, fontsize=self.fontsize)
            axs[x][y].set_ylabel(self.y_label, fontsize=self.fontsize)
            axs[x][y].set_title(titles[counter], fontsize=self.fontsize)
            axs[x][y].legend(loc=4, fontsize=self.fontsize, linewidth=1)
            counter += 1
            y = 1 if y == 0 else 0
            x = 1 if (counter % 2 == 0) else x
        if self.scatter_point:
            plt.plot(self.scatter_point[0], self.scatter_point[1], 'o')
        plt.tight_layout()
        plt.savefig(self.dirpath, dpi=500, bbox_inches='tight')
        print('file {0} saved.'.format(self.dirpath))
        plt.close("all")
