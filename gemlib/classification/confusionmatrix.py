from gemlib.abstarct.basefunctionality import BaseConfusionDefinition
import pandas as pd
import numpy as np
from gemlib.visualization import barplots, heatmaps
import gemlib.validation.utilities as utils
import sys, os


class confusion_matrix(BaseConfusionDefinition):
    def run(self):
        results = {}
        if self.groupby is None:
            c_tab = pd.crosstab(self.df[self.predicted],
                                self.df[self.truth],
                                rownames=['Predicted'],
                                colnames=['Truth'])
            c_tab = self.apply_indices(c_tab)
            self.output(c_tab, self.title)
            results['single'] = c_tab
        else:
            for key, df0 in self.df.groupby(self.groupby):
                c_tab = pd.crosstab(df0[self.predicted],
                                    df0[self.truth],
                                    rownames=['Predicted'],
                                    colnames=['Truth'])
                c_tab = self.apply_indices(c_tab)
                self.output(c_tab, utils.resolve_string_tuple(key))
                results[str(key)] = c_tab
        return results

    def apply_indices(self, c_tab):
        if self.rowindex is not None:
            c_tab = c_tab.reindex(self.rowindex)
        if self.columnindex is not None:
            label = [x for x in self.columnindex if x in c_tab.columns.tolist()]
            c_tab = c_tab[label]
        return c_tab

    def output(self, c_tab, label):
        c_tab.to_csv(self.dirpath + label + '_counts.csv')
        c_tab = c_tab.apply(lambda r: np.round(100.0 * r / r.sum(), 1))
        c_tab.to_csv(self.dirpath + label + '.csv')
        htmap = heatmaps.HeatMaps(df=c_tab,
                                  max_v=100,
                                  min_v=0,
                                  title=label,
                                  dirpath=self.dirpath,
                                  cmap='Fovio')

        htmap.run(dpi=600)


class accuracy(BaseConfusionDefinition):
    def run(self):
        acc = []
        xlabels = []
        results = {}
        if self.groupby is None:
            acc.append(self.get_acc(self.df))
            xlabels.append('Total_Acc')
        else:
            for key, df0 in self.df.groupby(self.groupby):
                acc.append(self.get_acc(df0))
                xlabels.append(utils.resolve_string_tuple(key))
        results['single'] = self.get_df(xlabels, acc)
        if self.groupby is None:
            self.output(xlabels, acc, 'total_acc', results['single'])
        else:
            self.output(xlabels, acc, '_'.join(self.groupby) + '_acc', results['single'])
        return results

    def get_acc(self, df):
        return len(df[df[self.predicted] == df[self.truth]]) * 100 / len(df)

    def get_df(self, xlabels, acc):
        if self.groupby is None:
            return pd.DataFrame(np.array(acc), columns=['Rate'], index=['Total_Acc'])
        values = np.stack((np.array(xlabels), np.array(acc)), axis=-1)
        return pd.DataFrame(values, columns=['Label', 'Acc'])

    def output(self, Xlabels, Y, filename, df_acc):
        df_acc.to_csv(self.dirpath + filename + '.csv', index=False)
        if self.groupby is not None:
            if self.validvisualization == 'bar':
                barplot = barplots.BarPlotting(ticks_min_y=0, ticks_max_y=110,
                                               fontsize=10, linewidth=0.2, padsize=1)
                barplot.bar(x_labels=Xlabels, y_values=Y, filename=filename, dirpath=self.dirpath,
                            xlabel=self.groupby[0], ylabel=self.title)
            elif self.validvisualization == 'heatmap':
                htmap = heatmaps.HeatMaps(df=df_acc,
                                          max_v=100,
                                          min_v=0,
                                          title=self.title,
                                          dirpath=self.dirpath,
                                          cmap='Fovio')
                htmap.run()


class TP_FP(BaseConfusionDefinition):  # for True/False cases
    def run(self):
        acc = []
        results = {}
        if self.groupby is None:
            acc.append(self.get_tp_fp(self.df) + ['TP_FP'])
        else:
            for key, df0 in self.df.groupby(self.groupby):
                acc.append(self.get_tp_fp(df0) + [utils.resolve_string_tuple(key)])
        df = self.get_df(acc)
        self.output(df)
        results['single'] = df
        return results

    def get_tp_fp(self, df):
        t_count = len(df[df[self.truth] == True])
        m_count = len(df[df[self.predicted] == True])
        tp_count = len(df[(df[self.predicted] == True) & (df[self.truth] == True)])
        fp_count = len(df[(df[self.predicted] == True) & (df[self.truth] == False)])
        fn_count = len(df[(df[self.predicted] == False) & (df[self.truth] == True)])
        tn_count = len(df[(df[self.predicted] == False) & (df[self.truth] == False)])
        tp = tp_count * 100 / (fn_count + tp_count)
        fp = fp_count * 100 / (fp_count + tn_count)
        fd = fp_count * 100 / (fp_count + tp_count)

        if len(df[df[self.truth] == True]) == 0 and fp == 0.0:
            return [np.nan, 0]
        return [m_count, t_count, fp_count, tp_count, fd, tp, fp]

    def get_df(self, acc):
        df = pd.DataFrame(acc, columns=["Measured_Count", "Truth_Count", "FP_Count", "TP_Count", 'FD%', 'TP%',
                                        'FP%', 'Label'])
        df.set_index('Label', inplace=True)
        return df

    def output(self, df_acc):
        df_acc.to_csv(self.dirpath + f'{self.truth}_TP_FP.csv')


class TP(BaseConfusionDefinition):
    truth_unique_items = []
    groupby_results = {}

    def run(self):
        try:
            self.truth_unique_items = self.df[self.truth].unique()
            if self.groupby is None:
                c_tab = pd.crosstab(self.df[self.predicted],
                                    self.df[self.truth],
                                    rownames=['Predicted'],
                                    colnames=['Truth']) \
                    .apply(lambda r: np.round(100.0 * r / r.sum(), 1))
            else:
                for truth in self.truth_unique_items:
                    self.groupby_results[truth] = {}
                    df = self.df[self.df[self.truth] == truth]
                    for key, df0 in df.groupby(self.groupby):
                        c_tab = pd.crosstab(df0[self.predicted],
                                            df0[self.truth],
                                            rownames=['Predicted'],
                                            colnames=['Truth']) \
                            .apply(lambda r: np.round(100.0 * r / r.sum(), 1))
                        if isinstance(key, str):  # one element in groupby
                            key = [key]
                        else:  # it's a tuple
                            key = list(key)
                        if len(key) <= 2:
                            self.get_tp(c_tab, truth, key)
                        else:
                            print("Error: TP for groupby of more than two elements is not " +
                                  "implemented.")
                            raise NotImplementedError
            return self.output()
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            print("tp_run failed!!!\n {0}".format(exceptionValue))

    def get_tp(self, c_tab, truth, key):
        v1 = None
        v2 = None
        try:
            if len(key) == 1:
                v1 = key[0]
            else:
                v1, v2 = key
            label = list(set(c_tab.index.tolist()).intersection(c_tab.columns.tolist()))
            tp = 0
            if (label is not None) and (len(label) > 0):
                tp = c_tab.loc[label][truth].sum(axis=0)

            if v1 in self.groupby_results[truth]:
                if v2 is not None:
                    self.groupby_results[truth][v1][v2] = tp
                else:
                    self.groupby_results[truth][v1] = tp
            else:
                if v2 is not None:
                    self.groupby_results[truth][v1] = {}
                    self.groupby_results[truth][v1][v2] = tp
                else:
                    self.groupby_results[truth][v1] = tp
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            print("get_tp failed!!!\n {0}".format(exceptionValue))

    def output(self):
        results = {}
        if len(self.groupby_results) <= 0:
            print("no TP results !!! try again.")
            return
        try:
            for truth in self.groupby_results:
                if len(self.groupby_results[truth]) != 0:
                    dct = self.groupby_results[truth]
                    # df = pd.DataFrame({'TP_Rate': list(dct.values())}, index=list(dct.keys()))
                    df = pd.DataFrame.from_dict(self.groupby_results[truth])

                    if self.title:
                        title = self.title
                    else:
                        title = str(truth)

                    filename = title + "_TP"
                    df.to_csv(self.dirpath + filename + '.csv')

                    if self.type == 'bar':
                        barplot = barplots.BarPlotting(ticks_min_y=0, ticks_max_y=110)
                        # barplot.bar(x_labels=df.index, y_values=df['TP_Rate'].values, filename=filename,
                        #            dirpath=self.dirpath, title=self.title, rot_angle=90)

                        # support bar plotting when df has multiple columns
                        # generally caused by multiple group-bys
                        for colname in df.columns:
                            per_column_filename = filename + '-' + colname
                            tdf = df.sort_values(by=colname, ascending=False)
                            barplot.bar(
                                x_labels=tdf.index,
                                y_values=tdf[colname].values,
                                filename=per_column_filename,
                                dirpath=self.dirpath,
                                title=per_column_filename,
                                rot_angle=90)
                    else:
                        htmap = heatmaps.HeatMaps(df=df,
                                                  max_v=100,
                                                  min_v=0,
                                                  title=filename,
                                                  dirpath=self.dirpath,
                                                  cmap='Fovio')
                        htmap.run()
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            print("tp_output failed!!!\n {0}".format(exceptionValue))
            print("line number  {0}".format(exceptionTraceback.tb_lineno))
        return results


class FP(BaseConfusionDefinition):
    truth_unique_items = []
    groupby_results = {}

    def run(self):
        try:
            self.truth_unique_items = self.df[self.truth].unique()
            if self.groupby is None:
                c_tab = pd.crosstab(self.df[self.predicted],
                                    self.df[self.truth],
                                    rownames=['Predicted'],
                                    colnames=['Truth']) \
                    .apply(lambda r: np.round(100.0 * r / r.sum(), 1))
            else:
                for truth in self.truth_unique_items:
                    self.groupby_results[truth] = {}
                    df = self.df[self.df[self.truth] == truth]
                    for key, df0 in df.groupby(self.groupby):
                        c_tab = pd.crosstab(df0[self.predicted],
                                            df0[self.truth],
                                            rownames=['Predicted'],
                                            colnames=['Truth']) \
                            .apply(lambda r: np.round(100.0 * r / r.sum(), 1))
                        if isinstance(key, str):  # one element in groupby
                            key = [key]
                        else:  # it's a tuple
                            key = list(key)
                        if len(key) < 2:
                            self.get_fp(c_tab, truth, key)
                        else:
                            print("Error: FP for groupby of more than one elements is not " +
                                  "implemented.")
                            raise NotImplementedError
            return self.output()
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            print("fp_run failed!!!\n {0}".format(exceptionValue))

    def get_fp(self, c_tab, truth, key):
        v1 = None
        if len(key) == 1:
            v1 = key[0]

        label = set(c_tab.index.tolist()) - set(c_tab.columns.tolist())
        fps = {}
        if (label is not None) and (len(label) > 0):
            for l in label:
                fps[l] = c_tab.loc[l].sum(axis=0)
            if truth in self.groupby_results:
                if self.groupby_results[truth].get(v1) is None:
                    self.groupby_results[truth][v1] = fps
                else:
                    self.groupby_results[truth][v1].update(fps)

    def output(self):
        for truth in self.groupby_results:
            if len(self.groupby_results[truth]) != 0:
                df_x = pd.DataFrame.from_dict(self.groupby_results[truth])
                df_x_T = df_x.T
                df_x_T.rename(columns=lambda x: "FP_Rate(" + str(x) + ")", inplace=True)
                base_filename = str(truth) + "_FP"
                df_x_T.to_csv(os.path.join(self.dirpath, (base_filename + '.csv')))
                if self.type == 'bar':
                    barplot = barplots.BarPlotting(ticks_min_y=0, ticks_max_y=110,
                                                   fontsize=10, linewidth=0.2,
                                                   padsize=1)
                    barplot.bar(x_labels=df_x_T.index, y_values=df_x_T.iloc[:, 0].values,
                                filename=base_filename, dirpath=self.dirpath,
                                rot_angle=90, xlabel=self.groupby[0], ylabel=self.title)
                else:
                    htmap = heatmaps.HeatMaps(df=df_x_T,
                                              max_v=100,
                                              min_v=0,
                                              title=str(truth),
                                              dirpath=self.dirpath,
                                              cmap='Fovio')
                    htmap.run()


class event(BaseConfusionDefinition):

    def run(self):
        if self.groupby is None:
            self.groupby = ['ShortFilename']
            df = self.get_agg_df()
            t_count = df['Truth_Event_Count'].sum()
            m_count = df['Meas_Event_Count'].sum()
            tp = df['TP_Count'].sum()
            fp = df['FP_Count'].sum()
            tpr = (df['TP_Count'].sum() / t_count) * 100
            fdr = (df['FP_Count'].sum() / (fp + tp)) * 100
            df = pd.DataFrame([[m_count, t_count, fp, tp, fdr, tpr]],
                              columns=['Meas_Event_Count', 'Truth_Event_Count',
                                       'FP_Count', 'TP_Count', 'FDR', 'TPR'])
            df = self.rename_columns(df)
            return self.output(df, 'summary_event_results')
        else:
            df = self.get_agg_df()
            df = self.rename_columns(df)
            return self.output(df, 'event_results_per_' + '_'.join(self.groupby))

    def get_agg_df(self):
        columns = self.groupby + ['result']
        event_info = self.groupby + ['Event_ID']
        df_temp = self.df[
            self.keepcolumnvalues + ['ShortFilename']].copy() if self.keepcolumnvalues is not None else None
        df0 = self.df[columns + [self.timecolumn]].groupby(columns).count().unstack()
        df1 = self.df[event_info + [self.timecolumn]].groupby(event_info).count().unstack()
        df1.columns = ['_'.join(col).strip() for col in df1.columns.values]
        df0.columns = ['_'.join(col).strip() for col in df0.columns.values]

        df_f = df1.join(df0, on=self.groupby).fillna(0)
        if df_temp is not None:
            df_temp.set_index('ShortFilename', inplace=True)
            df_f = df_f.join(df_temp, how='left')

        df_f.rename({self.timecolumn + '_TP': 'TP_Count',
                     self.timecolumn + '_FP': 'FP_Count',
                     self.timecolumn + '_t': 'Truth_Event_Count',
                     self.timecolumn + '_m': 'Meas_Event_Count',
                     }, axis='columns', inplace=True)

        if 'TP_Count' not in df_f.columns:
            df_f['TP_Count'] = 0
        if 'FP_Count' not in df_f.columns:
            df_f['FP_Count'] = 0
        if 'Meas_Event_Count' not in df_f.columns:
            df_f['Meas_Event_Count'] = 0
        if 'Truth_Event_Count' in df_f.columns:
            df_f['TPR'] = (df_f['TP_Count'] / df_f['Truth_Event_Count']) * 100
        else:
            df_f['Truth_Event_Count'] = 0
            df_f['TPR'] = 0
        df_f['FDR'] = (df_f['FP_Count'] / (df_f['FP_Count'] + df_f['TP_Count'])) * 100
        df_f = df_f[~(df_f.index == 'nan')].copy()
        return df_f

    def output(self, df, label=''):
        df.to_csv(self.dirpath + label + '.csv')
        return {'single': df}
