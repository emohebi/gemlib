import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics
from typing import List
from gemlib.abstarct.basefunctionality import BaseRocCurves, BaseConfusionDefinition
from gemlib.visualization import lineplots
import sys


class RocCurves(BaseRocCurves):
    def __init__(self, truth, predicted, groupby, validpreprocessing,
                 title, thresholds, multi_plot, figsize, fontsize, scatter_point):

        super().__init__(truth=truth, predicted=predicted,
                         groupby=groupby, validpreprocessing=validpreprocessing,
                         title=title, figsize=figsize, fontsize=fontsize,
                         scatter_point=scatter_point)
        self.thresholds = thresholds
        self.multi_plot = multi_plot

    def binarize_truth_signal(self) -> List[int]:
        # Binarize the truth signal
        binarized_signal_multi_thresholds = []
        for threshold in self.thresholds:
            binarized_signal_multi_thresholds.append([1 if x > threshold else 0 for x in self.df[self.truth]])
        return binarized_signal_multi_thresholds

    def roc_curve(self, binned_truth:np.ndarray, predicted):
        """ Run scikit-learn roc_curve """
        try:
            fpr, tpr, thresholds = metrics.roc_curve(binned_truth, self.df[predicted])
            auc = metrics.roc_auc_score(binned_truth, self.df[predicted])
            return fpr, tpr, auc
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            print("running ROC curve failed!!!\n {0} ".format(exceptionValue))

    def run(self):
        if not isinstance(self.predicted, list):
            self.predicted = [self.predicted]
        # Drop missing values
        self.remove_nans()
        # Binarize truth based on multiple thresholds
        binarized_truth = self.binarize_truth_signal()
        fprs = []
        tprs = []
        legends = []
        titles = []

        try:
            for single_thresh_truth, threshold in zip(binarized_truth, self.thresholds):
                # Run roc_curve for each threshold
                fpr, tpr, auc = self.roc_curve(single_thresh_truth, self.predicted[0])
                tprs.append(tpr)
                fprs.append(fpr)
                roc_curves_df = pd.DataFrame()
                roc_curves_df['tpr'] = tpr
                roc_curves_df['fpr'] = fpr
                self.output(roc_curves_df, str(threshold))
                legends.append(str(threshold) + ', AUC: ' +str(float("{0:.2f}".format(auc))))
                titles.append(self.title + str(threshold))
            if len(self.predicted) == 2: # to add reference signal to the roc curve
                fpr, tpr, auc = self.roc_curve(binarized_truth[0], self.predicted[1])
                tprs.append(tpr)
                fprs.append(fpr)
                roc_curves_df = pd.DataFrame()
                roc_curves_df['tpr'] = tpr
                roc_curves_df['fpr'] = fpr
                self.output(roc_curves_df, 'perclos_')
                legends.append(str(self.thresholds[0]) + ', AUC: ' +
                               str(float("{0:.2f}".format(auc))) + f" ({self.predicted[1]})")
                titles.append(self.title + str(self.thresholds[0]) + f" ({self.predicted[1]})")
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            print("ROC curve failed!!!\n {0}".format(exceptionValue))

        if len(fprs) > 0 and len(tprs) > 0:
            self.visualize(fprs, tprs, legends, titles)

    def output(self, df, label=''):
        df.to_csv(Path(self.dirpath + label + 'roc_curves.csv'), index=False)

    def visualize(self, x, y, legends, titles):
        if not self.multi_plot:
            titles = [self.title + " " + str(self.thresholds)]
        output_filepath = Path(self.dirpath + '/roc_curves.png')

        lineplot = lineplots.LinePlotting(x=x, y=y, legend=legends, x_label='False Positive rate',
                                          y_label='True Positive rate', dirpath=output_filepath,
                                          figsize=self.figsize, fontsize=self.fontsize,
                                          scatter_point=self.scatter_point)
        if self.multi_plot:
            lineplot.multi_plots(titles, legends)
        else:
            lineplot.single_plot(titles[0].replace("'", ""))