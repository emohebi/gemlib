from gemlib.abstarct.basefunctionality import BaseClassifier
from sklearn import tree
import sys
# import pydotplus as pydot
import matplotlib.pyplot as plt
import numpy as np

class Decisiontree(BaseClassifier):

    def run_testing(self):
        pass

    def get_model(self):
        pass

    def run_training(self, columns):
        X = self.df[columns].values
        Y = self.df[self.target].values
        clf = tree.DecisionTreeClassifier(max_depth=self.depth).fit(X, Y)
        # dot_data = tree.export_graphviz(clf, out_file=None, feature_names=columns,
        #                                 class_names=self.df[self.target].unique().tolist())
        # graph = pydot.graph_from_dot_data(dot_data)
        # path = self.dirpath + '_vs_'.join(columns[0:2]) + '_' + 'decisiontree.pdf'
        # graph.write_pdf(path)
        # if len(columns) == 2 and self.df[self.target].dtype != object:
        #     plot_colors = "bry"
        #     plot_step = 0.001
        #     fig = plt.figure(figsize=(5, 5))  # figsize=(5, 5), dpi=600
        #     x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        #     y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        #     xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
        #                          np.arange(y_min, y_max, plot_step))

        #     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        #     Z = Z.reshape(xx.shape)
        #     cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

        #     for i, color in zip([1, 0], plot_colors):
        #         idx = np.where(Y == i)
        #         plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired, s=1)

        #     plt.axis("tight")
        #     plt.suptitle("Decision surface of a decision tree using paired features")
        #     plt.legend()
        #     fig.savefig(self.dirpath + '_vs_'.join(columns[0:2]), dpi=300, bbox_inches='tight')


    def run(self, multiproc=None):
        if self.target in self.dimensions:
            self.dimensions.remove(self.target)
        columns = [x for x in self.dimensions]

        try:
            self.run()
        except Exception:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            print("classifier failed!!!\n {0}".format(exceptionValue))
