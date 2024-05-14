from gemlib.abstarct.basefunctionality import BaseClassifier
from sklearn import svm
from gemlib.visualization.scatterplots import scatter
from gemlib.classification import classvalidator


class SvmSvcClassifier(BaseClassifier):
    def run(self):
        pass

    def run_testing(self):
        pass

    def get_model(self):
        pass

    def run_training(self, columns):
        X = self.df[columns].values
        Y = self.df[self.target].values

        clf = svm.SVC(kernel=self.kernel, C=self.alpha)
        clf.fit(X, Y)
        scatter(df=self.df, columns=columns, target=self.target, filename='svc_' + self.kernel, dirpath=self.dirpath,
                decisionfun=clf.decision_function)
        print('columns {0} score: {1}'.format(columns, classvalidator.get_cross_validation_average_score(clf, X, Y)))
        return [['score', ['X', 'Y', 'score'], [[columns[0], columns[1], classvalidator.get_cross_validation_average_score(clf, X, Y)]]]]

