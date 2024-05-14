from sklearn.neighbors import KNeighborsClassifier
from gemlib.abstarct.basefunctionality import BaseClassifier
from gemlib.visualization.scatterplots import scatter
from gemlib.classification import classvalidator

class KnnClassifier(BaseClassifier):

    def run(self):
        pass

    def run_testing(self):
        pass

    def get_model(self):
        pass

    def run_training(self, columns):
        X = self.df[columns].values
        Y = self.df[self.target].values
        # fit the model
        clf = KNeighborsClassifier(self.k)
        clf.fit(X, Y)
        scatter(df=self.df, columns=columns, target=self.target, filename='knn', dirpath=self.dirpath,
                predictproba=clf.predict_proba)
        print('columns {0} score: {1}'.format(columns, classvalidator.get_cross_validation_average_score(clf, X, Y)))
        return [['score', ['X', 'Y', 'score'], [[columns[0], columns[1], classvalidator.get_cross_validation_average_score(clf, X, Y)]]]]