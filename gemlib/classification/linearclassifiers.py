from gemlib.abstarct.basefunctionality import BaseClassifier
from sklearn import svm
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import class_weight
from gemlib.visualization.scatterplots import scatter
from gemlib.classification.classvalidator import get_cross_validation_average_score
import sys
import os
import pickle
from gemlib.validation import utilities
import pandas as pd
import numpy as np
from pathlib import Path


class StochasticGradientDecent(BaseClassifier):
    def __init__(self, x_train, x_test, y_train, y_test, top_n=1, max_iter=1000, tol=1e-3, random_state=49):
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        BaseClassifier.__init__(self, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, top_n=top_n, algo_name='svm')

    def run(self):
        self.model = self.get_model()
        if self.mode == 'training':
            self.run_training()
        elif self.mode == 'testing':
            self.run_testing()
        else:
            utilities._error(NotImplementedError, f"the mode:{self.mode} is not a correct mode.")

    def run_testing(self):
        x_test = utilities.resolve_caching_load(self.resources, self.x_test)
        preds = self.model.decision_function(x_test)
        if preds.ndim == 1:
            preds = preds.reshape(-1,1)
        top_n = self.top_n
        if preds.shape[1] < top_n:
            top_n = 1
        df = pd.DataFrame(preds.argsort()[:,-top_n:]).T.unstack().to_frame().reset_index().rename({'level_0':'index', 
                                                                                                    'level_1':'rank',
                                                                                                    0:'prediction'}, axis='columns')
        df_p = pd.DataFrame(np.sort(preds)[:,-top_n:]).T.unstack().to_frame().reset_index().rename({'level_0':'index', 
                                                                                                    'level_1':'rank',
                                                                                                    0:'probability'}, axis='columns')
        df = df.merge(df_p, on=['index', 'rank'])
        _mapping = utilities.resolve_caching_load(self.resources, self.y_col_map)
        if not isinstance(_mapping, dict) and _mapping is not None:
            _mapping = {c:_mapping[c] for c in range(_mapping.shape[0])}
            df['prediction'] = df['prediction'].map(_mapping)
        return df

    def get_model(self):
        if self.model_path is not None and os.path.isfile(self.model_path):
            return pickle.load(open(self.model_path, 'rb'))

        class_weights = class_weight.compute_class_weight('balanced', np.unique(self.y_train), self.y_train)
        cl_weights = {i : class_weights[i] for i in range(class_weights.shape[0])}
        return make_pipeline(StandardScaler(),
                             linear_model.SGDClassifier(max_iter=self.max_iter, 
                                                        tol=self.tol, 
                                                        random_state=self.random_state, 
                                                        penalty='elasticnet',
                                                        class_weight=cl_weights))

    def run_training(self):
        
        self.model.fit(self.x_train, self.y_train)
        utilities._info(f"train score:{self.model.score(self.x_train, self.y_train)}")
        if self.x_test is not None:
            utilities._info(f"test score:{self.model.score(self.x_test, self.y_test)}")
        model_path = str(Path(self.dirpath) / f'{self.name}_{self.model_name}_model.pkl')
        pickle.dump(self.model, open(model_path, 'wb'))

class LinearSvmClassifier(BaseClassifier):
    def __init__(self, dimensions):
        BaseClassifier.__init__(self, dimensions=dimensions, algo_name='svm')

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
        try:
            clf = svm.LinearSVC(C=1.0, max_iter=self.num_iter, fit_intercept=True)
            clf.fit(X, Y)
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            print("fitting model failed!!!\n {0}".format(exceptionValue))
        scatter(df=self.df, columns=columns, target=self.target, filename='svm', dirpath=self.dirpath,
                decisionfun=clf.decision_function)
        print('columns {0} score: {1}'.format(columns, get_cross_validation_average_score(clf, X, Y)))
        return [
            ['score', ['X', 'Y', 'score'], [[columns[0], columns[1], get_cross_validation_average_score(clf, X, Y)]]],
            ['clf', columns, clf]]

class LinearLogisticRegressionClassifier(BaseClassifier):
    def __init__(self, dimensions):
        BaseClassifier.__init__(self, dimensions=dimensions, algo_name='lr')

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
        try:
            clf = linear_model.LogisticRegression(C=1e5, max_iter=self.num_iter, fit_intercept=True)
            clf.fit(X, Y)
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            print("fitting model failed!!!\n {0}".format(exceptionValue))
        scatter(df=self.df, columns=columns, target=self.target, filename='logreg', dirpath=self.dirpath,
                decisionfun=clf.decision_function)
        print('columns {0} score: {1}'.format(columns, get_cross_validation_average_score(clf, X, Y)))
        return [
            ['score', ['X', 'Y', 'score'], [[columns[0], columns[1], get_cross_validation_average_score(clf, X, Y)]]],
            ['clf', columns, clf]]

class LinearLogisticRegressionCVClassifier(BaseClassifier):
    def __init__(self, dimensions):
        BaseClassifier.__init__(self, dimensions=dimensions, algo_name='lrc')

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
        try:
            clf = linear_model.LogisticRegressionCV(solver='liblinear', fit_intercept=True, class_weight='balanced')
            clf.fit(X, Y)
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            print("fitting model failed!!!\n {0}".format(exceptionValue))
        scatter(df=self.df, columns=columns, target=self.target, filename='logregcv', dirpath=self.dirpath,
                decisionfun=clf.decision_function)
        print('columns {0} score: {1}'.format(columns, get_cross_validation_average_score(clf, X, Y)))
        return [
            ['score', ['X', 'Y', 'score'], [[columns[0], columns[1], get_cross_validation_average_score(clf, X, Y)]]],
            ['clf', columns, clf]]

class LinearRidgeClassifier(BaseClassifier):
    def __init__(self, dimensions):
        BaseClassifier.__init__(self, dimensions=dimensions, algo_name='ridge')

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
        try:
            clf = linear_model.RidgeClassifier(alpha=self.alpha, max_iter=self.num_iter, fit_intercept=True)
            clf.fit(X, Y)
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            print("fitting model failed!!!\n {0}".format(exceptionValue))
        scatter(df=self.df, columns=columns, target=self.target, filename='ridge', dirpath=self.dirpath,
                decisionfun=clf.decision_function)
        print('columns {0} score: {1}'.format(columns, get_cross_validation_average_score(clf, X, Y)))
        return [
            ['score', ['X', 'Y', 'score'], [[columns[0], columns[1], get_cross_validation_average_score(clf, X, Y)]]],
            ['clf', columns, clf]]