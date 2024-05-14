from gemlib.abstarct.basefunctionality import BaseRegressor
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor
import sys
from gemlib.visualization.barplots import bar
from gemlib.classification.modelvalidation import ValidateModelSingleFold

class FRegressorAlgorithm(BaseRegressor):

    def run_algo(self):
        X = self.df[self.features].values
        y = self.df[self.target].values
        try:
            F, pval = f_regression(X, y)
            # bar(x_labels=self.features, y_values=F, filename='f_reg', dirpath=self.dirpath)
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            print("fitting regression model failed!!!\n {0}".format(exceptionValue))

class RForest(BaseRegressor):

    def __init__(self, n_estimator, n_jobs, max_depth, **kwargs):
        super(RForest, self).__init__(**kwargs)
        self.n_estimator = n_estimator
        self.n_jobs = n_jobs
        self.max_depth = max_depth

    def init_model(self):
        self.model = RandomForestRegressor(random_state=self.random_state, 
                                            n_estimators=self.n_estimator, 
                                            n_jobs=self.n_jobs, 
                                            oob_score=True,
                                            max_depth=self.max_depth,
                                            min_samples_split=5)

        self.model_validation = ValidateModelSingleFold(self.df,
                                                    self.target,
                                                    self.features,
                                                    self.model,
                                                    self.algo_name)

    def run(self):
        return super().run()




