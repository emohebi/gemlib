
from gemlib.abstarct.basefunctionality import BaseModelValidation
from sklearn.model_selection import train_test_split
from gemlib.validation import utilities as utils
from gemlib.validation.utilities import Spinner
import time

class ValidateModelSingleFold(BaseModelValidation):

    def populate_train_data(self):
        train_mask = ~self.df[self.target].isnull()
        self.X_train = self.df[train_mask][self.features].values
        self.y_train = self.df[train_mask][self.target].values
        self.X_test = self.df[~train_mask][self.features]

    def run(self):
        self.populate_train_data()

        utils._info('Training and testing on the train set only...')
        X_train, X_test, y_train, y_test = train_test_split(self.X_train, 
                                                            self.y_train, 
                                                            test_size=0.2, 
                                                            random_state=42, 
                                                            shuffle=False)
        utils._info(f'Running {self.algo_name} model...')
        with Spinner():
            model = self.model.fit(X_train, y_train)
            time.sleep(3)
        utils._info('Model trained!!!')
        preds = model.predict(X_test)
        
        utils._info(f'Scores: \n train: {model.score(X_train, y_train)},' 
                    f' test: {model.score(X_test, y_test)}')