from gemlib.framework import pipeline
import pandas as pd
import numpy as np
from gemlib.abstarct.basefunctionality import BaseClassifier
from gemlib.abstarct.basetask import task


class Discovery(pipeline.Pipeline):
    def biclasses_discovery(self, lin_classification_task):
        clf = lin_classification_task.output['clf'][','.join(self.knowledge.best_sparable_features)]
        features = self.knowledge.best_sparable_features
        self.df['UP_DECISION_FUN'] = self.df[features[0]] * clf.coef_[0,0] + self.df[features[1]] * clf.coef_[0,1] \
                   + clf.intercept_[0] >= 0
        self.knowledge.bicluster_decision_column = 'UP_DECISION_FUN'
        print(clf.intercept_[0] / clf.coef_[0, 0])
        print(clf.intercept_[0] / clf.coef_[0, 1])
        print(self.knowledge.best_sparable_features)
        print(len(self.df))

    def classification_discovery(self):
        score_table = pd.DataFrame(columns=['X', 'Y', 'score'])
        lin_classification_task = None
        assert isinstance(self.tasks, list), 'tasks is not a type of list, instead it is {0}.'.format(type(self.tasks))
        for single_task in self.tasks:
            assert isinstance(single_task, task), 'task is not an instance of task in knowledge discovery.'
            if not isinstance(single_task.definedtask, BaseClassifier):
                continue
            if 'score' in single_task.definedtask.output:
                score_table = score_table.append(single_task.definedtask.output['score'])
            if single_task.definedtask.algo_name == 'sgd':
                lin_classification_task = single_task.definedtask
        if len(score_table) < 1:
            print('no classification scores found.')
            return
        else:
            grouped_score = score_table.groupby(['X', 'Y']).mean()
            grouped_score = grouped_score.reset_index()
            grouped_score = grouped_score[grouped_score['score'] == np.max(grouped_score['score'])]
            self.knowledge.best_sparable_features = grouped_score.head(1)[['X', 'Y']].values.tolist()[0]
            self.knowledge.classification_score_table = score_table

            if lin_classification_task is None:
                return
            self.knowledge.best_sparable_features_linear_clf = lin_classification_task.output['clf'][','.join(self.knowledge.best_sparable_features)]
            if len(self.classdefinition.classes) <= 2:
                self.biclasses_discovery(lin_classification_task)

    def discover(self):
        self.classification_discovery()
