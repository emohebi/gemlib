from gemlib.validation.inputvalidation import *
from gemlib.abstarct.basefunctionality import BaseClassifier, BaseClusterer, BaseRegressor, \
    BasePlotter, BaseFeatureEngineering, BaseConfusionDefinition, BaseDataFilter, BaseDataLoader, \
    BaseRocCurves, BaseTextPreprocessing, BaseTopicModelling
import sys
import multiprocessing
from gemlib.framework.knowledge import BasicKnowledge
import pandas as pd
from gemlib.visualization import cmapsetup
from typing import Dict, List
import traceback
from gemlib.validation import utilities
import logging


class Pipeline(object):
    # creates a pipeline for data processing
    def __init__(self, artifact_folder, set_logger=True):
        cmapsetup.setup()
        self.artifact_folder = artifact_folder
        self.data_loaders:List[BaseDataLoader] = None
        self.classification:BaseClassifier = None
        self.clustering:BaseClusterer = None
        self.regression:BaseRegressor = None
        self.visualization:BasePlotter = None
        self.featureselection = None
        self.df:pd.DataFrame = None
        self.multiproc = False
        self.tasks:List[task] = None
        self.preprocessings:List[preprocessing] = None
        self.knowledge = BasicKnowledge()
        self.dirpath:str = None
        self.resources:Dict = {} # dictionary of data
        self.data_subdir = 'data'
        self.prep_subdir = 'preprocessing'
        self.tasks_subdir = 'tasks'
        self.set_logger = set_logger

    def __call__(self, task):
        return self.apply_single_task(task)

    def setup_pipeline_inputs(self, object):
        try:
            validation = CwdValidation(object, self.artifact_folder)
            self.dirpath = validation.get_valid_cwd()
            if self.set_logger:
                logger = utilities.SetLogger(self.dirpath)
                logger.set()
            # create 'data' folder in cwd for staging purposes
            Path(self.dirpath / self.data_subdir).mkdir(parents=True, exist_ok=True)
            Path(self.dirpath / self.prep_subdir).mkdir(parents=True, exist_ok=True)
            Path(self.dirpath / self.tasks_subdir).mkdir(parents=True, exist_ok=True)

            dataloader_validate = ListOfDataloaderValidation(object)
            self.data_loaders = dataloader_validate.get_valid_list_of_dataloader()
            preprocessing_validate = PreprocessingValidation(object)
            self.preprocessings = preprocessing_validate.get_valid_list_of_preprocessing()
            validation = TasksValidation(object, self.artifact_folder)
            self.tasks = validation.get_valid_list_of_tasks()
        except:
            traceback.print_exc(file=sys.stdout)

    def load_data(self):
        try:
            if self.data_loaders is None:
                utilities._info("dataloader is missing in the pipeline!!!!")
                return
            
            for dataloader in self.data_loaders:
                if utilities.check_if_exists_in_resources(self.resources, dataloader.name):
                    utilities._info(f'skipping {dataloader.name}')
                    self.apply_preprocessing(dataloader.validpreprocessing)
                    continue
                utilities._info('loading data from {0}'.format(dataloader.suffix))
                dataloader.dirpath = self.dirpath # this is cwd path
                dataloader.resources = self.resources # loaded resources (data) so far

                dict_of_data = utilities.resolve_caching_stage(dataloader.cache, dataloader.load(), self.dirpath, self.data_subdir, dataloader.name)
                self.resources.update(dict_of_data)
                if dataloader.validpreprocessing:
                    self.apply_preprocessing(dataloader.validpreprocessing)
            return
        except:
            traceback.print_exc(file=sys.stdout)

    def apply_filter(self, preprocess, df=None):
        try:
            filter = preprocess.definedpreprocessing
            filter.name = preprocess.preprocessing_name
            if utilities.check_if_exists_in_resources(self.resources, filter.name):
                utilities._info(f'skipping {filter.name}')
                return
            if filter is not None:
                filter.dirpath = self.dirpath
                df = utilities.resolve_caching_load(self.resources, filter.input, filter.concat)
                df = utilities.resolve_caching_stage(filter.cache, filter.apply(df), self.dirpath, self.prep_subdir, filter.name)
                self.resources.update(df)
                utilities._info('({}) filtering applied.'.format(preprocess.preprocessing_name))
        except:
            traceback.print_exc(file=sys.stdout)

    def apply_feature_engineering(self, preprocess, df=None):
        try:
            featureengineering = preprocess.definedpreprocessing
            if featureengineering:
                featureengineering.name = preprocess.preprocessing_name
                featureengineering.resources = self.resources
                featureengineering.dirpath = self.dirpath
                if utilities.check_if_exists_in_resources(self.resources, featureengineering.name):
                    utilities._info(f'skipping {featureengineering.name}')
                    return
                if featureengineering.input:
                    df = utilities.resolve_caching_load(self.resources, featureengineering.input, featureengineering.concat)
                    utilities._info(f'the type of input in {featureengineering.name} is: {type(df)}')
                    df = utilities.resolve_caching_stage(featureengineering.cache, featureengineering.apply(df), self.dirpath, self.prep_subdir, featureengineering.name)
                    self.resources.update(df)
                else:
                    df = utilities.resolve_caching_stage(featureengineering.cache, featureengineering.apply(None), self.dirpath, self.prep_subdir, featureengineering.name)
                    self.resources.update(df)
                utilities._info('({}) feature engineering applied.'.format(preprocess.preprocessing_name))
        except:
            traceback.print_exc(file=sys.stdout)

    def apply_classification(self, task:task):
        classification = task.definedtask
        classification.name = task.task_name
        classification.resources = self.resources
        classification.dirpath = self.dirpath if task.cwd is None else task.cwd
        if classification.validpreprocessing:
            self.apply_preprocessing(classification.validpreprocessing)
        # classification.df = self.df.copy(deep=True)
        # classification.target = self.classdefinition.target
        classification.run()
        return

    def apply_clustering(self, task:task):
        clustering = task.definedtask
        if clustering.dimensions is None:
            if len(self.df.columns) <= 20:
                clustering.dimensions = self.df.columns.tolist()
            else:
                utilities._info('too many columns in clustering calculation!!! [max: 20], ignoring clustering...')
                return
        clustering.dirpath = self.dirpath if task.cwd is None else task.cwd
        clustering.df = self.df.copy(deep=True)
        clustering.run()
        return

    def apply_confusion_matrix(self, task:task):
        try:
            confusion = task.definedtask
            confusion.dirpath = self.dirpath if task.cwd is None else task.cwd
            confusion.df = self.df.copy(deep=True)
            confusion.title = task.title
            if confusion.validpreprocessing is not None:
                self.apply_preprocessing(confusion.validpreprocessing)
            self.resources[task.task_name] = confusion.run()
            if confusion.validvisualization:
                for data in self.resources[task.task_name]:
                    # confusion.validvisualization.definedtask.title = data
                    confusion.validvisualization.cwd = confusion.dirpath
                    self.apply_visualization(confusion.validvisualization,
                                             self.resources[task.task_name][data])

            return
        except:
            traceback.print_exc(file=sys.stdout)

    def apply_roc_curve(self, task:task):
        try:
            roc_curve = task.definedtask
            roc_curve.dirpath = self.dirpath if task.cwd is None else task.cwd
            roc_curve.df = self.df.copy(deep=True)
            roc_curve.title = task.title
            self.resources[task.task_name] = roc_curve.run()
        except:
            traceback.print_exc(file=sys.stdout)

    def apply_regression(self, task:task):
        regression = task.definedtask
        regression.dirpath = self.dirpath if task.cwd is None else task.cwd
        regression.df = self.df.copy(deep=True)
        if regression.validpreprocessing is not None:
                self.apply_preprocessing(regression.validpreprocessing)
        
        if regression.features is None:
            if len(regression.df.columns) <= 2000:
                regression.features = list(set(regression.df.columns.tolist()) - set([regression.target]))
            else:
                utilities._info('too many columns in clustering calculation!!! [max: 2000], ignoring regression...')
                return
        else:
            regression.features = utilities.refine_lists(regression.features, 
                                                         regression.df.columns.tolist())
        regression.run()
        return

    def apply_visualization(self, task:task, df=None):
        try:
            plotter = task.definedtask
            if df is not None:
                plotter.df = df
            else:
                plotter.df = self.df.copy(deep=True)
            # if plotter.columns is None:
            #     if len(plotter.df.columns) <= 20:
            #         plotter.columns = plotter.df.columns.tolist()
            #     else:
            #         utilities._info('too many columns in visualization!!! [max: 20], ignoring plotting...')
            #         return
            plotter.dirpath = self.dirpath if task.cwd is None else task.cwd
            if plotter.validpreprocessing is not None:
                self.apply_preprocessing(plotter.validpreprocessing)
            # plotter.df = plotter.df.round(1)
            plotter.run()
            return
        except:
            traceback.print_exc(file=sys.stdout)

    def apply_textpreprocessing(self,preprocess, df=None):
        try:
            txtpreprocessing = preprocess.definedpreprocessing
            if txtpreprocessing is not None:
                txtpreprocessing.dirpath = self.dirpath
                utilities._info(f'text preprocesisng.... len({df.shape[0]})')
                df = txtpreprocessing.apply(df)
                utilities._info(f'({preprocess.preprocessing_name}) txtpreprocessing applied. len({df.shape[0]})')
                return df
        except:
            traceback.print_exc(file=sys.stdout)

    def apply_topicmodelling(self, task:task):
        try:
            tmodelling = task.definedtask
            tmodelling.output_dir = self.dirpath if task.cwd is None else task.cwd
            tmodelling.corpus_object.output_dir = self.dirpath if task.cwd is None else task.cwd
            tmodelling.taskname = task.task_name
            tmodelling.corpus_object.taskname = task.task_name
            tmodelling.df = self.df.copy(deep=True)
            if tmodelling.preprocessing:
                utilities._info(f'going for preprocessing... len({tmodelling.df.shape[0]})')
                self.apply_preprocessing(tmodelling.preprocessing)
            utilities._info(f'going for topic modelling.... len({tmodelling.df.shape[0]})')
            self.resources[task.task_name] = tmodelling.run()
            if tmodelling.validvisualization:
                for data in self.resources[task.task_name]:
                    # confusion.validvisualization.definedtask.title = data
                    tmodelling.validvisualization.cwd = tmodelling.output_dir
                    self.apply_visualization(tmodelling.validvisualization,
                                             self.resources[task.task_name][data])

            return
        except:
            traceback.print_exc(file=sys.stdout)


    def apply_write_to_db(self, task:task):
        db = task.definedtask
        db.df = self.df.copy(deep=True)
        db.write()
        return

    def apply_single_preprocessing(self, process:preprocessing):
        if isinstance(process.definedpreprocessing, BaseFeatureEngineering):
            self.apply_feature_engineering(process)
        elif isinstance(process.definedpreprocessing, BaseDataFilter):
            self.apply_filter(process)
        elif isinstance(process.definedpreprocessing, BaseTextPreprocessing):
            self.apply_textpreprocessing(process)

    def apply_single_task(self, task):
        if isinstance(task.definedtask, BaseClassifier):
            self.apply_classification(task)
        elif isinstance(task.definedtask, BaseClusterer):
            self.apply_clustering(task)
        elif isinstance(task.definedtask, BaseRegressor):
            self.apply_regression(task)
        elif isinstance(task.definedtask, BaseConfusionDefinition):
            self.apply_confusion_matrix(task)
        elif isinstance(task.definedtask, BaseDataLoader):
            self.apply_write_to_db(task)
        elif isinstance(task.definedtask, BaseRocCurves):
            self.apply_roc_curve(task)
        elif isinstance(task.definedtask, BasePlotter):
            self.apply_visualization(task)
        elif isinstance(task.definedtask, BaseTopicModelling):
            self.apply_topicmodelling(task)
        return

    def apply_preprocessing(self, preprocessings=None):
        if preprocessings is None:
            preprocessings = self.preprocessings
        if preprocessings is None:
            return
        try: 
            for process in preprocessings:
                self.apply_single_preprocessing(process)
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            utilities._info("running preprocessing list failed!!!\n {0}".format(exceptionValue))

    def apply_tasks(self):
        if self.tasks is None:
            return
        try:
            if self.multiproc:  # only helpful when tasks complexities are same
                pool = multiprocessing.Pool(multiprocessing.cpu_count())
                pool.map(self, self.tasks)
                pool.close()
                pool.join()
                return
            else:
                for task in self.tasks:
                    self.apply_single_task(task)
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            utilities._info("running tasks failed!!!\n {0}".format(exceptionValue))

    def run(self, object):
        try:
            self.resources = utilities.load_resources_dict(self.dirpath)
            for action in object:
                if action == DATA_LOADER:
                    self.load_data()
                elif action == PREPROCESSING:
                    self.apply_preprocessing()
                elif action == TASKS:
                    self.apply_tasks()
            if len(self.resources) > 0:
                    utilities._info(f'resources {list(self.resources.keys())} loaded.')
                    utilities.stage_resources_dict(self.resources, self.dirpath)
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            utilities._info("running pipeline failed!!!\n {0}".format(exceptionValue))


