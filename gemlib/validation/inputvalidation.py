from collections import OrderedDict
from gemlib.dataloaderlayer.filedataloaders import data_loader
from gemlib.filtering.simple_filter import DataColumnFilter, QueryFilter
from gemlib.classification.linearclassifiers import StochasticGradientDecent, LinearSvmClassifier, \
    LinearLogisticRegressionClassifier, LinearRidgeClassifier, LinearLogisticRegressionCVClassifier
from gemlib.classification.deeplearning import DeepLearning
from gemlib.classification.decisiontree import Decisiontree
from gemlib.classification.svm import SvmSvcClassifier
from gemlib.classification.knn import KnnClassifier
from gemlib.clustering.clusterers import HierClustering, AffinityPropagationClustering
from gemlib.abstarct.basetask import task, preprocessing
from gemlib.regression.regressors import FRegressorAlgorithm, RForest
from gemlib.visualization.barplots import BarPlotting
from gemlib.visualization.heatmaps import CustomHeatMaps
from gemlib.visualization.hexbins import HexbinPlot
from gemlib.visualization.scatterplotmatrix import ScatterPlotMatrix
from gemlib.visualization.boxplots import BoxPlotting
from gemlib.visualization.facetgridplot import Drilldownplot
from gemlib.visualization.scatterplots import ScatterPlot
from gemlib.visualization.lineplots import LinePlot
from gemlib.visualization.histogram import HistPlot
from gemlib.featureengineering.featureengineer import feature_engineer_regex, feature_aggregation, \
    feature_value_mapping, feature_engineer_conditional, \
    feature_concat, feature_value_list_to_string, feature_abse, feature_absolute, \
    feature_mean, feature_se, feature_sqrt, feature_square, feature_std, feature_subtract, feature_sum, \
    text_tokenization, one_hot_encoding, aggregation, text_cleaning, train_test_splitting, text_encoding, \
    factorization, embedding
from gemlib.classification.confusionmatrix import confusion_matrix, accuracy, TP, event, TP_FP, FP
from gemlib.classification.roccurve import RocCurves
#from gemlib.dataloaderlayer.nosqldataloaders import redis_data_writer, redis_data_loader
from gemlib.dataloaderlayer.odbcdataloaders import sql_data_loader
from gemlib.validation import  utilities
import sys
import os
from gemlib.validation.textpreprocessingvalidation import TextPreprocessingValidation
from gemlib.classification.topicmodelling.lda import LDATopicModelling
from gemlib.classification.topicmodelling.corpusgenerator import Corpus
from gemlib.classification.topicmodelling.lsi import LSITopicModelling
from pathlib import Path
from gemlib.dataloaderlayer.dataprocessing import data_concat, data_joiner
from gemlib.validation.keywords import *


class CwdValidation(object):
    def __init__(self, object, artifact_folder):
        self.validate_instance = object
        self.artifact_folder = artifact_folder

    def is_valid_dict(self):
        try:
            assert isinstance(self.validate_instance, dict)
        except AssertionError as e:
            raise AssertionError(
                'type error: should be dictionary but it is {0}, {1}'.format(
                    type(self.validate_instance), e))
        return True

    def get_valid_cwd(self):
        if not self.is_valid_dict():
            utilities._error(0, 'CWD config is not a valid dictionary... try again!!')
        if CWD not in self.validate_instance:
            utilities._error(0, 'working directory is not set correctly... try again!!!')
        dir = Path(self.artifact_folder) / Path(self.validate_instance[CWD]) if \
            self.artifact_folder is not None else \
            Path(self.validate_instance[CWD])

        if not dir.exists():
            dir.mkdir(parents=True)
            utilities._info('[{}] directory created!!!'.format(dir))

        return dir


class DbValidation(object):
    def __init__(self, object):
        self.validate_instance = object

    def initial_db(self):
        port = None

        if self.validate_instance[PATH] is None:
            utilities._info('db ip is not defined. ignored... ')
            return None
        if self.validate_instance[PORT] is None:
            utilities._info('db port is not defined. ignored... ')
            return None
        if utilities.type_is_int(self.validate_instance[PORT]):
            port = int(self.validate_instance[PORT])

        return True

    def get_valid_db(self):
        if not PATH in self.validate_instance:
            utilities._info('critical error: db path is missing in the configuration. ignored...')
            return None
        if not DBTYPE in self.validate_instance:
            utilities._info('Type of db is not defined. ignored...')
            return None
        try:
            db = self.initial_db()
            if db is None:
                utilities._info("not a valid db!!! ignored:{}...".format(self.validate_instance[DBTYPE]))
            return db
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            utilities._info("dataloader initialization failed!!!\n {0}".format(exceptionValue))


class DataloaderValidation(object):
    def __init__(self, object, name=None):
        self.validate_instance = object
        self.name = name

    def initial_data_loader(self):
        replace = None
        dtypes = None
        holdon = None
        head = None
        exclude = None
        definition = None
        addtofilename = None
        preprocessing = None
        keys = None
        port = None
        concat = True
        truthpath = None
        joinon = None
        missed = None
        metafile = None
        how = 'left'
        query = None
        connection_str = None
        type = None
        stage = None
        fillna = None
        drop_na = None
        cache = False
        num_rows = None
        sheet_name = 'Sheet1'

        if SHEET_NAME in self.validate_instance:
            sheet_name = self.validate_instance[SHEET_NAME]

        if NUM_ROWS in self.validate_instance:
            num_rows = self.validate_instance[NUM_ROWS]

        if CACHE in self.validate_instance:
            cache = True if self.validate_instance[CACHE].lower() == 'true' else False

        if DROPNA in self.validate_instance:
            if not isinstance(self.validate_instance[DROPNA], list):
                utilities._info('"drop_na" should be a dictionary of column:value. ignored!!!')
            else:
                drop_na = self.validate_instance[DROPNA]

        if FILLNA in self.validate_instance:
            if not isinstance(self.validate_instance[FILLNA], dict):
                utilities._info('"fill_na" should be a dictionary of column:value. ignored!!!')
            else:
                fillna = {}
                for k in self.validate_instance[FILLNA].keys():
                    if utilities.type_is_int(self.validate_instance[FILLNA][k]):
                        fillna[k] = int(self.validate_instance[FILLNA][k])
                    elif utilities.type_is_float(self.validate_instance[FILLNA][k]):
                        fillna[k] = float(self.validate_instance[FILLNA][k])
                    else:
                        fillna[k] = self.validate_instance[FILLNA][k]

        if REPLACE in self.validate_instance:
            replace = self.validate_instance[REPLACE]

        if QUERY in self.validate_instance:
            query = self.validate_instance[QUERY]
        
        if CONNECTION_STR in self.validate_instance:
            connection_str = self.validate_instance[CONNECTION_STR]

        if TYPE in self.validate_instance:
            type = self.validate_instance[TYPE].lower()

        if DTYPES in self.validate_instance:
            if not isinstance(self.validate_instance[DTYPES], dict):
                utilities._info('"dtype" should be a dictionary of column:type. ignored!!!')
            else:
                dtypes = {}
                for k in self.validate_instance[DTYPES].keys():
                    if self.validate_instance[DTYPES][k].lower() == 'int':
                        dtypes[k] = int
                    elif self.validate_instance[DTYPES][k].lower() == 'float':
                        dtypes[k] = float
                    elif self.validate_instance[DTYPES][k].lower() == 'bool':
                        dtypes[k] = bool
        if NUMFRAMESHOLD in self.validate_instance:
            if not utilities.type_is_int(self.validate_instance[NUMFRAMESHOLD]):
                utilities._info('"numframes_hold" should be of type of integer. ignored!!!')
            else:
                if HEAD not in self.validate_instance:
                    utilities._info('"head" should be defined along with "numframes_hold", ignored!!!')
                elif not utilities.type_is_bool(self.validate_instance[HEAD]):
                    utilities._info('"head" should be of type bool, ignored!!!')
                else:
                    holdon = int(self.validate_instance[NUMFRAMESHOLD])
                    head = True if self.validate_instance[HEAD].lower() == 'true' else False

        if NUMFRAMEEXCLUDE in self.validate_instance:
            exclude = int(self.validate_instance[NUMFRAMEEXCLUDE])
            head = True if self.validate_instance[HEAD].lower() == 'true' else False

        if SAVE in self.validate_instance:
            if not utilities.type_is_bool(self.validate_instance[SAVE]):
                utilities._info('"stage" should be of type bool, ignored!!!')
            else:
                stage = True if self.validate_instance[SAVE].lower() == 'true' else False

        if CONCAT in self.validate_instance:
            if not utilities.type_is_bool(self.validate_instance[CONCAT]):
                utilities._info('"concat" should be of type bool, ignored!!!')
            else:
                concat = True if self.validate_instance[CONCAT].lower() == 'true' else False

        if DEFINITION in self.validate_instance:
            if not isinstance(self.validate_instance[DEFINITION], dict):
                utilities._info('"definition" should be a dictionary of column:value. ignored!!!')
            else:
                definition = self.validate_instance[DEFINITION]

        if ADDTOFILENAME in self.validate_instance:
            addtofilename = self.validate_instance[ADDTOFILENAME]

        if PREPROCESSING in self.validate_instance:
            preprocessing_validate = PreprocessingValidation(self.validate_instance, self.name)
            preprocessing = preprocessing_validate.get_valid_list_of_preprocessing()

        if KEYS in self.validate_instance:
            if not (isinstance(self.validate_instance[KEYS], dict) or
                    isinstance(self.validate_instance[KEYS], list)):
                utilities._info('"Keys" should be of type dictionary or list. ignored!!!')
            else:
                keys = self.validate_instance[KEYS]
        if PORT in self.validate_instance and utilities.type_is_int(self.validate_instance[PORT]):
            port = int(self.validate_instance[PORT])

        if METADATAPATH in self.validate_instance:
            if not os.path.isfile(self.validate_instance[METADATAPATH]):
                raise FileNotFoundError
            metafile = self.validate_instance[METADATAPATH]

        if TRUTH in self.validate_instance:
            if PATH in self.validate_instance[TRUTH]:
                truthpath = self.validate_instance[TRUTH][PATH]
            else:
                utilities._info('truth path for data is not defined... ignoring truth.')
            if JOINON in self.validate_instance[TRUTH]:
                if not isinstance(self.validate_instance[TRUTH][JOINON], list):
                    utilities._info(f'"joinon" option should be a list. '
                          f'But It is {type(self.validate_instance[TRUTH][JOINON])}')
                else:
                    joinon = self.validate_instance[TRUTH][JOINON]
            if IFMISSED in self.validate_instance[TRUTH]:
                if not isinstance(self.validate_instance[TRUTH][IFMISSED], dict):
                    utilities._info('"ifnotavailable" should be a dictionary of column:value. ignored!!!')
                else:
                    missed = self.validate_instance[TRUTH][IFMISSED]
            if 'how' in self.validate_instance[TRUTH]:
                how = self.validate_instance[TRUTH]['how']

        if self.validate_instance[TYPE].lower() in ['csv']:
            if self.validate_instance[PATH] is None:
                utilities._info('data loader path is not defined. terminating... ')
                return None
            if not os.path.isfile(self.validate_instance[PATH]) and \
                    not os.path.isdir(self.validate_instance[PATH]):
                utilities._info('data loader path is not a path to a file/directory!!!. terminating...')
                return None

        if self.validate_instance[TYPE].lower() == 'sql':
            return sql_data_loader(connection_string=connection_str, definition=definition,
                               replace=replace, dtypes=dtypes, hold_on=holdon,
                               head=head, exclude=exclude, addtofilename=addtofilename,
                               validpreprocessing=preprocessing, concat=concat,
                               truthpath=truthpath, joinon=joinon, missed=missed,
                               metafile=metafile, suffix=self.validate_instance[TYPE].lower(),
                               how=how, query=query, stage=stage, fillna=fillna, drop_na=drop_na,
                               name=self.name)
        elif self.validate_instance[TYPE].lower() == 'concat':
            return data_concat(path=None, definition=definition,
                               dtypes=dtypes, hold_on=holdon, head=head, exclude=exclude,
                               validpreprocessing=preprocessing, concat=concat, joinon=joinon,
                               suffix=self.validate_instance[TYPE].lower(),
                               how=how, fillna=fillna, drop_na=drop_na, cache=cache,
                               name=self.name, num_rows=num_rows, sheet_name=sheet_name)
        elif self.validate_instance[TYPE].lower() == 'join':
            return data_joiner(path=None, definition=definition,
                               dtypes=dtypes, hold_on=holdon, head=head, exclude=exclude,
                               validpreprocessing=preprocessing, concat=concat, joinon=joinon,
                               suffix=self.validate_instance[TYPE].lower(),
                               how=how, fillna=fillna, drop_na=drop_na, cache=cache,
                               name=self.name, num_rows=num_rows, sheet_name=sheet_name)
        else:
            return data_loader(self.validate_instance[PATH], definition=definition,
                               dtypes=dtypes, hold_on=holdon, head=head, exclude=exclude,
                               validpreprocessing=preprocessing, concat=concat, joinon=joinon,
                               suffix=self.validate_instance[TYPE].lower(),
                               how=how, fillna=fillna, drop_na=drop_na, cache=cache,
                               name=self.name, num_rows=num_rows, sheet_name=sheet_name)

    def get_valid_dataloader(self):
        #if not PATH in self.validate_instance:
        #    utilities._info('critical error: dataloader path is missing in the configuration. terminating...')
        #    return None
        if not TYPE in self.validate_instance:
            utilities._info('Type of data loader is not defined. terminating...')
            return None
        try:
            data_loader = self.initial_data_loader()
            if data_loader is None:
                utilities._info("not a valid dataloader!!! ignored path:{}...".format(self.validate_instance[PATH]))
            return data_loader
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            utilities._info("dataloader initialization failed!!!\n {0}".format(exceptionValue))


class ListOfDataloaderValidation(object):
    def __init__(self, object):
        self.validate_instance = object
        self.dataloaderlist = []

    def initial_dataloaders(self):
        for data_name in self.validate_instance[DATA_LOADER]:
            data_loader = self.validate_instance[DATA_LOADER][data_name]
            valid_dataloader = DataloaderValidation(data_loader, data_name)
            valid_dataloader = valid_dataloader.get_valid_dataloader()
            if valid_dataloader is not None:
                self.dataloaderlist.append(valid_dataloader)

    def get_valid_list_of_dataloader(self):
        if not DATA_LOADER in self.validate_instance:
            return None
        if len(self.validate_instance[DATA_LOADER]) < 1:
            utilities._info('there is not any valid dataloader defined on input data.')
            return None
        try:
            self.initial_dataloaders()
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            utilities._info("dataloaders initialization failed!!!\n {0}".format(exceptionValue))

        if len(self.dataloaderlist) < 1:
            utilities._info("not valid dataloaders!!! dataloaders are ignored ...")
            return None
        return self.dataloaderlist


class FeatureEngineeringValidation(object):
    def __init__(self, object):
        self.validate_instance = object

    def initial_feature_engineering(self):
        save = False
        definition = None
        durationthr = -1
        initialbuffer = -1
        keepcolumnsvals = None
        axis = 1
        truth_col = None
        input = None
        groupby = None
        columns = None
        column = None
        concat = False
        x_cols = None
        y_col = None
        test_ratio = 0.2
        tokenizer = None
        sequence_length = None
        chunk_size= None
        device = 'cuda'
        models = {'1':'roberta-large-nli-stsb-mean-tokens', 
                  '3':'bert-large-nli-stsb-mean-tokens'}

        if MODELS in self.validate_instance:
            if isinstance(self.validate_instance[MODELS], dict) or isinstance(self.validate_instance[MODELS], OrderedDict):
                models = self.validate_instance[MODELS]
            else:
                utilities._error(ValueError, f'"{MODELS}" should be of type dictionary try again!')

        if DEVICE in self.validate_instance:
            device = self.validate_instance[DEVICE]

        if TOKENIZER in self.validate_instance:
            tokenizer = self.validate_instance[TOKENIZER]

        if CHUNK_SIZE in self.validate_instance:
            chunk_size = self.validate_instance[CHUNK_SIZE]

        if SEQUENCE_LEN in self.validate_instance:
            sequence_length = self.validate_instance[SEQUENCE_LEN]

        if COLUMN in self.validate_instance:
            column = self.validate_instance[COLUMN]

        if TEST_RATIO in self.validate_instance:
            test_ratio = self.validate_instance[TEST_RATIO]

        if X_COLS in self.validate_instance:
            if not isinstance(self.validate_instance[X_COLS], list):
                utilities._error(ValueError, f'"{X_COLS}" should be of type list. try again!')
            x_cols = self.validate_instance[X_COLS]

        if Y_COL in self.validate_instance:
            if not isinstance(self.validate_instance[Y_COL], str):
                utilities._error(ValueError, f'"{Y_COL}" should be of type str. try again!')
            y_col = self.validate_instance[Y_COL]

        if CONCAT in self.validate_instance:
            if not utilities.type_is_bool(self.validate_instance[CONCAT]):
                utilities._info('"concat" should be of type bool, ignored!!!')
            else:
                concat = True if self.validate_instance[CONCAT].lower() == 'true' else False

        if COLUMNS in self.validate_instance:
            if not isinstance(self.validate_instance[COLUMNS], list):
                utilities._error(ValueError, f'"{COLUMNS}" should be of type list. try again!')
            columns = self.validate_instance[COLUMNS]

        if GROUPBY in self.validate_instance:
            if not isinstance(self.validate_instance[GROUPBY], list):
                utilities._error(ValueError, f'"{GROUPBY}" should be of type list. try again!')
            groupby = self.validate_instance[GROUPBY]

        if INPUT_ in self.validate_instance:
            input = self.validate_instance[INPUT_]

        if SAVE in self.validate_instance:
            if self.validate_instance[SAVE].lower() == 'true':
                save = True
        if DEFINITION in self.validate_instance:
            if isinstance(self.validate_instance[DEFINITION], dict):
                definition = self.validate_instance[DEFINITION]
            else:
                utilities._info('feature engineering definition should be a dictionary!!!, ignored.')
                return None

        if KEEPCOLUMNVALUES in self.validate_instance:
            if isinstance(self.validate_instance[KEEPCOLUMNVALUES], list):
                keepcolumnsvals = self.validate_instance[KEEPCOLUMNVALUES]
            else:
                utilities._info(f'warning: {KEEPCOLUMNVALUES} should be of type list. ignored!!!')

        if AXIS in self.validate_instance:
            axis = int(self.validate_instance[AXIS])

        # TODO: move all string types to keywords.py 
        if self.validate_instance[TYPE].lower() == 'regex':
            return feature_engineer_regex(definition=definition, save=save)
        elif self.validate_instance[TYPE].lower() == 'mapping':
            return feature_value_mapping(definition=definition, save=save)
        elif self.validate_instance[TYPE].lower() == 'aggregation':
            if not GROUPBY in self.validate_instance:
                utilities._info('"groupby" is missing in aggregation definition!!! ignored.')
                return None
            elif isinstance(self.validate_instance[GROUPBY], list):
                    columns = self.validate_instance[GROUPBY]
                    return aggregation(definition=definition,
                                               groupby=columns)
            else:
                utilities._info('"groupby" should be a list of columns!!! ignored.')
                return None
        elif self.validate_instance[TYPE].lower() == 'conditional':
            return feature_engineer_conditional(definition=definition, save=save)
        elif self.validate_instance[TYPE].lower() == 'concatenate':
            return feature_concat(definition=definition, save=save)
        elif self.validate_instance[TYPE].lower() == 'list_to_string':
            return feature_value_list_to_string(definition=definition, save=save)
        elif self.validate_instance[TYPE].lower() == 'event_extraction':
            if TIMECOLUMN not in self.validate_instance:
                utilities._info(f'error: {TIMECOLUMN} is not defined in the process. feature engineering ignored!!!')
                return None
            if TRUTH in self.validate_instance:
                truth_col = self.validate_instance[TRUTH]
            if PREDICTED not in self.validate_instance:
                utilities._info(f'error: {PREDICTED} is not defined in the process. feature engineering ignored!!!')
                return None
            if TARGETEVENTVALUE not in self.validate_instance:
                utilities._info(f'error: {TARGETEVENTVALUE} is not defined in the process. feature engineering ignored!!!')
                return None
            if INITIALBUFFER in self.validate_instance:
                if utilities.type_is_int(self.validate_instance[INITIALBUFFER]):
                  initialbuffer = int(self.validate_instance[INITIALBUFFER])
                else:
                    utilities._info(f'warning: {INITIALBUFFER} should be of type int. ignored!!!')

        elif self.validate_instance[TYPE].lower() == SE:
            return feature_se(definition=definition, save=save)
        elif self.validate_instance[TYPE].lower() == ABSE:
            return feature_abse(definition=definition, save=save)
        elif self.validate_instance[TYPE].lower() == SUBTRACT:
            return feature_subtract(definition=definition, save=save)
        elif self.validate_instance[TYPE].lower() == MEAN:
            return feature_mean(definition=definition, save=save, axis=axis)
        elif self.validate_instance[TYPE].lower() == SQUARE:
            return feature_square(definition=definition, save=save)
        elif self.validate_instance[TYPE].lower() == ABSOLUTE:
            return feature_absolute(definition=definition, save=save)
        elif self.validate_instance[TYPE].lower() == SUM:
            return feature_sum(definition=definition, save=save, axis=axis)
        elif self.validate_instance[TYPE].lower() == SQRT:
            return feature_sqrt(definition=definition, save=save)
        elif self.validate_instance[TYPE].lower() == STD:
            return feature_std(definition=definition, save=save, axis=axis)
        elif self.validate_instance[TYPE].lower() == EMBEDDING:
            all_req_props = [EMBEDDING_DIM, EMBEDDING_PATH, NUM_VOCABS, TOKENIZER]
            for prop in all_req_props:
                if not prop in self.validate_instance:
                    utilities._error(ValueError, f'"{prop}" is missing in embedding. '
                    'please define it and try again!')
            self.validate_instance[EMBEDDING_DIM] = int(self.validate_instance[EMBEDDING_DIM])
            self.validate_instance[NUM_VOCABS] = int(self.validate_instance[NUM_VOCABS])
            if not os.path.isfile(self.validate_instance[EMBEDDING_PATH]):
                utilities._error(ValueError, f'embedding_path : {self.validate_instance[EMBEDDING_PATH]} '
                'is not valid. please fix it and try again!')
            return embedding(tokenizer=tokenizer,
                                  embedding_dim=self.validate_instance[EMBEDDING_DIM],
                                  embedding_path=self.validate_instance[EMBEDDING_PATH],
                                  num_vocab=self.validate_instance[NUM_VOCABS],
                                  input=input)
        elif self.validate_instance[TYPE].lower() == ONE_HOT:
            return one_hot_encoding(input=input, columns=columns)
        elif self.validate_instance[TYPE].lower() == TEXT_CLEANING:
            return text_cleaning(input=input, columns=columns)
        elif self.validate_instance[TYPE].lower() == TRAIN_TEST_SPLIT:
            if x_cols and y_col:
                return train_test_splitting(x_cols=x_cols, y_col=y_col, input=input, test_ratio=test_ratio)
            else:
                utilities._error(ValueError, f'both {X_COLS} and {Y_COL} should be defined for {TRAIN_TEST_SPLIT}')
        elif self.validate_instance[TYPE].lower() == TEXT_TOKENIZATION:
            return text_tokenization(input=input, concat=concat, tokenizer=tokenizer, column=column,
                                     sequence_lenght=sequence_length)
        elif self.validate_instance[TYPE].lower() == TEXT_ENCODING:
            return text_encoding(input=input, concat=concat, column=column, chunk_size=chunk_size, device=device, models=models)
        elif self.validate_instance[TYPE].lower() == FACTORIZATION:
            return factorization(input=input, concat=concat, column=column)
        else:
            utilities._info('"{}" is not a valid feature engineering type!!!, ignored.'.format(self.validate_instance[TYPE]))
        return None

    def get_valid_feature_engineering(self):
        if not TYPE in self.validate_instance:
            utilities._info('"type" is not defined in feature engineering on process!!!, ignored.')
            return None
        # if not DEFINITION in self.validate_instance:
        #     utilities._info('"definition" of feature engineering is missing!!!, ignored.')
        #     return None
        try:
            fengineering = self.initial_feature_engineering()
            if fengineering is None:
                utilities._info("not a valid feature engineering set up!!! ignored.")
            return fengineering
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            utilities._info("feature engineering initialization failed!!!\n {0}".format(exceptionValue))


class DatafilterValidation(object):
    def __init__(self, object):
        self.validate_instance = object

    def initial_data_filter(self):
        save = False
        input = None

        if INPUT_ in self.validate_instance:
            input = self.validate_instance[INPUT_]

        if SAVE in self.validate_instance:
            if self.validate_instance[SAVE].lower() == 'true':
                save = True
        if self.validate_instance[TYPE].lower() == 'column':
            if COLUMNS in self.validate_instance:
                if isinstance(self.validate_instance[COLUMNS], list):
                    return DataColumnFilter(list_col_names=self.validate_instance[COLUMNS], 
                                            save=save, input=input)
        elif self.validate_instance[TYPE].lower() == 'query':
            if QUERY in self.validate_instance:
                if isinstance(self.validate_instance[QUERY], str):
                    if COLUMNS in self.validate_instance:
                        return QueryFilter(list_col_names=self.validate_instance[COLUMNS],
                                       query=self.validate_instance[QUERY], save=save,
                                       input=input)
                    else:
                        return QueryFilter(query=self.validate_instance[QUERY], save=save,
                                           input=input)
        return None

    def get_valid_filter(self):
        if not PROCESS in self.validate_instance:
            utilities._info('there is not filtering on input data.')
            return None
        if not TYPE in self.validate_instance:
            utilities._info('"type" is not defined in filter!!!, ignored.')
            return None
        try:
            filter = self.initial_data_filter()
            if filter is None:
                utilities._info("not a valid filter!!! filtering ignored...")
            return filter
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            utilities._info("data filtering initialization failed!!!\n {0}".format(exceptionValue))


class ClassificationValidation(object):
    def __init__(self, object, name=None):
        self.validate_instance = object
        self.name = name

    def initial_classifier(self):
        dimensions = None
        sequence_length = None
        input = None
        x_train = None
        x_test = None
        y_train = None
        y_test = None
        embedding_matrix = None
        epoch = 100
        num_classes = None
        model_name = None
        mode = None
        preprocessing = None
        batch_size= None
        model_path = None
        y_col_map = None
        top_n = 1
        lr = 0.001
        decay_n_epoch = 5

        if DECAY_N_EPOCH in self.validate_instance:
            decay_n_epoch = self.validate_instance[DECAY_N_EPOCH] 

        if LEARNING_RATE in self.validate_instance:
            lr = self.validate_instance[LEARNING_RATE] 

        if TOP_N in self.validate_instance:
            top_n = self.validate_instance[TOP_N]

        if Y_COL_MAP in self.validate_instance:
            y_col_map = self.validate_instance[Y_COL_MAP]

        if MODEL_PATH in self.validate_instance:
            model_path = self.validate_instance[MODEL_PATH]

        if BATCH_SIZE in self.validate_instance:
            batch_size = self.validate_instance[BATCH_SIZE]

        if MODEL_NAME in self.validate_instance:
            model_name = self.validate_instance[MODEL_NAME]

        if MODE in self.validate_instance:
            mode = self.validate_instance[MODE]

        if X_TRAIN in self.validate_instance:
            x_train = self.validate_instance[X_TRAIN]

        if X_TEST in self.validate_instance:
            x_test = self.validate_instance[X_TEST]

        if Y_TRAIN in self.validate_instance:
            y_train = self.validate_instance[Y_TRAIN]

        if Y_TEST in self.validate_instance:
            y_test = self.validate_instance[Y_TEST]

        if INPUT_ in self.validate_instance:
            input = self.validate_instance[INPUT_]

        if EMBEDDING_MATRIX in self.validate_instance:
            embedding_matrix = self.validate_instance[EMBEDDING_MATRIX]

        if EPOCH in self.validate_instance:
            epoch = self.validate_instance[EPOCH]

        if NUM_CLASSES in self.validate_instance:
            num_classes = self.validate_instance[NUM_CLASSES]

        if SEQUENCE_LEN in self.validate_instance:
            sequence_length = self.validate_instance[SEQUENCE_LEN]

        if COLUMNS in self.validate_instance.keys():
            if isinstance(self.validate_instance[COLUMNS], list) and \
                            len(self.validate_instance[COLUMNS]) > 1:
                dimensions = self.validate_instance[COLUMNS]
            else:
                utilities._info('either "columns" in classification is not a list or it has less than 2 element!!!')

        if PREPROCESSING in self.validate_instance:
            preprocessing_validate = PreprocessingValidation(self.validate_instance, self.name)
            preprocessing = preprocessing_validate.get_valid_list_of_preprocessing()

        if self.validate_instance[ALGORITHM].lower() == 'sgd':
            return StochasticGradientDecent(dimensions=dimensions)
        elif self.validate_instance[ALGORITHM].lower() == 'dtree':
            return Decisiontree(dimensions=dimensions)
        elif self.validate_instance[ALGORITHM].lower() == 'svc':
            kernel = self.validate_instance[KERNEL].lower() if KERNEL in \
                                                               self.validate_instance.keys() else None
            return SvmSvcClassifier(dimensions=dimensions, kernel=kernel)
        elif self.validate_instance[ALGORITHM].lower() == 'knn':
            return KnnClassifier(dimensions=dimensions)
        elif self.validate_instance[ALGORITHM].lower() == 'linsvc':
            return LinearSvmClassifier(dimensions=dimensions)
        elif self.validate_instance[ALGORITHM].lower() == 'linlogreg':
            return LinearLogisticRegressionClassifier(dimensions=dimensions)
        elif self.validate_instance[ALGORITHM].lower() == 'linlogregcv':
            return LinearLogisticRegressionCVClassifier(dimensions=dimensions)
        elif self.validate_instance[ALGORITHM].lower() == 'ridge':
            return LinearRidgeClassifier(dimensions=dimensions)
        elif self.validate_instance[ALGORITHM].lower() == DEEPLEARNING:
            return DeepLearning(input=input, x_train=x_train, x_test=x_test,
                                y_train=y_train, y_test=y_test, epoch=epoch,
                                num_classes=num_classes, embedding_matrix=embedding_matrix,
                                sequence_lenght=sequence_length, model_name=model_name,
                                mode=mode, validpreprocessing=preprocessing, batch_size=batch_size,
                                model_path=model_path, y_col_map=y_col_map, top_n=top_n, lr=lr, 
                                decay_n_epoch=decay_n_epoch)

        return None

    def get_valid_classifier(self):
        try:
            classifier = self.initial_classifier()
            if classifier is None:
                utilities._info("not a valid classifier!!! classification ignored ...")
            return classifier
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            utilities._info("classifier initialization failed!!!\n {0}".format(exceptionValue))


class ClusteringValidation(object):
    def __init__(self, object):
        self.validate_instance = object

    def initial_clusterer(self):
        dimensions = None
        target = None
        filter = None
        if COLUMNS in self.validate_instance.keys():
            if isinstance(self.validate_instance[COLUMNS], list) and \
                            len(self.validate_instance[COLUMNS]) > 0:
                dimensions = self.validate_instance[COLUMNS]
            else:
                utilities._info('either "columns" in classification is not a list or it has less than 1 element!!!')

        if CLASSDEFINITION in self.validate_instance.keys():
            target = self.validate_instance[CLASSDEFINITION]  # this is for cluster

        if FILTER in self.validate_instance.keys():
            filter = self.validate_instance[FILTER]  # this is for cluster

        if self.validate_instance[ALGORITHM].lower() == 'hca':
            return HierClustering(dimensions=dimensions, target=target, filter=filter,
                                  algo_name='Hierarchical')
        elif self.validate_instance[ALGORITHM].lower() == 'affpro':
            return AffinityPropagationClustering(dimensions=dimensions, target=target, filter=filter,
                                                 algo_name='AffinityPropagation')

        return None

    def get_valid_clusterer(self):
        try:
            clusterer = self.initial_clusterer()
            if clusterer is None:
                utilities._info("not a valid clusterer!!! clustering ignored ...")
            return clusterer
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            utilities._info("clusterer initialization failed!!!\n {0}".format(exceptionValue))


class RocCurveValidation(object):
    def __init__(self, object):
        self.validate_instance = object

    def initial_roc_curve(self):
        groupby = None
        preprocessing = None
        truth = None
        predicted = None
        title = None
        thresholds = []
        multi_plot = False
        scatter_point = None

        if TRUTH in self.validate_instance and PREDICTED in self.validate_instance:
            truth = self.validate_instance[TRUTH]
            predicted = self.validate_instance[PREDICTED]

        if GROUPBY in self.validate_instance:
            if isinstance(self.validate_instance[GROUPBY], list):
                groupby = self.validate_instance[GROUPBY]
            else:
                utilities._info('either "columns" in confusion is not a list or empty!!!')
        if TITLE in self.validate_instance:
            title = self.validate_instance[TITLE]
        if PREPROCESSING in self.validate_instance:
            preprocessing_validate = PreprocessingValidation(self.validate_instance)
            preprocessing = preprocessing_validate.get_valid_list_of_preprocessing()

        if THRESHOLDS in self.validate_instance:
            if isinstance(self.validate_instance[THRESHOLDS], list):
                thresholds = [float(x) for x in self.validate_instance[THRESHOLDS]]
            else:
                utilities._info('either "threshold" in ROC curve is not a list or empty!!!')
        if 'figsize' in self.validate_instance:
            if isinstance(self.validate_instance['figsize'], list):
                figsize = tuple([float(x) for x in self.validate_instance['figsize']])
        if 'fontsize' in self.validate_instance:
            fontsize = float(self.validate_instance['fontsize'])
        if MULTIPLOT in self.validate_instance:
            if self.validate_instance[MULTIPLOT].lower() == 'true':
                multi_plot = True
        if 'scatter_point' in self.validate_instance:
            scatter_point = [float(x) for x in self.validate_instance['scatter_point']]
        return RocCurves(truth=truth, predicted=predicted, groupby=groupby,
                         validpreprocessing=preprocessing, title=title, thresholds=thresholds,
                         multi_plot=multi_plot, figsize=figsize, fontsize=fontsize,
                         scatter_point=scatter_point)

    def get_valid_roc_curve(self):

        try:
            roc = self.initial_roc_curve()
            if roc is None:
                utilities._info("Not a valid ROC curve!!! roc curve is terminated...")
            return roc
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            utilities._info("Roc Curve initialization failed :((!!!\n {0}".format(exceptionValue))


class ConfusionMatrixValidation(object):
    def __init__(self, object):
        self.validate_instance = object

    def initial_confusion(self):
        columnindex = None
        rowindex = None
        groupby = None
        preprocessing = None
        visualization = None
        truth = None
        predicted = None
        title = None
        visualization_type = None
        column_map = None
        keepcolumnvalues = None

        if 'keepcolumnvalues' in self.validate_instance:
            keepcolumnvalues = self.validate_instance['keepcolumnvalues']

        if 'column_map' in self.validate_instance:
            column_map = self.validate_instance['column_map']
        if TRUTH in self.validate_instance and PREDICTED in self.validate_instance:
            truth = self.validate_instance[TRUTH]
            predicted = self.validate_instance[PREDICTED]

        if COLUMNINDEX in self.validate_instance:
            if isinstance(self.validate_instance[COLUMNINDEX], list):
                columnindex = self.validate_instance[COLUMNINDEX]
            else:
                utilities._info('either "conlumnindex" in confusion is not a list or empty!!!')
        if ROWINDEX in self.validate_instance:
            if isinstance(self.validate_instance[ROWINDEX], list):
                rowindex = self.validate_instance[ROWINDEX]
            else:
                utilities._info('either "rowindex" in confusion is not a list or empty!!!')
        if GROUPBY in self.validate_instance:
            if isinstance(self.validate_instance[GROUPBY], list):
                groupby = self.validate_instance[GROUPBY]
            else:
                utilities._info('either "columns" in confusion is not a list or empty!!!')
        if TITLE in self.validate_instance:
            title = self.validate_instance[TITLE]
        if PREPROCESSING in self.validate_instance:
            preprocessing_validate = PreprocessingValidation(self.validate_instance)
            preprocessing = preprocessing_validate.get_valid_list_of_preprocessing()
        if VISUALIZATION in self.validate_instance:
            visualization_valid = PlotValidation(self.validate_instance[VISUALIZATION])
            visualization = task(visualization_valid.get_valid_plotter(), '', None)
        if VISUALIZATIONTYPE in self.validate_instance:
            visualization_type = self.validate_instance[VISUALIZATIONTYPE]
            assert(visualization_type in ['bar', 'heatmap'])
        if SUBTASK in self.validate_instance:
            if self.validate_instance[SUBTASK].lower() == 'accuracy':
                return accuracy(truth=truth, predicted=predicted, rowindex=rowindex,
                                columnindex=columnindex, groupby=groupby,
                                validpreprocessing=preprocessing, validvisualization=visualization,
                                type=visualization_type, column_map=column_map)
            elif self.validate_instance[SUBTASK].lower() == 'tp':
                return TP(truth=truth, predicted=predicted, rowindex=rowindex,
                                columnindex=columnindex, groupby=groupby,
                                validpreprocessing=preprocessing, validvisualization=visualization,
                                type=visualization_type, column_map=column_map)
            elif self.validate_instance[SUBTASK].lower() == 'fp':
                return FP(truth=truth, predicted=predicted, rowindex=rowindex,
                                columnindex=columnindex, groupby=groupby,
                                validpreprocessing=preprocessing, validvisualization=visualization,
                                type=visualization_type, column_map=column_map)
            elif self.validate_instance[SUBTASK].lower() == 'tp_fp':
                return TP_FP(truth=truth, predicted=predicted, rowindex=rowindex,
                                columnindex=columnindex, groupby=groupby,
                                validpreprocessing=preprocessing, validvisualization=visualization,
                                type=visualization_type)
            elif self.validate_instance[SUBTASK].lower() == 'event':
                if TIMECOLUMN in self.validate_instance:
                    return event(timecolumn=self.validate_instance[TIMECOLUMN], groupby=groupby,
                                 validpreprocessing=preprocessing, validvisualization=visualization,
                                 column_map=column_map, keepcolumnvalues=keepcolumnvalues)
                utilities._info("error: TIMECOLUMN is missing in event task definition. ignored!!!")
                return

        return confusion_matrix(truth=truth, predicted=predicted, rowindex=rowindex,
                                columnindex=columnindex, groupby=groupby,
                                validpreprocessing=preprocessing, validvisualization=visualization,
                                title=title)

    def get_valid_confusion(self):
        # if TRUTH not in self.validate_instance or \
        #                 PREDICTED not in self.validate_instance:
        #     utilities._info('neither truth nor predicted are defined in confusion. terminating...')
        #     return None
        try:
            confusion = self.initial_confusion()
            if confusion is None:
                utilities._info("not a valid confusion!!! confusoin ignored ...")
            return confusion
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            utilities._info("confusion matrix initialization failed!!!\n {0}".format(exceptionValue))


class RegressionValidation(object):
    def __init__(self, object, name=None):
        self.validate_instance = object
        self.name = name

    def initial_regressor(self):
        dimensions = None
        target = None
        to_numerical = None
        features = None

        if TARGET in self.validate_instance:
            target = self.validate_instance[TARGET]
        else:
            raise ValueError(f'"{TARGET}" must be defined in the config file. try again!')

        if TO_NUMERICAL in self.validate_instance:
            if isinstance(self.validate_instance[TO_NUMERICAL], list) and \
                            len(self.validate_instance[TO_NUMERICAL]) > 0:
                to_numerical = self.validate_instance[TO_NUMERICAL]

        if COLUMNS in self.validate_instance.keys():
            if isinstance(self.validate_instance[COLUMNS], list) and \
                            len(self.validate_instance[COLUMNS]) > 1:
                features = self.validate_instance[COLUMNS]
            else:
                utilities._info('either "columns" in regressor is not a list or it has less than 2 element!!!')
        
        if PREPROCESSING in self.validate_instance:
            preprocessing_validate = PreprocessingValidation(self.validate_instance, self.name)
            preprocessing = preprocessing_validate.get_valid_list_of_preprocessing()

        if self.validate_instance[ALGORITHM].lower() == 'random_forest':
            n_estimators = 1000 
            n_jobs = 3
            oob_score = True
            max_depth = 10
            if 'n_estimator' in self.validate_instance:
                n_estimators = int(self.validate_instance['n_estimator'])
            if 'n_jobs' in self.validate_instance:
                n_jobs = int(self.validate_instance['n_jobs'])
            if 'max_depth' in self.validate_instance:
                max_depth = int(self.validate_instance['max_depth'])

            return RForest(features=features, algo_name='random_forest',
                           n_estimator=n_estimators,
                           target=target, n_jobs=n_jobs,
                           max_depth=max_depth,
                           validpreprocessing=preprocessing)

        return None

    def get_valid_regressor(self):
        try:
            regressor = self.initial_regressor()
            if regressor is None:
                utilities._info("not a valid regressor!!! regression ignored ...")
            return regressor
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            utilities._info("regressor initialization failed!!!\n {0}".format(exceptionValue))


class PlotValidation(object):
    def __init__(self, object):
        self.validate_instance = object

    def initial_plot(self):
        columns = None
        target = None
        x = None
        y = None
        z = None
        d1 = None
        d2 = None
        d3 = None
        fun = None
        order = None
        preprocessing = None
        title = 'untitled'
        x_label = 'unlabeled'
        y_label = 'unlabeled'
        legend_names = None
        legend = ''
        s = None
        annot = False
        ticks_max_x = None
        ticks_max_y = None
        ticks_min_x = None
        ticks_min_y = None
        figsize = (5, 5)
        min_v = 0
        max_v = 20
        transpose = False
        groupby = None
        add_error = False
        bins = 10
        x_col = None
        fontsize = 8
        linewidth = 2
        padsize = 1
        multiaxis = False
        alpha = 1.0
        stacked = False
        binsize = None

        if PLOT not in self.validate_instance:
            utilities._info('there is no plot defined in visualization. terminating...')
            return None
        if CLASSDEFINITION in self.validate_instance:
            target = self.validate_instance[CLASSDEFINITION]
        if COLUMNS in self.validate_instance.keys():
            if isinstance(self.validate_instance[COLUMNS], list) and \
                            len(self.validate_instance[COLUMNS]) >= 1:
                columns = self.validate_instance[COLUMNS]
            else:
                utilities._info('either "columns" in plotter is not a list or it has less than 2 element!!!')
        if XCOL in self.validate_instance.keys():
            if isinstance(self.validate_instance[XCOL], str):
                x_col = self.validate_instance[XCOL]
            else:
                utilities._info('x_col should be a string referring to a column in dataframe')
        if PREPROCESSING in self.validate_instance:
            preprocessing_validate = PreprocessingValidation(self.validate_instance)
            preprocessing = preprocessing_validate.get_valid_list_of_preprocessing()
        if GROUPBY in self.validate_instance:
            groupby = self.validate_instance[GROUPBY]
        if 'stacked' in self.validate_instance:
            stacked = True if self.validate_instance['stacked'].lower()=="true" else False
        if 'alpha' in self.validate_instance:
            alpha = float(self.validate_instance['alpha'])
        if 'fontsize' in self.validate_instance:
            fontsize = int(self.validate_instance['fontsize'])
        if 'add_error' in self.validate_instance:
            add_error = True if self.validate_instance['add_error'].lower()=="true" else False
        if 's' in self.validate_instance:
            s = int(self.validate_instance['s'])
        if 'bins' in self.validate_instance:
            bins = int(self.validate_instance['bins'])
        if 'binsize' in self.validate_instance:
            binsize = float(self.validate_instance['binsize'])
        if 'min_v' in self.validate_instance:
            min_v = int(self.validate_instance['min_v'])
        if 'max_v' in self.validate_instance:
            max_v = int(self.validate_instance['max_v'])
        if 'order' in self.validate_instance:
            order = self.validate_instance['order']
        if 'x' in self.validate_instance:
            x = self.validate_instance['x']
        if 'y' in self.validate_instance:
            y = self.validate_instance['y']
        if 'z' in self.validate_instance:
            z = self.validate_instance['z']
        if 'd1' in self.validate_instance:
            d1 = self.validate_instance['d1']
        if 'd2' in self.validate_instance:
            d2 = self.validate_instance['d2']
        if 'd3' in self.validate_instance:
            d3 = self.validate_instance['d3']
        if TITLE in self.validate_instance:
            title = self.validate_instance[TITLE]
        if 'x_label' in self.validate_instance:
            x_label = self.validate_instance['x_label']
        if 'y_label' in self.validate_instance:
            y_label = self.validate_instance['y_label']
        if 'legend' in self.validate_instance:
            legend = self.validate_instance['legend']
        if 'legend_names' in self.validate_instance:
            legend_names = self.validate_instance['legend_names']
        if 'annot' in self.validate_instance:
            if utilities.type_is_bool(self.validate_instance['annot']):
                annot = True if self.validate_instance['annot'].lower()=="true" else False
            else:
                utilities._info('error: "annot" should be bool.... ignored!!!')
        if 'transpose' in self.validate_instance:
            if utilities.type_is_bool(self.validate_instance['transpose']):
                transpose = True if self.validate_instance['transpose'].lower()=="true" else False
            else:
                utilities._info('error: "annot" should be bool.... ignored!!!')
        if 'ticks_x' in self.validate_instance:
            ticks_min_x, ticks_max_x = [float(x) for x in self.validate_instance['ticks_x']]
        if 'ticks_y' in self.validate_instance:
            ticks_min_y, ticks_max_y = [float(x) for x in self.validate_instance['ticks_y']]
        if 'figsize' in self.validate_instance:
            figsize = tuple([float(x) for x in self.validate_instance['figsize']]) # list to tuple
        if 'multiaxis' in self.validate_instance:
            multiaxis = True if self.validate_instance['multiaxis'].lower()=="true" else False
        if 'fontsize' in self.validate_instance:
            fontsize = float(self.validate_instance['fontsize'])
        if 'linewidth' in self.validate_instance:
            linewidth = float(self.validate_instance['linewidth'])
        if 'padsize' in self.validate_instance:
            padsize = float(self.validate_instance['padsize'])
        if 'kind' in self.validate_instance:
            fun = self.validate_instance['kind']
            if fun not in ['hist', 'scatter']:
                utilities._info('wrong kind. [kind] must be in [hist, scatter] for drilldown '
                      'visualization. terminating...')
                return None
            if fun == 'hist' and x is None:
                utilities._info('x is not defined in drilldown visualization (hist). terminating...')
                return None
            if fun == 'scatter' and (x is None or y is None):
                utilities._info('both x and y must be defined in drilldown visualization (scatter). '
                      'terminating...')
                return None
        if self.validate_instance[PLOT].lower() == 'hexbin':
            return HexbinPlot(columns=columns, target=target)
        elif self.validate_instance[PLOT].lower() == 'scattermatrix':
            return ScatterPlotMatrix(columns=columns, target=target, s=s)
        elif self.validate_instance[PLOT].lower() == 'boxplot':
            return BoxPlotting(columns=columns, target=target, x=x, y=y, z=z, order=order,
                               valipreprocessing=preprocessing)
        elif self.validate_instance[PLOT].lower() == 'bar':
            return BarPlotting(columns=columns, figsize=figsize, x_label=x_label,
                               y_label=y_label, trans=transpose, title=title,
                               valipreprocessing=preprocessing, fontsize=fontsize)
        elif self.validate_instance[PLOT].lower() == 'drilldown':
            if fun is None:
                utilities._info('there is no [kind] (like hist or scatter) defined in '
                      'drilldown visualization. terminating...')
                return None
            return Drilldownplot(x=x, y=y, z=z, fun=fun, d1=d1, d2=d2, d3=d3)
        elif self.validate_instance[PLOT].lower() == 'scatter':
            return ScatterPlot(columns=columns, target=target)
        elif self.validate_instance[PLOT].lower() == 'heatmap':
            return CustomHeatMaps(x=x, y=y, z=z, min_v=min_v, max_v=max_v, groupby=groupby,
                                  title=title, x_label=x_label,
                                  y_label=y_label, annot=annot,
                                  ticks_max_x=ticks_max_x, ticks_max_y=ticks_max_y,
                                  ticks_min_x=ticks_min_x, ticks_min_y=ticks_min_y,
                                  figsize=figsize, valipreprocessing=preprocessing, bins=bins)
        elif self.validate_instance[PLOT].lower() == 'line':
            return LinePlot(columns=columns, groupby=groupby, add_error=add_error,
                            legend=legend, x_label=x_label, y_label=y_label, title=title,
                            valipreprocessing=preprocessing, figsize=figsize, x_col=x_col,
                            ticks_max_x=ticks_max_x, ticks_max_y=ticks_max_y,
                            ticks_min_x=ticks_min_x, ticks_min_y=ticks_min_y, fontsize=fontsize,
                            stacked=stacked)
        elif self.validate_instance[PLOT].lower() == 'hist':
            return HistPlot(columns=columns, groupby=groupby,
                            legend=legend, legend_names=legend_names, x_label=x_label,
                            y_label=y_label, title=title, bins=bins, binsize=binsize,
                            valipreprocessing=preprocessing, figsize=figsize, fontsize=fontsize,
                            ticks_max_x=ticks_max_x, ticks_max_y=ticks_max_y,
                            ticks_min_x=ticks_min_x, ticks_min_y=ticks_min_y, alpha=alpha)

        return None

    def get_valid_plotter(self):
        try:
            plotter = self.initial_plot()
            if plotter is None:
                utilities._info("not a valid plotter!!! plotting ignored ...")
            return plotter
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            utilities._info("plotter initialization failed!!!\n {0}".format(exceptionValue))

class TopicModellingValidation(object):
    def __init__(self, object):
        self.validate_instance = object

    def initial(self):
        min_num_docs = 10
        allowed_postags = None
        groupby = None
        per_geo = False
        file_label = ''
        use_phrases = False
        use_tfidf = False
        num_topics = 5
        modelname = 'lda'
        perwordtopic = False
        allowext = False
        preprocessing = None
        visualization = None
        trainmodel = True

        if 'preprocessing' in self.validate_instance:
            preprocc = PreprocessingValidation(self.validate_instance)
            preprocessing = preprocc.get_valid_list_of_preprocessing()

        if VISUALIZATION in self.validate_instance:
            visualization_valid = PlotValidation(self.validate_instance[VISUALIZATION])
            visualization = task(visualization_valid.get_valid_plotter(), '', None)

        if 'textcolumn'not  in self.validate_instance:
            utilities._error(0, '"textcolumn" must be defined in the config.')
        textcolumn = self.validate_instance['textcolumn']

        if 'allowed_postags' in self.validate_instance:
            if not isinstance(self.validate_instance['allowed_postags'], list):
                utilities._error(0, '"allowed_postags" should be type of list... terminated..')
            allowed_postags = self.validate_instance['allowed_postags']

        if 'groupby' in self.validate_instance:
            if not isinstance(self.validate_instance['groupby'], list):
                utilities._error(0, '"groupby" should be type of list... terminated..')
            groupby = self.validate_instance['groupby']

        if 'min_num_docs' in self.validate_instance:
            min_num_docs = int(self.validate_instance['min_num_docs'])

        if 'num_topics' in self.validate_instance:
            num_topics = int(self.validate_instance['num_topics'])

        if 'use_tfidf' in self.validate_instance:
            use_tfidf = True if self.validate_instance['use_tfidf'].lower()=='true' else False

        if 'trainmodel' in self.validate_instance:
            trainmodel = True if self.validate_instance['trainmodel'].lower()=='true' else False

        if 'use_phrases' in self.validate_instance:
            use_phrases = True if self.validate_instance['use_phrases'].lower()=='true' else False

        if 'per_geo' in self.validate_instance:
            per_geo = True if self.validate_instance['per_geo'].lower()=='true' else False

        if 'per_word_topic' in self.validate_instance:
            perwordtopic = True if self.validate_instance['per_word_topic'].lower()=='true' else False

        if 'modelname' in self.validate_instance:
            modelname = self.validate_instance['modelname'].lower()

        corpus = Corpus(min_num_docs=min_num_docs, allowed_postags=allowed_postags,
                         use_phrases=use_phrases, use_tfidf=use_tfidf, allowext=allowext, file_label=file_label)

        if modelname == 'lda':
            return LDATopicModelling(num_topics=num_topics, groupby=groupby, per_geo=per_geo,
                                     perwordtopic=perwordtopic, corpus_object=corpus, textcolumn=textcolumn,
                                     preprocessing=preprocessing, file_label=file_label, visualization=visualization,
                                     trainmodel=trainmodel)
        elif modelname == 'lsi':
            return LSITopicModelling(num_topics=num_topics, groupby=groupby, per_geo=per_geo,
                                     perwordtopic=perwordtopic, corpus_object=corpus, textcolumn=textcolumn,
                                     preprocessing=preprocessing, file_label=file_label, visualization=visualization)
        else:
            raise NotImplementedError


    def get_valid_topicmodelling(self):
        try:
            topicmodel = self.initial()
            if topicmodel is None:
                utilities._info("not a valid topicmodel!!! ...")
            return topicmodel
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            utilities._info("topicmodelling initialization failed!!!\n {0}".format(exceptionValue))


class PreprocessingValidation(object):
    def __init__(self, object, preprocessing_parent_name=None):
        self.validate_instance = object
        self.preprocessinglist = []
        self.preprocessing_parent_name = preprocessing_parent_name + '_' if preprocessing_parent_name else ''

    def initial_preprocessing(self):
        for preprocess_name in self.validate_instance[PREPROCESSING]:
            single_preprocessing = self.validate_instance[PREPROCESSING][preprocess_name]
            if PROCESS not in single_preprocessing:
                utilities._info('"process" must be defined in {} preprocessing module. ignored!!!'.format(preprocess_name))
                continue
            if single_preprocessing[PROCESS].lower() == FEATUREENGINEERING:
                feature_engineer_validate = FeatureEngineeringValidation(single_preprocessing)
                valid_preprocessing = feature_engineer_validate.get_valid_feature_engineering()
                if valid_preprocessing is None:
                    utilities._info('not a valid feature engineering \n{}\n, this is ignored...'.format(single_preprocessing))
                    continue
                self.preprocessinglist.append(preprocessing(valid_preprocessing, 
                                    f'{self.preprocessing_parent_name}{preprocess_name}', 
                                    self.preprocessing_parent_name))
            elif single_preprocessing[PROCESS].lower() == FILTER:
                filter_validate = DatafilterValidation(single_preprocessing)
                valid_preprocessing = filter_validate.get_valid_filter()
                if valid_preprocessing is None:
                    utilities._info('not a valid filter \n{}\n, this filtering is ignored...'.format(
                        single_preprocessing))
                    continue
                self.preprocessinglist.append(preprocessing(valid_preprocessing, 
                                    f'{self.preprocessing_parent_name}{preprocess_name}', 
                                    self.preprocessing_parent_name))
            elif single_preprocessing[PROCESS].lower() == 'textfeatureengineering':
                preprocess_txt_validate = TextPreprocessingValidation(single_preprocessing)
                valid_preprocessing = preprocess_txt_validate.get_valid_textpreprocessing()
                if valid_preprocessing is None:
                    utilities._info('not a valid text preprocessing \n{}\n, this preprocessing is ignored...'.format(
                        single_preprocessing))
                    continue
                self.preprocessinglist.append(preprocessing(valid_preprocessing, 
                                    f'{self.preprocessing_parent_name}{preprocess_name}', 
                                    self.preprocessing_parent_name))
            else:
                raise NotImplementedError
        return None

    def get_valid_list_of_preprocessing(self):
        if not PREPROCESSING in self.validate_instance:
            return None
        if len(self.validate_instance[PREPROCESSING]) < 1:
            utilities._info('there is not any valid preprocessing defined on input data.')
            return None
        try:
            self.initial_preprocessing()
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            utilities._info("preprocessing initialization failed!!!\n {0}".format(exceptionValue))

        if len(self.preprocessinglist) < 1:
            utilities._info("not even a valid preprocessing!!! preprocessings are ignored ...")
            return None
        return self.preprocessinglist


class TasksValidation(object):
    def __init__(self, object, artifact_folder):
        self.validate_instance = object
        self.artifact_folder = artifact_folder
        self.tasks = []

    def initial_tasks(self):

        for task_name in self.validate_instance[TASKS]:
            task_cwd = None
            title = self.validate_instance[TASKS][task_name].get(TITLE, None)
            if CWD in self.validate_instance[TASKS][task_name]:
                cwd_valid = CwdValidation(self.validate_instance[TASKS][task_name], self.artifact_folder)
                task_cwd = cwd_valid.get_valid_cwd()

            if self.validate_instance[TASKS][task_name][TYPE].lower() == CLASSIFICATION:
                valid_classifier = ClassificationValidation(self.validate_instance[TASKS][task_name], task_name)
                valid_task = valid_classifier.get_valid_classifier()
                if valid_task is None:
                    utilities._info('not a valid {0} classifier, this classification is ignored...'.format(
                        self.validate_instance[TASKS][task_name][ALGORITHM]))
                    continue
                self.tasks.append(task(valid_task, task_name, task_cwd, title))
            elif self.validate_instance[TASKS][task_name][TYPE].lower() == CLUSTERING:
                valid_clusterer = ClusteringValidation(self.validate_instance[TASKS][task_name])
                valid_task = valid_clusterer.get_valid_clusterer()
                if valid_task is None:
                    utilities._info('not a valid {0} clusterer, this clustering is ignored...'.format(
                        self.validate_instance[TASKS][task_name][ALGORITHM]))
                    continue
                self.tasks.append(task(valid_task, task_name, task_cwd, title))
            elif self.validate_instance[TASKS][task_name][TYPE].lower() == REGRESSION:
                valid_regressor = RegressionValidation(self.validate_instance[TASKS][task_name], task_name)
                valid_task = valid_regressor.get_valid_regressor()
                if valid_task is None:
                    utilities._info('not a valid {0} regressor this regressor is ignored...'.format(
                        self.validate_instance[TASKS][task_name][ALGORITHM]))
                    continue
                self.tasks.append(task(valid_task, task_name, task_cwd, title))
            elif self.validate_instance[TASKS][task_name][TYPE].lower() == VISUALIZATION:
                valid_plotter = PlotValidation(self.validate_instance[TASKS][task_name])
                valid_task = valid_plotter.get_valid_plotter()
                if valid_task is None:
                    utilities._info('not a valid {0} plotter this visualization is ignored...'.format(
                        self.validate_instance[TASKS][task_name][PLOT]))
                    continue
                self.tasks.append(task(valid_task, task_name, task_cwd, title))
            elif self.validate_instance[TASKS][task_name][TYPE].lower() == CONFUSION:
                valid_confusion = ConfusionMatrixValidation(self.validate_instance[TASKS][task_name])
                valid_task = valid_confusion.get_valid_confusion()
                if valid_task is None:
                    utilities._info('not a valid {0} confusion matrix this task is ignored...'.format(
                        self.validate_instance[TASKS][task_name][CONFUSION]))
                    continue
                self.tasks.append(task(valid_task, task_name, task_cwd, title))
            elif self.validate_instance[TASKS][task_name][TYPE].lower() == ROCCURVE:
                valid_roc = RocCurveValidation(self.validate_instance[TASKS][task_name])
                valid_task = valid_roc.get_valid_roc_curve()
                if valid_task is None:
                    utilities._info('not a valid {0} confusion matrix this task is ignored...'.format(
                        self.validate_instance[TASKS][task_name][ROCCURVE]))
                    continue
                self.tasks.append(task(valid_task, task_name, task_cwd, title))
            elif self.validate_instance[TASKS][task_name][TYPE].lower() == WRITETODB:
                valid_db = DbValidation(self.validate_instance[TASKS][task_name])
                valid_task = valid_db.get_valid_db()
                if valid_task is None:
                    utilities._info('not a valid {0} db task this task is ignored...'.format(
                        self.validate_instance[TASKS][task_name]))
                    continue
                self.tasks.append(task(valid_task, task_name, task_cwd, title))
            elif self.validate_instance[TASKS][task_name][TYPE].lower() == TOPICMODELLING:
                valid_db = TopicModellingValidation(self.validate_instance[TASKS][task_name])
                valid_task = valid_db.get_valid_topicmodelling()
                if valid_task is None:
                    utilities._info('not a valid {0} topic modelling task this task is ignored...'.format(
                        self.validate_instance[TASKS][task_name]))
                    continue
                self.tasks.append(task(valid_task, task_name, task_cwd, title))
        return None

    def get_valid_list_of_tasks(self):
        if not TASKS in self.validate_instance:
            utilities._info('there is not any tasks on input data.')
            return None
        if len(self.validate_instance[TASKS]) < 1:
            utilities._info('there is not any valid tasks defined on input data.')
            return None
        try:
            self.initial_tasks()
        except:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            utilities._info("tasks initialization failed!!!\n {0}".format(exceptionValue))

        if len(self.tasks) < 1:
            utilities._info("not even a valid task!!! tasks are ignored ...")
            return None
        return self.tasks
