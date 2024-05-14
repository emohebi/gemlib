from abc import ABC
import abc
import multiprocessing
import itertools
import pandas as pd
import numpy as np
from gemlib.validation import utilities
import os, sys
from pathlib import Path
from tqdm import tqdm
# from gensim.models import CoherenceModel
import pickle


# TODO: Move the bits for reading or writing only into those subclasses.
class BaseData(ABC):

    def __init__(self, path:str,
                 hold_on:int=None, head:bool=None, exclude:int=None,
                 dtypes:dict=None, definition:dict=None, validpreprocessing=None,
                 port:int=None, concat:bool=False, joinon:list=None, suffix:str=None, 
                 how:str=None, fillna:dict=None, drop_na:list=None, cache:bool=False, 
                 name:str=None, num_rows:int=None, sheet_name:str=None):
        """Abstract class of data loader.

        Args:
            path (str): Path to a data directory or single file.
            hold_on (int, optional): Number of data points to be kept in data. ``head`` should be set. Defaults to None.
            head (bool, optional): If ``True`` then the data will be trimmed from head. Defaults to None.
            exclude (int, optional): Number of data to be excluded from data. 'head' should be set. Defaults to None.
            dtypes (dict, optional): Dictionary of data types in a data frame. i.e. {``<col_name>``:``int``}. Defaults to None.
            definition (dict, optional): Dictionary in the format of {``<new_col_name>``:``<value>``} to define new columns to data having constant value of <value>. Defaults to None.
            validpreprocessing (gemlib.preprocessing, optional): A gemlib preprocessing block attached to the data loader. Defaults to None.
            port (int, optional): Port number to connect to the data source. Defaults to None.
            concat (bool, optional): If set to ``True`` then all the files in the directory will be concatinated along ``axis=0``. Defaults to None.
            joinon (list, optional): A list of columns to join the datasets on [``<left_data_col>``, ``<right_data_col>``]. Defaults to None.
            suffix (str, optional): Data file fromat (``.csv`` or ``.xlsx``).
            how (str, optional): The way that join should be run (left, right ,...). Defaults to None.
            fillna (dict, optional): A dictionary of the format {``<col_name>``: ``<value>``} to fill ``nulls`` by ``<value>``. Defaults to None.
            drop_na (list, optional): A list of columns to drop nulls. Defaults to None.
            cache (bool, optional): If ``True`` the the data will be kept in the memory. Defaults to ``False``.
            name (str, optional): Name of the data loader. Defaults to None.
            num_rows (int, optional): Number of rows to load from data file. Defaults to None.
            sheet_name (str, optional): Name of spread sheet in an excel file. Defaults to None.
        """
        self.path = path # this is IP for dbs
        self.dirpath = None # The current working directory
        self.hold_on = hold_on
        self.head = head
        self.exclude = exclude
        self.df:pd.DataFrame = None
        self.dtypes = dtypes
        self.definition = definition
        self.validpreprocessing = validpreprocessing
        self.port = port
        self.concat = concat
        self.name = name
        self.joinon = joinon
        self.suffix = suffix
        self.how = how
        self.resources = None
        self.fillna = fillna
        self.drop_na = drop_na
        self.cache = cache
        self.num_rows = num_rows
        self.sheet_name = sheet_name

    def combine_with_duplicate(self, root:str, rel_path:str):
        """
        Finds the union of full path and relative path to a directory.

        Args:
            root (str): full path to a directory
            rel_path (str): relative path to the root path

        Returns:
            str: unified path

        Example:

            .. code-block::

                root = "C:/data/foo/dir_1"
                rel_path = "foo/dir_1/dir_2"

                output = "C:/data/foo/dir_1/dir_2"
        """
        rs = root.split("/")
        rps = rel_path.split("/")
        popped = False
        for v in rs:
            if v == rps[0]:
                rps.pop(0)
                popped = True
            elif popped:
                break

        return "/".join(rs + rps)

    def apply_preprocessing(self):
        """ 
        Applies the preprocessing in dataloaders
        """
        self.apply_drop_na()
        self.apply_definition()
        self.apply_dtypes()
        self.cut_off()
        self.apply_fillna()

    def apply_dtypes(self):
        """
        Applies dtypes to the dataframe in dataloders. 
        ``dtype`` keyword should be set in dataloader.
        """
        if self.dtypes is None:
            return
        self.df = self.df.astype(self.dtypes)
        print(self.df.dtypes)

    def apply_drop_na(self):
        """
        Drops ``null`` values in dataframe using the subset in ``drop_na``. 
        ``drop_na`` keyword should be set in dataloader.
        """
        if self.drop_na:
            self.df.dropna(subset=self.drop_na, inplace=True)

    def cut_off(self):
        """
        Trims the dataframe from head or tail. ``hold_on`` number of data 
        from head if ``head`` is ``True`` otherwise tail.
        """
        if self.exclude is not None and self.head is True:
            self.df = self.df.tail(len(self.df) - self.exclude)
            self.df.reset_index(inplace=True, drop=True)
            return
        if self.hold_on is None and self.head is None:
            return
        elif self.hold_on is not None and self.head is None:
            utilities._info('Error, "hold_on" should be defined along side the "head [True/False]", cut off terminated ...')
            return
        elif self.hold_on is None and self.head is not None:
            utilities._info('Error, "hold_on" should be defined along side the "head [True/False]", cut off terminated ...')
            return
        if self.head:
            self.df = self.df.head(self.hold_on)
            self.df.reset_index(inplace=True, drop=True)
        else:
            self.df = self.df.tail(self.hold_on)
            self.df.reset_index(inplace=True, drop=True)

    def apply_definition(self):
        """
        Defines a new column with constant value in the data frame. 
        ``definition`` keyword should be set to a dictionary of {``<column name>``: ``<value>``, ...} 
        """
        if self.definition is None:
            return
        for k in self.definition:
            if k not in self.df.columns.tolist():
                if utilities.type_is_int(self.definition[k]):
                    self.definition[k] = int(self.definition[k])
                self.df[k] = self.definition[k]
            else:
                self.df.rename({k : self.definition[k]}, axis='columns', inplace=True)

    def apply_fillna(self):
        """
        Fills ``null`` values in a data frame. ``fillna`` keyword in dataloader should be set to a dictionary.
        """
        if self.fillna is None:
            return
        for k in self.fillna:
            self.df[k].fillna(self.fillna[k], inplace=True)


class BaseDataLoader(BaseData):
    """ Read fileformats into a pandas dataframe.

    This abstract base class is implemented to read various file formats, and must return a
    pandas dataframe.
    """
    @abc.abstractmethod
    def load(self):
        """Load a source file and return the contents as a pandas dataframe
        """


class BaseDataWriter(BaseData):
    """ Write a pandas dataframe to various file formats.

    This abstract base class is implemented to write a pandas dataframe into various file formats.
    """
    @abc.abstractmethod
    def write(self):
        """Write the supplied data to the specified format.
        """


class BaseFeatureEngineering(ABC):
    def __init__(self, name:str=None, definition:dict=None, save:bool=False, dirpath:str=None,
                 groupby:list=None, condition:dict=None,
                 keepcolumnsvalues:list=None, axis:int=None, num_vocab:int=None,
                 embedding_dim:int=None, embedding_path:str=None, input:list=None,
                 columns:list=None, cache:bool=False, concat:bool=False,
                 x_cols:list=None, y_col:str=None, test_ratio:float=None, tokenizer=None, 
                 column:str=None, sequence_lenght:int=None, chunk_size:int=None,
                 models:dict=None, text_col:str=None, id_col:str=None, terms_col:str=None,
                 num_cores:int=None, df_aux:pd.DataFrame=None):
        """Abstract class of feature engineering

        Args:
            name (str, optional): Name of the preprocessing step. Defaults to None.
            definition (dict, optional): A dictionary defines new columns to the dataframe. Defaults to None.
            save (bool, optional): If True then the results will be staged. Defaults to False.
            dirpath (str, optional): Path to the preprocessing directory for staging purposes. Defaults to None.
            groupby (list, optional): A list of columns to apply the group by. Defaults to None.
            condition (dict, optional): A dictionary which defines a condition on data as preprocessing step. Defaults to None.
            keepcolumnsvalues (list, optional): List of column names to be kept after aggregation. Defaults to None.
            axis (int, optional): Choosing the axis. Defaults to None.
            num_vocab (int, optional): Number of vocabulary in tokenization process. Defaults to None.
            embedding_dim (int, optional): Defines embedding dimension in embedding matrix generation. Defaults to None.
            embedding_path (str, optional): Path to an existing embeddings. Defaults to None.
            input (list, optional): List of inputs from the pipeline/resources or objects. Defaults to None.
            columns (list, optional): List of columns to be used in the preprocessing. Defaults to None.
            cache (bool, optional): If True the the data stored in the memory. Defaults to False.
            concat (bool, optional): If True the the data stored in the memory. Defaults to False.
            x_cols (list, optional): List of columns to be used in preprocessing process. Defaults to None.
            y_col (str, optional): A singl column to be used in the preprocessing process. Defaults to None.
            test_ratio (float, optional): The ratio of data to be used as validation set. Defaults to None.
            tokenizer ([type], optional): A tokenizer from keras. Defaults to None.
            column (str, optional): A single column to be used in the preprocessing step. Defaults to None.
            sequence_lenght (int, optional): A single column to be used in the preprocessing step. Defaults to None.
            chunk_size (int, optional): To run the preprocessing step in chunks (when memory is limited). Defaults to None.
        """
        self.definition = definition
        self.name:str = name
        self.save = save
        self.dirpath = dirpath
        self.groupby = groupby
        self.condition = condition
        self.keepcolumnsvalues = keepcolumnsvalues
        self.axis = axis
        self.num_vocab = num_vocab
        self.embedding_dim = embedding_dim
        self.embedding_path:str = embedding_path
        self.input = input
        self.columns = columns
        self.cache = cache
        self.concat = concat
        self.x_cols = x_cols
        self.y_col = y_col
        self.test_ratio = test_ratio
        self.tokenizer = tokenizer
        self.column = column
        self.sequence_lenght = sequence_lenght
        self.resources = None
        self.chunk_size = chunk_size
        self.inner_chunk_size = None
        self.factor_mapping = None
        self.models = models
        self.text_col = text_col
        self.terms_col = terms_col
        self.id_col = id_col
        self.num_cores = num_cores
        self.df_aux = df_aux
        self.is_encode_terms = True
        self.is_run_text_extractor = True
        self.is_encode_expressions = True
        self.is_overlap_detection = True
        self.is_run_similarity_check = True
        self.threshold = 0.5
        self.device = 'cuda'
        self.sub_folder = 'preprocessing'

    @abc.abstractmethod
    def apply(self, df):
        """Apply feature engineering on a pandas dataframe."""


class BaseDataFilter(ABC):
    def __init__(self, list_col_names:list=None, query:str=None, 
                 dirpath:str=None, save:bool=False, input:list=None, cache:bool=False):
        """Abstract class of data filtering

        Args:
            list_col_names (list, optional): List of columns to keep in dataframe. Defaults to None.
            query (str, optional): The string query to filter dataframe. Defaults to None.
            dirpath (str, optional): Path to stage intermediate data. Defaults to None.
            save (bool, optional): If ``True`` then the intermediate results will be staged in ``dirpath``. Defaults to False.
            input (list, optional): List of inputs to this preprocessing step. Defaults to None.
            cache (bool, optional): If ``True`` then the intermediate results will be staged in ``dirpath``. Defaults to False.
        """
        self.name = ''
        self.column_names = list_col_names
        self.query = query
        self.dirpath = dirpath
        self.save = save
        self.input = input
        self.resources = None
        self.cache = cache

    @abc.abstractmethod
    def apply(self, df):
        """Apply filter on a pandas dataframe."""


class BasePlotter(ABC):
    def __init__(self, df=None, columns=None, target=None, title=None,
                 x=None, y=None, z=None, d1=None, d2=None, dirpath=None,
                 d3=None, fun=None, order=None, min_v=None, max_v=None,
                 cmap=None, valipreprocessing=None, s=None, legend=None,
                 x_label=None, y_label=None, aggfunc=None, annot=None,
                 ticks_max_x=None, ticks_max_y=None, figsize=None, trans=None,
                 ticks_min_x=None, ticks_min_y=None, add_error=None,
                 groupby=None, x_col=None, fontsize=None, linewidth=None,
                 padsize=None, multiaxis=False, legend_names=None, bins=None,
                 alpha=None, scatter_point=None, stacked=None, binsize=None,
                 input=None):

        self.df = df
        self.columns = columns
        self.target = target
        self.colors = ['plasma', 'Spectral', 'inferno', 'cool',
                       'Greens', 'Reds', 'autumn', 'Set1',
                       'Purples', 'Blues', 'jet', 'Paired', 'terrain']
        self.plt = None
        self.x = x
        self.y = y
        self.z = z
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.fun = fun
        self.dirpath = dirpath
        self.order = order
        self.min_v = min_v
        self.max_v = max_v
        self.title = title
        self.cmap = cmap
        self.validpreprocessing = valipreprocessing
        self.s = s
        self.legend = legend
        self.legend_names = legend_names
        self.x_label = x_label
        self.y_label = y_label
        self.aggfunc = aggfunc
        self.annot = annot
        self.ticks_max_y = ticks_max_y
        self.ticks_max_x = ticks_max_x
        self.ticks_min_x = ticks_min_x
        self.ticks_min_y = ticks_min_y
        self.figsize = figsize
        self.trans = trans
        self.add_error = add_error
        self.groupby = groupby
        self.bins = bins
        self.alpha = alpha
        self.x_col = x_col
        self.fontsize = fontsize
        self.linewidth = linewidth
        self.padsize = padsize
        self.multiaxis = multiaxis
        self.scatter_point = scatter_point
        self.stacked = stacked
        self.binsize= binsize
        self.resources = None
        self.input = input

        if padsize:
            self.labelpad = padsize - 1

    def save(self, path, dpi=None):
        if dpi is not None:
            self.plt.savefig(path, dpi=300, bbox_inches='tight')
        else:
            self.plt.savefig(path, dpi=300, bbox_inches='tight')
        utilities._info('file {0} saved.'.format(path))

    def remove_na(self):
        if self.target is None:
            self.df = self.df[self.columns].dropna()
        else:
            self.df = self.df[self.columns + [self.target]].dropna()

    @abc.abstractmethod
    def run(self):
        """plotter run"""


class BaseConfusionDefinition(ABC):
    def __init__(self, df=None, truth=None, predicted=None,
                 rowindex=None, columnindex=None, groupby=None,
                 validpreprocessing=None, validvisualization = None,
                 type=None, timecolumn=None, title=None, column_map=None, 
                 keepcolumnvalues=None, input=None):
        self.truth = truth
        self.predicted = predicted
        self.columnindex = columnindex
        self.rowindex = rowindex
        self.groupby = groupby
        self.dirpath = None
        self.df = df
        self.validpreprocessing = validpreprocessing
        self.validvisualization = validvisualization
        self.type = type
        self.timecolumn = timecolumn
        self.title = title
        self.column_map = column_map
        self.keepcolumnvalues = keepcolumnvalues
        self.resources = None
        self.input = input

    def rename_columns(self, df):
        if self.column_map:
            df.rename(self.column_map, axis='columns', inplace=True)
        return df

    @abc.abstractmethod
    def run(self):
        """generates confusion matrix from data"""

class BaseModelValidation(ABC):
    def __init__(self, 
                 df=None, 
                 target=None, 
                 features=None, 
                 model=None,
                 algo_name=None,
                 input=None):
        self.df = df
        self.target = target
        self.features = features
        self.model = model
        self.algo_name = algo_name
        self.resources = None
        self.input = input

    @abc.abstractmethod
    def run(self):
        """ run the spli function """

class BaseRegressor(ABC):
    def __init__(self, 
                 df=None, 
                 random_state=None,
                 features=None, 
                 target=None, 
                 algo_name=None, 
                 to_numerical=None,
                 validpreprocessing=None,
                 input=None):
        self.df = df
        self.features = features
        self.target = target
        self.algo_name = algo_name
        self.dirpath = None
        self.validpreprocessing = validpreprocessing
        self.random_state = random_state
        self.model = None
        self.model_validation:BaseModelValidation = None
        self.resources = None
        self.input = input

    @abc.abstractmethod
    def init_model(self):
        """ initialises the model """

    @abc.abstractmethod
    def run(self):
        self.init_model()
        self.model_validation.run()

class BaseClassifier(ABC):
    def __init__(self, df=None, dimensions=None, target=None, 
                 alpha=0.0001, num_iter=100, loss='hinge', kernel=None,
                 k=3, depth=4, algo_name=None, input=None, validpreprocessing=None,
                 x_train=None, x_test=None, y_train=None, y_test=None, 
                 embedding_matrix=None, sequence_lenght=None, batch_size=None,
                 epoch=None, num_classes=None, model_name=None, mode=None, 
                 model_path=None, y_col_map=None, top_n=1, lr=0.001, decay_n_epoch=5):
        self.k = k
        self.alpha = alpha
        self.num_iter = num_iter
        self.loss = loss
        self.dimensions = dimensions
        self.target = target
        self.df = df
        self.output = {}
        self.kernel = 'linear' if kernel is None else kernel
        self.depth = depth
        self.algo_name = algo_name
        self.dirpath = None
        self.resources = None
        self.input = input
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.embedding_matrix = embedding_matrix
        self.sequence_lenght = sequence_lenght
        self.epoch = epoch
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = None
        self.mode = mode
        self.validpreprocessing = validpreprocessing
        self.name = None
        self.embedding_layer_keras = None 
        self.model_path = model_path
        self.y_col_map = y_col_map
        self.top_n = top_n
        self.lr = lr
        self.decay_n_epoch = decay_n_epoch

    @abc.abstractmethod
    def run_training(self):
        """ train the initialiazed or loaded model """

    @abc.abstractmethod
    def run_testing(self):
        """ test the trained or loaded model """

    @abc.abstractmethod
    def run(self):
        """ runs the tasks on the model """

    @abc.abstractmethod
    def get_model(self):
        """ returns the trained or loaded model """


class BaseClusterer(ABC):
    def __init__(self, df=None, dimensions=None, target=None, filter=None, algo_name=None, input=None):
        self.df = df
        self.dimensions = dimensions
        self.target = target
        self.filter = filter
        self.n_clusters = None
        self.algo_name = algo_name
        self.dirpath = None
        self.resources = None
        self.input = input

    def validate_data(self, data):
        if len(data) > 2e4:
            utilities._info("this is a big data for clustering on a single machine. terminating this task.")
            return False
        if len(data) < 2:
            utilities._info("there is not enough data for clustering purposes. length of data is: {0} ..terminating this task.".
                  format(len(data)))
            return False
        return True

    def get_prepared_data(self):
        if self.target is None: # no need for binning
            self.df['UD_TARGET'] = pd.DataFrame(range(0, len(self.df)))
            self.target = 'UD_TARGET'
            self.dimensions.append(self.target)
            return np.array(self.df[self.dimensions].values)
        assert isinstance(self.target, str) # go for binning
        cluster_points = self.df[self.target].unique().tolist()
        data = []
        for point in cluster_points:
            dim_data = []
            for feature in self.dimensions:
                min_f = np.min(self.df[feature])
                max_f = np.max(self.df[feature])
                hist = np.histogram(self.df.loc[self.df[self.target] == point, feature].values, bins=4,
                                    range=(min_f, max_f))
                dim_data.extend(hist[0].tolist())
            dim_data.append(point)
            data.append(dim_data)
        return np.array(data)

    @abc.abstractmethod
    def run_algo(self):
        """ main algorithm"""

    def run(self):
        if self.filter is not None:
            self.df = self.df.query(self.filter)

        try:
            self.run_algo()
        except Exception:
            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
            utilities._info("clusterer failed!!!\n {0}".format(exceptionValue))


class BaseRocCurves(ABC):
    def __init__(self, df=None, truth=None, predicted=None, groupby=None,
                 validpreprocessing=None, timecolumn=None, title=None,
                 figsize=None, fontsize=None, scatter_point=None, input=None):
        self.truth = truth
        self.predicted = predicted
        self.groupby = groupby
        self.dirpath = None
        self.df = df
        self.validpreprocessing = validpreprocessing
        self.timecolumn = timecolumn
        self.title = title
        self.figsize = figsize
        self.fontsize = fontsize
        self.scatter_point = scatter_point
        self.resources = None
        self.input = input

    @abc.abstractmethod
    def run(self):
        """generates Roc Curve from data"""

    def remove_nans(self):
        self.df = self.df.dropna()


class BaseTopicModelling(ABC):
    def __init__(self, df=None, min_num_docs=100, allowed_postags=None, groupby=None,
                 per_geo=False, output_dir=None, file_label='', use_phrases=False,
                 use_tfidf=False, num_topics=10, modelname='lda', perwordtopic=False,
                 allowext=False, corpus=None, corpus_object=None, textcolumn=None, taskname = '',
                 preprocessing=None, visualization=None, num_key_words=5, model=None, dictionary=None,
                 mask=None, trainmodel=True):
        self.min_num_docs = min_num_docs
        self.allowed_postags= allowed_postags
        self.df = df
        self.groupby = groupby
        self.per_geo = per_geo
        self.output_dir = output_dir
        self.file_label = file_label
        self.use_phrases = use_phrases
        self.use_tfidf = use_tfidf
        self.num_topics = num_topics
        self.modelname = modelname
        self.perwordtopic = perwordtopic
        self.allowext = allowext
        self.corpus_object:BaseTopicModelling = corpus_object
        self.corpus = corpus
        self.model = model
        self.dictionary = dictionary
        self.textcolumn = textcolumn
        self.preprocessing:BaseTextPreprocessing = preprocessing
        self.taskname = taskname
        self.mask = mask
        self.validvisualization = visualization
        self.num_key_words = num_key_words
        self.trainmodel = trainmodel
        self.resources = None

    def get_geo_topics(self):
        if self.groupby:
            all_weeks_df = []
            for week, _df_week in tqdm(self.df.groupby(self.groupby)):
                df_topics = self.get_per_geo_topics(_df_week)
                df_topics['groupby'] = week
                all_weeks_df.append(df_topics)
            df_all_topics = pd.concat(all_weeks_df)
            df_all_topics.reset_index(inplace=True, drop=True)
        else:
            df_all_topics = self.get_per_geo_topics(self.df)
            df_all_topics.reset_index(inplace=True, drop=True)
        return df_all_topics

    def get_per_geo_topics(self, df):
        all_topics = []
        for key, _df in tqdm(df.groupby(['lat_index', 'long_index'])):
            region = list(key)
            count = _df['count'].unique()[0]
            lat_mean = _df['lat_mean'].unique()[0]
            long_mean = _df['long_mean'].unique()[0]
            topics_words = self.output_topics(_df, )
            region.append(topics_words)
            region.extend([count] + [lat_mean] + [long_mean])
            all_topics.append(region)
        df_topics = pd.DataFrame(all_topics,
                                 columns=['lat_index', 'long_index', 'topics',
                                          'tweet_count', 'lat_m', 'long_m'])
        return df_topics

    def get_all_topics(self):
        all_topics = []
        topics_words = self.output_topics(self.df)
        all_topics.append([topics_words])
        df_topics = pd.DataFrame(all_topics, columns=['topics'])
        return df_topics

    def output_topics(self, _df):
        results = {}
        _df.reset_index(inplace=True, drop=True)
        if self.corpus == None:
            self.get_corpus() # generate corpus
        coherence_values = []
        all_topics = []
        data = None

        file = self.output_dir / Path(f'filteredTweets_{self.file_label}.csv')
        if self.mask and self.mask.any():
            self.df = self.df[self.mask]
        data = self.df[self.textcolumn].values.tolist()
        self.df.to_csv(file, index=False)
        utilities._info(f'len of data in before topic generation: {len(data)}')
        if self.trainmodel:
            if self.model == None:
                self.get_model() # generate new model
                if self.model == None:
                    utilities._info('model is "None" ...')
                    return []
            else:
                utilities._info("updating the model online ...")
                self.model.update(self.corpus)
                file = open(str(self.output_dir / Path(f'model_{self.file_label}.pkl')), 'wb')
                pickle.dump(self.model, file)
                file.close()

        x = self.model.show_topics(num_topics=self.num_topics, num_words=self.num_key_words, formatted=False)
        topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]
        all_topics.append([topics_words])
        df_topics = pd.DataFrame(all_topics, columns=['topics'])
        file = self.output_dir / Path(f'Topics_{self.file_label}.csv')
        df_topics.to_csv(file)

        sent_topics_df = self.format_topics_sentences(self.model, self.corpus, data)
        file = self.output_dir / Path(f'dominantTopics_{self.file_label}.csv')
        sent_topics_df.to_csv(file)
        # coherence = self.compute_coherence_values(self.model, self.dictionary, self.corpus, data)
        # utilities._info(
        #     f'coherence values is: {coherence} for num topics: {self.num_topics} --- task: {self.taskname}')
        # coherence_values.append([self.num_topics, coherence])
        # df_c = pd.DataFrame(coherence_values, columns=['NumTopics', 'Coherence'])
        # df_c.to_csv(self.output_dir / Path(f'coehrence_{self.taskname}.csv'), index=False)
        # assert self.df.shape[0]==sent_topics_df.shape[0], "error: len of input df is not same as topics outputs"
        # results['single'] = sent_topics_df
        # return results
        return self.df, sent_topics_df

    def run(self):
        return self.output_topics(self.df)

    def compute_coherence_values(self, model, dictionary, corpus, texts):
        text_pure = [[dictionary[word_id] for word_id, freq in doc] for doc in corpus]
        # coherencemodel = CoherenceModel(model=model, corpus=corpus, dictionary=dictionary, texts=text_pure,
        #                                 coherence='c_v')
        coherence_values = None#coherencemodel.get_coherence()
        return coherence_values

    def format_topics_sentences(self, ldamodel, corpus, texts):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        sent_topics_df.reset_index(inplace=True)
        sent_topics_df.columns = ['Doc_No', 'Dominant_Topic', 'Topic_Perc_Contribution', 'Topic_Keywords', 'Text']
        return sent_topics_df

    def get_corpus(self, data=None):
        if data is None:
            data = self.df[self.textcolumn].values.tolist()
        self.corpus, self.dictionary, self.mask = self.corpus_object.get_corpus(data)
        if self.corpus is None:
            utilities._info('corpus is Not a valid corpus...')
            return None, None, None
        else:
            utilities._info('corpus generated...')

    @abc.abstractmethod
    def get_model(self):
        """get the model"""

class BaseTextPreprocessing(ABC):
    def __init__(self, bins=10):
        super().__init__()
        self.bins = bins

    @abc.abstractmethod
    def apply(self, df):
        """runs the modelling"""

