from gemlib.abstarct.basefunctionality import BaseFeatureEngineering
import time
import pandas as pd
import numpy as np
from gemlib.validation import utilities
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle

class embedding(BaseFeatureEngineering):

    def get_word2vec(self):
        utilities._info(f'loading word vectors from {self.embedding_path} ...')
        if Path(self.embedding_path).suffix == ".txt":
            word2vec = {}
            with open(self.embedding_path, encoding="utf8") as f:
            # is just a space-separated text file in the format:
            # word vec[0] vec[1] vec[2] ...
                for line in f:
                    values = line.split()
                    word = values[0]
                    vec = np.asarray(values[1:], dtype='float32')
                    word2vec[word] = vec
        elif Path(self.embedding_path).suffix == ".pkl":
            word2vec = pickle.load(open(self.embedding_path, 'rb'))
        else :
            utilities._error(NotImplementedError, f'embeddings file should be glove (.txt) or a pickled dictionary as .pkl file')
        utilities._info('found %s word vectors.' % len(word2vec))
        return word2vec

    def apply(self, df=None):
        if not isinstance(self.tokenizer, Tokenizer) and self.tokenizer in self.resources:
            tokenizer = utilities.resolve_caching_load(self.resources, self.tokenizer)
        else:
            utilities._error(ValueError, f'tokenizer in {self.name} is not of correct type nor existed in resources.')
        word2idx = tokenizer.word_index
        word2vec = self.get_word2vec()
        count = 0
        utilities._info('filling pre-trained embeddings...')
        num_words = min(self.num_vocab, len(word2idx) + 1)
        embedding_matrix = np.zeros((num_words, self.embedding_dim))
        for word, i in word2idx.items():
            if i < self.num_vocab:
                embedding_vector = word2vec.get(word)
            if embedding_vector is not None:
            # words not found in embedding index will be all zeros.
                embedding_matrix[i] = embedding_vector
            else:
                count += 1
        utilities._info(f'num tokens not found in embedding: {count}')
        return {'0': [embedding_matrix]}

class factorization(BaseFeatureEngineering):

    def apply(self, df):
        dict_of_data = {}
        if not isinstance(df, list):
            df = [df]
        for i in range(len(df)):
            if isinstance(df[i], pd.DataFrame):
                df[i] = df[i][self.column].values.tolist()

        if self.factor_mapping:
            _mapping = utilities.resolve_caching_load(self.resources, self.factor_mapping)
            for i, data in enumerate(df):
                dict_of_data[f'{i}'] = list(map(lambda x: np.where(_mapping == x)[0][0], data))
            return dict_of_data
        factors, _mapping = pd.factorize(np.concatenate(df, axis=0))
        l_ind, up_ind = [0, 0]
        for i, data in enumerate(df):
            l_ind, up_ind = [l_ind + up_ind, up_ind + np.array(data).shape[0]]
            dict_of_data[f'{i}'] = [factors[l_ind: up_ind]]
        dict_of_data['factor_mapping'] = [_mapping]
        return dict_of_data

class text_encoding(BaseFeatureEngineering):

    def __init__(self, column, chunk_size, input=None, concat=None, device='cuda', models={'1':'roberta-large-nli-stsb-mean-tokens', 
                                                                                           '3':'bert-large-nli-stsb-mean-tokens'}):
        BaseFeatureEngineering.__init__(self, input=input, concat=concat, column=column, chunk_size=chunk_size)
        self.device = device
        self.models = models
        if self.device.lower() == 'cpu':
            self.models = {m:SentenceTransformer(self.models[m], device=self.device) for m in self.models}

    def runner(self, data, model):
        if isinstance(data, pd.DataFrame):
            data = data[self.column]
        all_embeddings = []
        n = self.chunk_size
        for i in range(0, data.shape[0], n):
            all_embeddings.append(model.encode(list(data[i:i+n]), show_progress_bar=True))
        return all_embeddings

    def get_model(self, name):
        if isinstance(self.models[name], str):
            return SentenceTransformer(self.models[name], device=self.device)
        return self.models[name]

    def apply(self, df):
        if not isinstance(df, list):
            df = [df]
        dict_of_data = {}

        for m in self.models:
            for i, data in enumerate(df):
                model_ = self.get_model(m)
                dict_of_data[f'{i}_model_{m}_encode'] = self.runner(data, model_)
            del model_
        return dict_of_data

class train_test_splitting(BaseFeatureEngineering):

    def apply(self, df):
        anzsco_list_tbd = df[self.y_col].value_counts().to_frame()[df[self.y_col].value_counts() == 1].reset_index()['index'].values.tolist()
        df = pd.concat([df, df[df[self.y_col].isin(anzsco_list_tbd)]])
        df.reset_index(drop=True, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(df[self.x_cols].values.reshape(-1,),
                                                          df[self.y_col].values.tolist(),
                                                          test_size=self.test_ratio,
                                                          stratify=df[self.y_col].values.tolist(), 
                                                          random_state=11)
        return {
            "x_train": [X_train], 
            "x_test": [X_test], 
            "y_train": [y_train], 
            "y_test": [y_test]
        }

class text_cleaning(BaseFeatureEngineering):

    def apply(self, df):
        """Applies simple text cleaning such as removing numbers and punctuations.

        Parameters
        ----------
        df : Dataframe
            Input dataframe.

        Returns
        -------
        list
            list of Dataframe (key is preprocessing name) with cleaned text and lowered case.
        """
        for col in self.columns:
            df[col] = df[col].str.replace(r'[^a-zA-z\s]', ' ').str.replace(r'\s+', ' ').str.lower()

        return {'0': [df]}

class text_tokenization(BaseFeatureEngineering):

    def runner(self, data, tokenizer):
        """AI is creating summary for runner

        Args:
            data ([type]): [description]
            tokenizer ([type]): [description]

        Returns:
            [type]: [description]
        """
        MAX_SEQUENCE_LENGTH = self.sequence_lenght
        if isinstance(data, pd.DataFrame):
            data = data[self.column]
        sequences = tokenizer.texts_to_sequences(data)
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        utilities._info(f'shape of data tensor: {data.shape}')
        return [data]

    def apply(self, df):
        MAX_VOCAB_SIZE = 4000000
        dict_of_data = {}
        if not isinstance(df, list):
            df = [df]
        
        if isinstance(self.tokenizer, Tokenizer):
            tokenizer = self.tokenizer
        elif isinstance(self.tokenizer, str) and self.resources and self.tokenizer in self.resources:
            tokenizer = utilities.resolve_caching_load(self.resources, self.tokenizer)
        else:
            text = np.concatenate(df, axis=0)
            tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
            tokenizer.fit_on_texts(text)
            word2idx = tokenizer.word_index
            utilities._info(f'found {len(word2idx)} unique tokens.')
        
        for i, data in enumerate(df):
            dict_of_data[f'{i}'] = self.runner(data, tokenizer)
        dict_of_data['tokenizer'] = [tokenizer]
        return dict_of_data

class one_hot_encoding(BaseFeatureEngineering):

    def apply(self, df):
        for col in self.columns:
            df = pd.concat([df, pd.get_dummies(df[col], prefix=self.name)], axis=1)
        return df


class aggregation(BaseFeatureEngineering):
    def __init__(self, definition:dict, groupby:list, name:str=None):
        """AI is creating summary for __init__

        Args:
            name (str, optional) : Name of the preprocessing step, by default None 
            definition (dict, optional) : A dictionary defines preprocessing in a data frame, by default None
            groupby (list, optional) : A list of columns to apply the group by, by default None
        
        Example:
            .. code-block:: json

                {
                    "type": "featureengineering",
                    "process": "aggregation",
                    "groupby": ["<list of column names>"]
                }
        """
        super(aggregation).__init__(self, definition=definition,
                                        groupby=groupby,
                                        name=name)
        
    def apply(self, df):
        for column in self.definition:
            for featurename in self.definition[column]:
                if self.definition[column][featurename].lower() == 'mean':
                    df_ = df.groupby(self.groupby)[column].mean().reset_index(name=f'{self.name}_{featurename}_m')
                elif self.definition[column][featurename].lower() == 'sum':
                    df_ = df.groupby(self.groupby)[column].sum().reset_index(name=f'{self.name}_{featurename}_s')
                elif self.definition[column][featurename].lower() == 'median':
                    df_ = df.groupby(self.groupby)[column].median().reset_index(name=f'{self.name}_{featurename}_me')
                else:
                    raise NotImplementedError(f'error: aggregation function: '
                    f'{self.definition[column][featurename]} has not been implemented.')
        res = df.set_index(self.groupby).join(df_.set_index(self.groupby), 
                                                           how='left')[f'{self.name}_{featurename}_m'].values.reshape(-1,1).tolist()
        return pd.concat([df, pd.DataFrame(res, columns=[f'{self.name}_{featurename}_m'])] , axis=1) 

class feature_engineer_regex(BaseFeatureEngineering):

    def apply(self, df):
        '''
        adds new features to the dataframe.
        :param df: DataFrame, a dataframe.
        :return: modified dataframe.
        '''
        '''
        definition : {
                    column_name : 
                                {
                                feature_name : regex,
                                feature_name : regex
                                }
                    column_name : 
                                {
                                feature_name : regex,
                                feature_name : regex
                                }
                    }
        '''
        if self.definition is None:
            return df
        print('len before feature engineering (regex) {}'.format(len(df)))
        for column in self.definition:
            if column not in df.columns.tolist():
                print('column {} does not exist in the dataframe!!!, ignored.'.format(column))
                continue
            for featurename in self.definition[column]:
                df[featurename] = df[column].str.extract(self.definition[column][featurename], expand=True)
        print('len after feature engineering (regex) {}'.format(len(df)))
        if self.save:
            df.to_csv(os.path.join(self.dirpath, 'modified_dataframe_' + str(time.time()) + '.csv'))
        return df


class feature_engineer_conditional(BaseFeatureEngineering):

    def parse_values(self, val):
        if utilities.type_is_int(val):
            val = int(val)
        elif utilities.type_is_float(val):
            val = float(val)
        elif utilities.type_is_bool(val):
            val = True if val.lower()=='true' else False
        return val

    def apply(self, df):
        '''
        adds new features to the dataframe.
        :param df: DataFrame, a dataframe.
        :return: modified dataframe.
        '''
        '''
        definition : {
                    feature_name : [condition, (optional)if True value, (optional)if False value],
                    feature_name : [condition, (optional)if True value, (optional)if False value]
                    }
        '''
        if self.definition is None:
            return df
        print('len before feature engineering (conditional) {}'.format(len(df)))
        for featurename in self.definition:
            condition, true_val, false_val = self.get_args(self.definition[featurename])
            if true_val:
                true_val = self.parse_values(true_val)
            if false_val:
                false_val = self.parse_values(false_val)
            if condition == None:
                print("feature engineering on {} is ignored!!! condition is missing...".format(featurename))
                continue
            elif true_val == None:
                df[featurename] = df.eval(condition)
            elif false_val == None:
                df[featurename] = df.eval(condition)
                df.loc[df[featurename], featurename] = true_val
            else:
                df[featurename] = df.eval(condition)
                df[featurename] = np.where(df[featurename], true_val, false_val)

        print('len after feature engineering (conditional) {}'.format(len(df)))
        if self.save:
            df.to_csv(os.path.join(self.dirpath, 'modified_dataframe_conditional_' + str(time.time()) + '.csv'))
        return df

    def get_args(self, list_of_args):
        if not isinstance(list_of_args, list):
            print("conditions for feature should be of type list!!!, ignored.")
        if len(list_of_args) == 3:
            return list_of_args
        elif len(list_of_args) == 2:
            return list_of_args + [None]
        elif len(list_of_args) == 1:
            return list_of_args + [None, None]
        else:
            print("conditional feature engineering has not been not defined correctly!!!, ignored. " +
                  "Help >> feature_name : [condition, (optional)if True value, (optional)if False value]")
            return None


class feature_aggregation(BaseFeatureEngineering):

    def apply(self, df):
        '''

        :param df:
        :return:
        '''
        '''
        groupby : [columns],
        definition : {
                column_name : 
                            {
                            feature_name : MEAN,
                            feature_name : MEDIAN
                            }
                column_name : 
                            {
                            feature_name : MEAN,
                            feature_name : MEDIAN
                            }
                }
        '''
        print('len before feature aggregation {}'.format(len(df)))
        if set(self.groupby) > set(df.columns.tolist()):
            print('"groupby" columns are not a subset of dataframe columns. ' +
                  'ignored!!! columns:{} vs {}'.format(df.columns.tolist(), self.groupby))
            return df
        df_list = []
        df_temp = df.copy(deep=True)
        for column in self.definition:
            for featurename in self.definition[column]:
                if self.definition[column][featurename] == 'MEDIAN':
                    df_list.append(df.groupby(self.groupby)[column].median().reset_index(name=featurename))
                elif self.definition[column][featurename] == 'MEAN':
                    df_list.append(df.groupby(self.groupby)[column].mean().reset_index(name=featurename))
                elif self.definition[column][featurename] == 'SUM':
                    df_list.append(df.groupby(self.groupby)[column].sum().reset_index(name=featurename))

        df = pd.concat(df_list, axis=1, join='inner')
        df = df.loc[:, ~df.columns.duplicated()]
        if self.keepcolumnsvalues is not None and not df_temp.empty:
            for x in self.keepcolumnsvalues:
                assert df_temp[x].unique().shape[0] == 1, "keepcolumnsvalues currently only supports file-level metadata"
                df[x] = df_temp[x].unique()[0]

        df.set_index(self.groupby, inplace=True)
        print('len after feature aggregation {}'.format(len(df)))
        if self.save:
            df.to_csv(os.path.join(self.dirpath, 'aggregated_dataframe_' + str(time.time()) + '.csv'))
        return df


class feature_value_mapping(BaseFeatureEngineering):

    def apply(self, df):
        '''
        :param df:
        :return:
        '''
        '''
        definition : {
                column_name : {
                        to_column_name :  
                                {
                                oldkey : newkey,
                                oldkey : newkey
                                }
                            },
                column_name : {
                        to_column_name :  
                                {
                                oldkey : newkey,
                                oldkey : newkey
                                }
                            }
                }
        '''
        print('len before mapping {}'.format(len(df)))
        for column in self.definition:
            for to_column_name in self.definition[column]:
                values = df[column].tolist()
                for old_key, new_key in self.definition[column][to_column_name].items():
                    if utilities.type_is_int(old_key) and utilities.type_is_int(new_key):
                        old_key = int(old_key)
                        new_key = int(new_key)
                    elif utilities.type_is_bool(old_key) and utilities.type_is_int(new_key):
                        if old_key in ['True', 'true']:
                            old_key = True
                        elif old_key in ['False', 'false']:
                            old_key = False
                        new_key = int(new_key)
                    elif utilities.type_is_int(old_key) and utilities.type_is_bool(new_key):
                        old_key = int(old_key)
                        if new_key in ['True', 'true']:
                            new_key = True
                        elif new_key in ['False', 'false']:
                            new_key = False
                    elif utilities.type_is_bool(old_key):
                        if old_key in ['True', 'true']:
                            old_key = True
                        elif old_key in ['False', 'false']:
                            old_key = False
                    elif utilities.type_is_int(new_key):
                        new_key = int(new_key)
                    elif utilities.type_is_int(old_key):
                        old_key = int(old_key)
                    values = [new_key if x == old_key else x for x in values]
                df[to_column_name] = values
        print('len after mapping {}'.format(len(df)))
        if self.save:
            df.to_csv(os.path.join(self.dirpath, df['ShortFilename'].unique()[0] + '_mapped_dataframe.csv'))
        return df


class feature_value_list_to_string(BaseFeatureEngineering):

    def apply(self, df):
        '''
        :param df:
        :return:
        '''
        '''
        definition : {
                column_name : to_column_name ,
                column_name : to_column_name 
                }
        '''
        print('len before list_to_string {}'.format(len(df)))
        for column in self.definition:
            to_column_name = self.definition[column]
            df[to_column_name] = [str(x[1]) for x in df[column].values]
        print('len after list_to_string {}'.format(len(df)))
        if self.save:
            df.to_csv(os.path.join(self.dirpath, 'list_to_string_dataframe_' + str(time.time()) + '.csv'))
        return df


class feature_concat(BaseFeatureEngineering):

    def apply(self, df):
        '''
        adds new features to the dataframe.
        :param df: DataFrame, a dataframe.
        :return: modified dataframe.
        '''
        '''
        definition : {
                    feature_name : [column1, column2, ...],
                    feature_name : [column1, column2, ...]
                    }
        '''
        if self.definition is None:
            return df
        # print('len before feature engineering (event duration) {}'.format(len(df)))
        for featurename in self.definition:
            df[featurename] = df[self.definition[featurename]].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        # print('len after feature engineering (event duration) {}'.format(len(df)))
        if self.save:
            df.to_csv(os.path.join(self.dirpath, 'modified_dataframe_concat_' + str(time.time()) + '.csv'))
        return df


class feature_se(BaseFeatureEngineering):

    def apply(self, df:pd.DataFrame) -> pd.DataFrame:
        '''
        adds new features to the dataframe.
        :param df: DataFrame, a dataframe.
        :return: modified dataframe.
        '''
        '''
        definition : {
                    column_name : [feature_name, feature_name]
                    column_name : [feature_name, feature_name]
                    }
        '''
        if self.definition is None:
            return df
        print('len before feature engineering (se) {}'.format(len(df)))
        for featurename in self.definition:
            column1, column2 = self.definition[featurename]
            if column1 not in df.columns or column2 not in df.columns:
                utilities._error(0, f'Missing defined column(s) {column1} or {column2} in data.')
            df[featurename] = np.square(df[column1] - df[column2])
        print('len after feature engineering (se) {}'.format(len(df)))
        if self.save:
            df.to_csv(os.path.join(self.dirpath, df['ShortFilename'].unique()[0] + '_rse.csv'))
        return df


class feature_abse(BaseFeatureEngineering):

    def apply(self, df:pd.DataFrame) -> pd.DataFrame:
        '''
        adds new features to the dataframe.
        :param df: DataFrame, a dataframe.
        :return: modified dataframe.
        '''
        '''
        definition : {
                    column_name : [feature_name, feature_name]
                    column_name : [feature_name, feature_name]
                    }
        '''
        if self.definition is None:
            return df
        print('len before feature engineering (abse) {}'.format(len(df)))
        for featurename in self.definition:
            column1, column2 = self.definition[featurename]
            if column1 not in df.columns or column2 not in df.columns:
                utilities._error(0, f'Missing defined column(s) {column1} or {column2} in data.')
            df[featurename] = np.abs(df[column1] - df[column2])
        print('len after feature engineering (abse) {}'.format(len(df)))
        if self.save:
            df.to_csv(os.path.join(self.dirpath, df['ShortFilename'].unique()[0] + '_abse.csv'))
        return df


class feature_subtract(BaseFeatureEngineering):

    def apply(self, df:pd.DataFrame) -> pd.DataFrame:
        '''
        adds new features to the dataframe.
        :param df: DataFrame, a dataframe.
        :return: modified dataframe.
        '''
        '''
        definition : {
                    column_name : [feature_name, feature_name, ...]
                    column_name : [feature_name, feature_name, ...]
                    }
        '''
        if self.definition is None:
            return df
        print('len before feature engineering (subtract) {}'.format(len(df)))
        for featurename in self.definition:
            column = self.definition[featurename][0]
            df[featurename] = df[column]
            self.definition[featurename].remove(column)
            for column in self.definition[featurename]:
                if column not in df.columns:
                    utilities._error(0, f'Missing defined column(s) {column} in data.')
                df[featurename] -= df[column]
        print('len after feature engineering (subtract) {}'.format(len(df)))
        if self.save:
            df.to_csv(os.path.join(self.dirpath, df['ShortFilename'].unique()[0] + '_subtract.csv'))
        return df


class feature_sum(BaseFeatureEngineering):

    def apply(self, df:pd.DataFrame) -> pd.DataFrame:
        '''
        adds new features to the dataframe.
        :param df: DataFrame, a dataframe.
        :return: modified dataframe.
        '''
        '''
        axis : 1
        definition : {
                    column_name : [feature_name, feature_name, ...]
                    column_name : [feature_name, feature_name, ...]
                    }
        '''
        if self.definition is None:
            return df
        print('len before feature engineering (sum) {}'.format(len(df)))
        for featurename in self.definition:
            if self.axis == 0:
                df = df[self.definition[featurename]].\
                    sum().to_frame().rename({0:featurename}, axis='columns')
            else:
                df[featurename] = df[self.definition[featurename]].sum(axis=1)
        print('len after feature engineering (sum) {}'.format(len(df)))
        if self.save:
            df.to_csv(os.path.join(self.dirpath, df['ShortFilename'].unique()[0] + '_sum.csv'))
        return df


class feature_mean(BaseFeatureEngineering):

    def apply(self, df:pd.DataFrame) -> pd.DataFrame:
        '''
        adds new features to the dataframe.
        :param df: DataFrame, a dataframe.
        :return: modified dataframe.
        '''
        '''
        axis : 1
        definition : {
                    column_name : [feature_name, feature_name, ...]
                    column_name : [feature_name, feature_name, ...]
                    }
        '''
        if self.definition is None:
            return df
        print('len before feature engineering (mean) {}'.format(len(df)))
        for featurename in self.definition:
            if self.axis == 0:
                df = df[self.definition[featurename]].\
                    mean().to_frame().rename({0:featurename}, axis='columns')
            else:
                df[featurename] = df[self.definition[featurename]].mean(axis=1)
        print('len after feature engineering (mean) {}'.format(len(df)))
        if self.save:
            df.to_csv(os.path.join(self.dirpath, df['ShortFilename'].unique()[0] + '_mean.csv'))
        return df


class feature_std(BaseFeatureEngineering):

    def apply(self, df:pd.DataFrame) -> pd.DataFrame:
        '''
        adds new features to the dataframe.
        :param df: DataFrame, a dataframe.
        :return: modified dataframe.
        '''
        '''
        axis : 1
        definition : {
                    column_name : [feature_name, feature_name, ...]
                    column_name : [feature_name, feature_name, ...]
                    }
        '''
        if self.definition is None:
            return df
        print('len before feature engineering (std) {}'.format(len(df)))
        for featurename in self.definition:
            if self.axis == 0:
                df = df[self.definition[featurename]]\
                    .std().to_frame().rename({0:featurename}, axis='columns')
            else:
                df[featurename] = df[self.definition[featurename]].std(axis=1)
        print('len after feature engineering (std) {}'.format(len(df)))
        if self.save:
            df.to_csv(os.path.join(self.dirpath, df['ShortFilename'].unique()[0] + '_std.csv'))
        return df


class feature_square(BaseFeatureEngineering):

    def apply(self, df:pd.DataFrame) -> pd.DataFrame:
        '''
        adds new features to the dataframe.
        :param df: DataFrame, a dataframe.
        :return: modified dataframe.
        '''
        '''
        definition : {
                    column_name : feature_name
                    column_name : feature_name
                    }
        '''
        if self.definition is None:
            return df
        print('len before feature engineering (square) {}'.format(len(df)))
        for featurename in self.definition:
            column= self.definition[featurename]
            if column not in df.columns:
                utilities._error(0, f'Missing defined column(s) {column} in data.')
            df[featurename] = np.square(df[column])
        print('len after feature engineering (square) {}'.format(len(df)))
        if self.save:
            df.to_csv(os.path.join(self.dirpath, df['ShortFilename'].unique()[0] + '_square.csv'))
        return df


class feature_sqrt(BaseFeatureEngineering):

    def apply(self, df:pd.DataFrame) -> pd.DataFrame:
        '''
        adds new features to the dataframe.
        :param df: DataFrame, a dataframe.
        :return: modified dataframe.
        '''
        '''
        definition : {
                    column_name : feature_name
                    column_name : feature_name
                    }
        '''
        if self.definition is None:
            return df
        print('len before feature engineering (sqrt) {}'.format(len(df)))
        for featurename in self.definition:
            column= self.definition[featurename]
            if column not in df.columns:
                utilities._error(0, f'Missing defined column(s) {column} in data.')
            df[featurename] = np.sqrt(df[column])
        print('len after feature engineering (sqrt) {}'.format(len(df)))
        if self.save:
            df.to_csv(os.path.join(self.dirpath, df['ShortFilename'].unique()[0] + '_sqrt.csv'))
        return df


class feature_absolute(BaseFeatureEngineering):

    def apply(self, df:pd.DataFrame) -> pd.DataFrame:
        '''
        adds new features to the dataframe.
        :param df: DataFrame, a dataframe.
        :return: modified dataframe.
        '''
        '''
        definition : {
                    column_name : feature_name
                    column_name : feature_name
                    }
        '''
        if self.definition is None:
            return df
        print('len before feature engineering (absolute) {}'.format(len(df)))
        for featurename in self.definition:
            column= self.definition[featurename]
            if column not in df.columns:
                utilities._error(0, f'Missing defined column(s) {column} in data.')
            df[featurename] = np.abs(df[column])
        print('len after feature engineering (absolute) {}'.format(len(df)))
        if self.save:
            df.to_csv(os.path.join(self.dirpath, df['ShortFilename'].unique()[0] + '_absolute.csv'))
        return df

