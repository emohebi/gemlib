from collections import OrderedDict
from datetime import datetime
import sys
import re
import time
import threading
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import os
import json
import logging
import spacy
from spacy.matcher import Matcher

EXIT_SUCCESS, EXIT_FAILURE = range(2)

def resolve_caching_load(dict_of_data, input, concat=False):
    data = []
    if isinstance(input, list):
        for inp in input:
            data.append(resolve_caching_load(dict_of_data, inp, True))
        return data

    if isinstance(input, str) and input not in dict_of_data:
        if Path(input).is_file() and Path(input).suffix == ".pkl":
            return pickle.load(open(input, 'rb'))
        if Path(input).is_file() and Path(input).suffix == ".csv":
            return pd.read_csv(input)
    elif not isinstance(input, str):
        _info(f'returning input as it is ...')
        return input 
    
    for d in dict_of_data[input]:
        if isinstance(d, str):
            if os.path.isfile(d):
                _info(f'loading file: {d}')
                if Path(d).suffix == '.csv':
                    data.append(pd.read_csv(d))
                if Path(d).suffix == '.pkl':
                    data.append(pickle.load(open(d, 'rb')))
            else:
                _info(f'string in resources is not a path to a file: {d}')
        else:
            data.append(d)
    if len(data) > 1:
        if concat:
            if isinstance(data[0], pd.DataFrame):
                data = pd.concat(data)
            else:
                data = np.concatenate(data, axis=0)
    else:
        data = data[0]
    return data
            
def resolve_caching_stage(cache, data, dirpath, subfolder, name, ind=0):
    (Path(dirpath) / Path(subfolder)).mkdir(parents=True, exist_ok=True)
    dict_of_data = {}
    if cache:
        dict_of_data[name] = data if isinstance(data, list) else [data]
    elif isinstance(data, list):
        dict_of_data[name] = []
        for ind, d in enumerate(data):
            if isinstance(d, pd.DataFrame):
                df_path = Path(dirpath) / subfolder / f'{name}_{ind}.csv'
                d.to_csv(df_path, index=False)
            else:
                df_path = Path(dirpath) / subfolder / f'{name}_{ind}.pkl'
                pickle.dump(d, open(df_path, 'wb'))
            dict_of_data[name].append(str(df_path))
    elif isinstance(data, dict):
        for key in data:
            dict_of_data[f'{name}_{key}'] = []
            for ind, d in enumerate(data[key]):
                if isinstance(d, pd.DataFrame):
                    df_path = Path(dirpath) / subfolder / f'{name}_{key}_{ind}.csv'
                    d.to_csv(df_path, index=False)
                else:
                    df_path = Path(dirpath) / subfolder / f'{name}_{key}_{ind}.pkl'
                    pickle.dump(d, open(df_path, 'wb'))
                dict_of_data[f'{name}_{key}'].append(str(df_path))
    else:
        dict_of_data[name] = []
        if isinstance(data, pd.DataFrame):
            df_path = Path(dirpath) / subfolder / f'{name}_{ind}.csv'
            data.to_csv(df_path, index=False)
        else:
            df_path = Path(dirpath) / subfolder / f'{name}_{ind}.pkl'
            pickle.dump(data, open(df_path, 'wb'))
        dict_of_data[name].append(str(df_path))
    return dict_of_data

def stage_resources_dict(resources, dirpath, label=None):
    path_ = Path(dirpath) / f'resources.json' if label is None else Path(dirpath) / f'resources_{label}.json'

    with open(path_, 'w') as f:
        f.write(json.dumps(resources, indent=4)) # use `json.loads` to do the reverse
    _info(f'resources saved: {path_}')

def load_resources_dict(dirpath, label=None):
    resources = {}
    path_ = Path(dirpath) / f'resources.json' if label is None else Path(dirpath) / f'resources_{label}.json'
    if path_.is_file():
        with open(path_, 'r') as f:
            resources = json.load(f)
        _info(f'resources loaded: {path_}')
    return resources

def _decode_list(data):
    rv = []
    for item in data:
        if isinstance(item, bytes):
            item = item.encode('utf-8')
        elif isinstance(item, list):
            item = _decode_list(item)
        elif isinstance(item, dict):
            item = _decode_dict(item)
        rv.append(item)
    return rv


def _decode_dict(pairs):
    new_pairs = []
    for key, value in pairs:
        if isinstance(key, bytes):
            key = key.encode('utf-8')
        if isinstance(value, bytes):
            value = value.encode('utf-8')
        elif isinstance(value, list):
            value = _decode_list(value)
        new_pairs.append((key, value))
    return OrderedDict(new_pairs)

def type_is_float(str):
    try:
        float(str)
        return True
    except ValueError:
        return False

def type_is_int(str):
    try:
        int(str)
        return True
    except ValueError:
        return False

def type_is_bool(str):
    try:
        return str.lower() in ['false', 'true']
    except ValueError:
        return False

def _info(mess):
    # print(f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}: {str}")
    logging.info(mess)

def _error(exit_code, message, *args): # pragma: no cover
    """Logs an error message and exits the script with an exit code"""
    raise Exception(message, *args)

def resolve_string_tuple(key):
    if isinstance(key, str):
        return key
    else:
        return '_'.join(list(key))

def refine_lists(columns:list, df_columns:list):
    temp_list = []
    for col in columns:
        if col in df_columns:
            temp_list.append(col)
        elif '*' in col:
            pattern = f'({col})'.replace('*', r'_\w*')
            temp_list.extend(re.findall(pattern, ' '.join(df_columns)))
            temp_list = sorted(temp_list)
    return temp_list

def check_if_exists_in_resources(resources, key):
    for k in resources:
        if k.find(key, 0, len(key)) + 1:
            return True
    return False

def get_pos_matcher(nlp):
    
    list_of_rules = [
        # ['ADJ', 'NOUN', 'PUNCT', 'NOUN', 'PUNCT', 'NOUN', 'PUNCT', 'NOUN', 'CCONJ', 'NOUN', 'ADP','NOUN'],
        # ['ADJ', 'PART', 'DET', 'VERB', 'CCONJ', 'ADJ', 'ADJ', 'VERB', 'ADJ', 'NOUN'],
        # ['ADJ', 'NOUN', 'CCONJ', 'NOUN', 'ADP', 'ADJ', 'NOUN', 'NOUN'],

        # ['VERB', 'ADJ', 'NOUN', 'NOUN', 'PART', 'VERB', 'CCONJ', 'ADJ', 'NOUN', 'NOUN'],
        # ['VERB', 'NOUN', 'PART', 'VERB', 'CCONJ', 'VERB', 'NOUN', 'VERB', 'NOUN'],
        # ['VERB', 'NOUN', 'VERB', 'CCONJ', 'VERB', 'NOUN', 'CCONJ', 'NOUN', 'NOUN'],
        # ['VERB', 'NOUN', 'ADP', 'ADJ', 'NOUN', 'VERB', 'NOUN', 'ADP', 'NOUN', 'NOUN'],
        # ['VERB', 'NOUN', 'ADP', 'ADV', 'PART', 'VERB', 'ADJ', 'NOUN', 'NOUN', 'PART', 'VERB'],
        # ['VERB', 'NOUN', 'CCONJ', 'NOUN', 'ADP', 'ADJ', 'NOUN', 'NOUN', 'ADV', 'CCONJ', 'ADV'],
        # ['VERB', 'NOUN', 'CCONJ', 'NOUN', 'NOUN', 'ADP', 'DET', 'ADJ', 'NOUN'],
        # ['VERB', 'NOUN', 'PART', 'VERB', 'ADJ', 'CCONJ', 'ADJ', 'NOUN'],
        # ['VERB', 'DET', 'ADJ', 'CCONJ', 'ADJ', 'NOUN', 'VERB', 'ADP', 'NOUN', 'NOUN'],
        # ['VERB', 'DET', 'NOUN', 'ADP', 'NOUN', 'NOUN', 'NOUN', 'ADP', 'ADJ', 'NOUN'],
        # ['VERB', 'DET', 'NOUN', 'ADP', 'NOUN', 'NOUN', 'VERB', 'NOUN', 'NOUN'],
        # ['VERB', 'DET', 'ADJ', 'NOUN', 'NOUN', 'VERB', 'ADJ', 'NOUN', 'NOUN', 'NOUN'],
        # ['VERB', 'CCONJ', 'VERB', 'DET', 'ADJ', 'NOUN', 'PUNCT', 'NOUN', 'NOUN'],
        # ['VERB', 'CCONJ', 'VERB', 'ADJ', 'NOUN', 'ADP', 'ADJ', 'NOUN'],
        # ['VERB', 'CCONJ', 'VERB', 'DET', 'ADJ', 'ADJ', 'NOUN', 'NOUN'],
        # ['VERB', 'CCONJ', 'VERB', 'PROPN', 'PROPN', 'NOUN'],
        # ['VERB', 'DET', 'NOUN', 'ADP', 'ADJ', 'NOUN', 'NOUN', 'NOUN'],
        # ['VERB', 'DET', 'NOUN', 'ADP', 'DET', 'NOUN', 'AUX', 'VERB'],
        # ['VERB', 'DET', 'NOUN', 'ADP', 'DET', 'ADJ', 'NOUN', 'NOUN'],
        # ['VERB', 'DET', 'NOUN', 'ADP', 'ADJ', 'NOUN', 'NOUN'],
        # ['VERB', 'DET', 'ADJ', 'NOUN', 'NOUN', 'VERB', 'PROPN'],
        # ['VERB', 'DET', 'NOUN', 'ADP', 'CCONJ', 'ADP', 'NOUN'],
        # ['VERB', 'DET', 'VERB', 'NOUN', 'DET', 'VERB', 'NOUN'],
        # ['VERB', 'NOUN', 'ADP', 'NOUN', 'ADP', 'NOUN', 'NOUN'],
        # ['VERB', 'NOUN', 'NOUN', 'CCONJ', 'NOUN', 'NOUN'],
        # ['VERB', 'ADJ', 'CCONJ', 'ADJ', 'NOUN', 'ADP', 'NOUN', 'NOUN'],
        # ['VERB', 'ADJ', 'NOUN', 'NOUN', 'NOUN', 'NOUN'],
        # ['VERB', 'NOUN', 'ADP', 'VERB', 'NOUN'],
        # ['VERB', 'NOUN', 'ADP', 'NOUN', 'NOUN'],
        # ['VERB', 'NOUN', 'PART', 'VERB', 'NOUN'],
        # ['VERB', 'CCONJ', 'VERB', 'NOUN', 'NOUN'],
        # ['VERB', 'CCONJ', 'VERB', 'ADJ', 'NOUN'],
        # ['VERB', 'DET', 'ADJ', 'NOUN', 'NOUN'],
        # ['VERB', 'ADJ', 'ADJ', 'NOUN', 'NOUN'],
        # ['VERB', 'ADJ', 'CCONJ', 'ADJ', 'NOUN'],
        # ['VERB', 'NOUN', 'CCONJ', 'NOUN'],
        # ['VERB', 'NOUN', 'NOUN', 'NOUN'],
        # ['VERB', 'NOUN', 'ADP', 'NOUN'],
        # ['VERB', 'PROPN', 'PROPN', 'PROPN'],
        # ['VERB', 'ADJ', 'NOUN', 'NOUN'],
        # ['VERB', 'ADJ', 'ADJ', 'NOUN'],
        # ['VERB', 'ADP', 'NOUN', 'NOUN'],      
        # ['VERB', 'DET', 'VERB', 'NOUN'],
        # ['VERB', 'CCONJ', 'VERB'],
        # ['VERB', 'PROPN', 'PROPN'],
        # ['VERB', 'NOUN', 'NOUN'],
        # ['VERB', 'ADJ', 'NOUN'],
        # ['VERB', 'NOUN', 'ADV'],
        
        # ['NOUN', 'PART', 'VERB', 'CCONJ', 'VERB', 'DET', 'NOUN', 'ADP', 'ADJ', 'NOUN', 'NOUN', 'ADP', 'PROPN', 'CCONJ','PROPN'],
        # ['NOUN', 'ADP', 'DET', 'NOUN', 'NOUN', 'SCONJ', 'DET', 'NOUN', 'ADP', 'NOUN', 'NOUN'],
        # ['NOUN', 'ADP', 'ADJ', 'NOUN', 'NOUN', 'PART', 'VERB', 'ADJ', 'ADJ', 'NOUN'],
        # ['NOUN', 'ADP', 'ADJ', 'NOUN', 'ADP', 'NOUN', 'CCONJ', 'NOUN', 'NOUN'],
        # ['NOUN', 'ADP', 'ADJ', 'CCONJ', 'ADJ', 'NOUN', 'ADP', 'ADJ', 'NOUN'],
        # ['NOUN', 'ADP', 'ADJ', 'NOUN', 'CCONJ', 'NOUN', 'NOUN'],
        # ['NOUN', 'ADP', 'VERB', 'PUNCT', 'VERB', 'PUNCT', 'CCONJ', 'NOUN', 'NOUN'],
        # ['NOUN', 'ADP', 'NOUN', 'ADP', 'DET', 'NOUN', 'NOUN', 'CCONJ', 'DET', 'NOUN'],
        # ['NOUN', 'NOUN', 'NOUN', 'VERB', 'DET', 'ADJ', 'ADJ', 'NOUN', 'NOUN'],
        # ['NOUN', 'NOUN', 'ADP', 'NOUN', 'ADP', 'ADJ', 'NOUN'],
        # ['NOUN', 'NOUN', 'CCONJ', 'NOUN', 'ADP', 'NOUN'],
        # ['NOUN', 'NOUN', 'NOUN', 'CCONJ', 'NOUN', 'NOUN'],
        # ['NOUN', 'NOUN', 'ADP', 'DET', 'NOUN', 'NOUN'],
        # ['NOUN', 'ADP', 'VERB', 'NOUN', 'VERB', 'NOUN'],
        # ['NOUN', 'ADP', 'DET', 'ADJ', 'NOUN', 'CCONJ', 'ADJ', 'NOUN', 'NOUN'],
        # ['NOUN', 'ADP', 'DET', 'ADJ', 'NOUN', 'NOUN'],
        # ['NOUN', 'PROPN', 'NOUN', 'NOUN', 'NOUN'],
        # ['NOUN', 'ADP', 'NOUN', 'ADJ', 'NOUN'],
        # ['NOUN', 'ADP', 'NOUN', 'NOUN', 'NOUN'],
        # ['NOUN', 'ADP', 'NOUN', 'CCONJ', 'ADJ'],
        # ['NOUN', 'ADP', 'ADJ', 'NOUN', 'NOUN'],
        # ['NOUN', 'ADP', 'DET', 'VERB', 'NOUN'],
        # ['NOUN', 'NOUN', 'ADP', 'ADJ', 'NOUN'],
        # ['NOUN', 'NOUN', 'NOUN', 'CCONJ', 'NOUN'],
        # ['NOUN', 'NOUN', 'NOUN', 'NOUN'],
        # ['NOUN', 'NOUN', 'NOUN', 'VERB'],
        # ['NOUN', 'NOUN', 'CCONJ', 'NOUN'],
        # ['NOUN', 'NOUN', 'ADP', 'NOUN'],
        # ['NOUN', 'PART', 'VERB', 'NOUN'],
        # ['NOUN', 'ADP', 'DET', 'NOUN'],
        # ['NOUN', 'ADP', 'NOUN', 'NOUN'], #
        # ['NOUN', 'ADP', 'NOUN', 'VERB'],
        # ['NOUN', 'ADP', 'VERB', 'NOUN'],
        # ['NOUN', 'CCONJ', 'NOUN', 'NOUN'],
        # ['NOUN', 'NOUN', 'VERB'],   
        # ['NOUN', 'NOUN', 'PROPN'],
        # ['NOUN', 'NOUN', 'NOUN'],
        # ['NOUN', 'PROPN', 'PROPN'],
        # ['NOUN', 'PROPN', 'NOUN'],
        # ['NOUN', 'VERB', 'ADV'],
        # ['NOUN', 'ADP', 'NOUN'], #

        # ['PROPN', 'PROPN', 'PROPN', 'VERB'],
        # ['PROPN', 'PROPN', 'PROPN', 'PROPN'],
        # ['PROPN', 'PROPN', 'ADP', 'ADJ'],
        # ['PROPN', 'PROPN', 'PROPN'],
        # ['PROPN', 'PROPN', 'NOUN'],
        # ['PROPN', 'PROPN', 'VERB'],
        # ['PROPN', 'NOUN', 'PROPN'],
        # ['PROPN', 'NOUN', 'NOUN'],
        
        # ['ADJ', 'ADJ', 'NOUN', 'CCONJ', 'NOUN', 'NOUN'],
        # ['ADJ', 'NOUN', 'NOUN', 'NOUN'],
        # ['ADJ', 'NOUN', 'ADP', 'NOUN'],
        # ['ADJ', 'PROPN', 'PROPN', 'PROPN'],
        # ['ADJ', 'NOUN', 'NOUN'],
        # ['ADJ', 'PROPN', 'PROPN'],
        # ['ADV', 'VERB', 'NOUN'],

        ['PROPN', 'VERB'],
        ['VERB', 'NOUN'],
        ['VERB', 'PROPN'],
        ['VERB', 'ADJ'],
        ['NOUN', 'VERB'],
        ['NOUN', 'NOUN'],
        ['ADJ', 'NOUN'],
        ['ADJ', 'PROPN'],
        ['VERB', 'ADV'],
        ['PROPN', 'PROPN'],
        ['NOUN', 'PROPN'],
        ['PROPN', 'NOUN'],
        ['ADV', 'NOUN'],
        ['DET', 'NOUN'],
        ['ADV', 'VERB'],
        ['ADJ', 'ADJ']
        
    #     ['NOUN'],
    # #     ['PROPN'],
    #     ['VERB']
    ]
    
    rules = [[{"POS": i} for i in j] for j in list_of_rules]

    matcher = Matcher(nlp.vocab)
    matcher.add("rules", None, *rules)
    return matcher


class Spinner:
    busy = False
    delay = 0.1

    @staticmethod
    def spinning_cursor():
        while 1: 
            for cursor in '|/-\\': yield cursor

    def __init__(self, delay=None):
        self.spinner_generator = self.spinning_cursor()
        if delay and float(delay): self.delay = delay

    def spinner_task(self):
        while self.busy:
            sys.stdout.write(f'{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}: {next(self.spinner_generator)}')
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write('\x1b[2K\r')
            sys.stdout.flush()

    def __enter__(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()

    def __exit__(self, exception, value, tb):
        self.busy = False
        time.sleep(self.delay)
        if exception is not None:
            return False


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

class SetLogger(object):

    def __init__(self, dirpath, set_output_file=True):
        self.dirpath = dirpath
        self.set_output_file = set_output_file
    
    def set(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)

        if self.set_output_file:
            fhandler = logging.FileHandler(filename=f'{self.dirpath}/out_{datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}.log', mode='a')
            fhandler.setFormatter(formatter)
            logger.addHandler(fhandler)
        