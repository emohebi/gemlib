from gemlib.abstarct.basefunctionality import BaseDataLoader
import os
import glob
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from gemlib.validation import utilities

class data_concat(BaseDataLoader):

    def load(self):
        utilities._info(f'{self.name}: concatenating {self.keys} dataset...')
        if len(set(self.keys) - set(self.resources.keys())) > 0:
            utilities._error(ValueError, f'set of data keys in {self.name} are not a subset of loaded resources! terminated.')
            raise ValueError
        self.df.reset_index(inplace=True, drop=True)
        self.apply_preprocessing()
        return [self.df]

class data_joiner(BaseDataLoader):

    def load(self):
        utilities._info(f'{self.name}: joining {self.keys} dataset...')
        if len(set(self.keys.keys()) - set(self.resources.keys())) > 0:
            utilities._error(ValueError, f'set of data keys in {self.name} are not a subset of loaded resources! terminated.')
            raise ValueError
        if ~isinstance(self.df, list):
            utilities._error(ValueError, f'the input in {self.name} should be a list of df! terminated.')
            raise ValueError

        cols = list(self.keys.values())
        df_left, df_right = self.df
        df_left.set_index(cols[0], inplace=True)
        df_right.set_index(cols[1], inplace=True)
        self.df = df_left.merge(df_right, 
                                how=self.how, 
                                left_index=True, 
                                right_index=True)
        self.df.reset_index(inplace=True, drop=True)
        self.apply_preprocessing()
        return [self.df] 