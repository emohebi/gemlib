from gemlib.abstarct.basefunctionality import BaseDataLoader
import os
import glob
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from gemlib.validation import utilities

class data_loader(BaseDataLoader):

    def read_file(self, filepath):
        if 'h5' in self.suffix:
            self.df = pd.read_hdf(filepath)
        elif 'csv' in self.suffix:
            self.df = pd.read_csv(filepath, nrows=self.num_rows)
        elif 'excel' in self.suffix:
            self.df = pd.read_excel(filepath, sheet_name=self.sheet_name)
        self.df.reset_index(inplace=True, drop=True)
        self.apply_preprocessing()
        return self.df

    def load(self):
        utilities._info(f'loading from {self.path}')
        list_of_data = []
        if os.path.isdir(self.path):  
            for filepath in tqdm([x for x in glob.glob(self.path + '/**/*' + self.suffix,
                                                       recursive=True)]):
                list_of_data.append(self.read_file(filepath))
        elif os.path.isfile(self.path):
                list_of_data.append(self.read_file(self.path))
        return list_of_data