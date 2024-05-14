from gemlib.abstarct.basefunctionality import BaseDataFilter
from gemlib.validation import utilities as utils
import time
import re

class DataColumnFilter(BaseDataFilter):

    def apply(self, df):
        self.column_names = utils.refine_lists(self.column_names, df.columns.tolist())
        if set(self.column_names) < set(df.columns.tolist()):
            df = df[self.column_names]
            df.reset_index(drop=True, inplace=True)
            if self.save:
                df.to_csv(self.dirpath + r'\filtered_data_frame_' + str(time.time()) + '.csv')
            return df
        else:
            missing = set(self.column_names) - set(df.columns.tolist())
            raise KeyError(f'error: there are some columns missing in data: {list(missing)}')

    



class QueryFilter(BaseDataFilter):
    def apply(self, df):
        if self.query is None:
            print('query is not defined. filtering is ignored...')
            return
        if self.column_names is not None:
            if set(self.column_names) < set(df.columns.tolist()):
                df = df[self.column_names].copy()
            else:
                avaliable = set(df.columns.tolist()).intersection(set(self.column_names))
                if len(avaliable) > 0:
                    df = df[list(avaliable)]
                print('columns in the filter definition are not subset of columns in the input data.')
        df = df.query(self.query).copy()
        df.reset_index(drop=True, inplace=True)
        if self.save:
            df.to_csv(self.dirpath + self.name + '_filtered_data_frame_' + str(time.time()) + '.csv')
        return df
