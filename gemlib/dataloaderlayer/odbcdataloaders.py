from gemlib.abstarct.basefunctionality import BaseDataLoader
import pandas as pd
import pyodbc
from gemlib.validation import utilities

class sql_data_loader(BaseDataLoader):

    def __init__(self, connection_string, **kwargs):
        super(sql_data_loader, self).__init__(**kwargs)
        self.conn_string = connection_string
        self.connection = pyodbc.connect(connection_string)

    def load(self):
        utilities._info(f'{self.name}: connection string: {self.conn_string}')
        self.df = pd.read_sql_query(str(self.query), self.connection)
        self.drop_na()
        self.apply_dtypes()
        self.cut_off()
        self.apply_definition()
        return {self.name: self.df}
            