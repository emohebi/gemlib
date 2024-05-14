from gemlib.abstarct.basefunctionality import BaseDataLoader

class spark_csv_loader(BaseDataLoader):

    def __init__(self, sc, path):
        pass
        # this.sc = sc
        # this.path = path

    def load(self):
        raise NotImplementedError
        # sqlContext = SQLContext(self.sc)
        # return sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(self.path).toPandas()