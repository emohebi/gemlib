from gemlib.abstarct.basefunctionality import BaseDataLoader
import sys
import pandas as pd

#class redis_data_loader(BaseDataLoader):

#    def load(self):
#        try:
#            db = redis.StrictRedis(host=self.path,
#                                   port=self.port,
#                                   db=0,
#                                   charset="utf-8", decode_responses=True)
#            list_records = []
#            dict_of_df = {}
#            for k in db.keys():
#                list_records.append(db.hgetall(k))
#            self.df = pd.DataFrame(list_records)
#            self.apply_dtypes()
#            dict_of_df['0'] = self.df
#            return dict_of_df
#        except:
#            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
#            print("loading data from Redis failed!!!\n {0}".format(exceptionValue))

#class redis_data_writer(BaseDataLoader):

#    def write(self):
#        try:
#            db = redis.StrictRedis(host=self.path,
#                                   port=self.port,
#                                   db=0,
#                                   charset="utf-8", decode_responses=True)
#            db.flushdb()
#            print("...db flushed...")
#            dict_value = self.df.T.to_dict()
#            for k in dict_value.keys():
#                db.hmset(k, dict_value[k])
#            print("data has been stored to Redis @ {}:{}".format(self.path, self.port))
#        except:
#            exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
#            print("staging data to Redis failed!!!\n {0}".format(exceptionValue))
