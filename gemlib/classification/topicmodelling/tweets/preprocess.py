import ast
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from gemlib.validation import utilities
from gemlib.abstarct.basefunctionality import BaseTextPreprocessing
import re

none_values = ['None', 'None', 'None', 'None', 'None', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                   np.nan]
columns = ['place_type', 'name', 'full_name', 'country_code', 'country', 'a.long', 'a.lat', 'b.long', 'b.lat',
               'c.long', 'c.lat', 'd.long', 'd.lat']

class TwitterPreprocessing(BaseTextPreprocessing):

    def get_cordinates(self, val):
        list_val = []
        for cor in val:
            list_val.extend(cor)
        return list_val

    def parse_place(self, val):
        list_val = []
        d = json.loads(val)
        list_val.append(d['place_type'])
        list_val.append(d['name'])
        list_val.append(d['full_name'])
        list_val.append(d['country_code'])
        list_val.append(d['country'])
        list_val.extend(self.get_cordinates(d['bounding_box']['coordinates'][0]))
        return list_val

    def add_features(self, df):
        df['c_long'] = df['a.long'] + df['d.long']
        df['c_long'] = df['c_long'] / 2
        df['c_lat'] = df['a.lat'] + df['b.lat']
        df['c_lat'] = df['c_lat'] / 2
        #bins 0.005:1km --- 0.05:6km --- 0.2:11km
        # df['lat_index'] = np.digitize(df.c_lat, [x for x in np.arange(np.min(df.c_lat), np.max(df.c_lat), self.bins)])
        # df['long_index'] = np.digitize(df.c_long, [x for x in np.arange(np.min(df.c_long), np.max(df.c_long), self.bins)])

        df['tweet.created_at'] = df['tweet.created_at'].astype(np.datetime64)
        df['tweet.created_at'] = df['tweet.created_at'] + pd.Timedelta(hours=10) # UTC to AEST
        df['hourofday'] = df['tweet.created_at'].dt.hour
        df['day'] = df['tweet.created_at'].dt.day
        df['month'] = df['tweet.created_at'].dt.month
        df['dayofweek'] = df['tweet.created_at'].dt.dayofweek
        df['dayofyear'] = df['tweet.created_at'].dt.dayofyear
        df['weekofyear'] = df['tweet.created_at'].dt.weekofyear

        # df_count = df.groupby(['lat_index', 'long_index'])[['tweet.id_str']].count().rename({'tweet.id_str': 'cell_tweet_count'},
        #                                                                                     axis='columns')
        # df_mean_long = df.groupby(['lat_index', 'long_index'])[['c_long']].mean().rename({'c_long': 'cell_long_mean'},
        #                                                                                axis='columns')
        # df_mean_lat = df.groupby(['lat_index', 'long_index'])[['c_lat']].mean().rename({'c_lat': 'cell_lat_mean'}, axis='columns')
        # df.set_index(['lat_index', 'long_index'], inplace=True)
        #
        # df = df.join(df_count)
        # df = df.join(df_mean_lat)
        # df = df.join(df_mean_long)
        df.reset_index(inplace=True)
        return df

    def preprocess_tweets(self, df:pd.DataFrame):
        df['tweet.place'] = df['tweet.place'].str.replace("\"", "'")
        df['tweet.place'] = df['tweet.place'].str.replace("{'", "{\"")
        df['tweet.place'] = df['tweet.place'].str.replace("':", "\":")
        df['tweet.place'] = df['tweet.place'].str.replace(": '", ": \"")
        df['tweet.place'] = df['tweet.place'].str.replace("',", "\",")
        df['tweet.place'] = df['tweet.place'].str.replace(", '", ", \"")

        all_values = []
        for p in tqdm(df['tweet.place']):
            try:
                if isinstance(p, float):
                    all_values.append(none_values)
                    continue
                all_values.append(self.parse_place(p))
            except Exception:
                p1 = p.replace("\":", "':")
                p1 = p1.replace(": \"", ": '")
                p1 = p1.replace("\",", "',")
                p1 = p1.replace(", \"", ", '")
                utilities._info(p)
                utilities._info(p1)
                all_values.append(none_values)
                pass
        df_place = pd.DataFrame(all_values, columns=columns)
        df = df.join(df_place)
        # filter other countries rather than Australia
        df = df[df['country'] == 'Australia'].copy()
        if len(df) < 10:
            raise Exception("Not enough data to run the preprocessing.")
        df.reset_index(inplace=True, drop=True)
        # decode utf-8
        df['tweet_text'] = [ast.literal_eval(x).decode('utf-8') for x in df['tweet_text']]
        df['tweet_text'].replace(',', '')
        # add more useful features
        df = self.add_features(df)
        return df

    def apply(self, df):
        return self.preprocess_tweets(df)

    def get_df(self):
        pass




