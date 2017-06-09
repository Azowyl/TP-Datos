# Python 2
from __future__ import division

# Preprocessing
from sklearn import preprocessing

# Dataframe
import pandas as pd
import numpy as np

# Time measurement
import datetime
from time import time

# Visu
import matplotlib.pyplot as plt

# The Hashing Trick
from sklearn.feature_extraction import FeatureHasher

# Dimension Reduction
from sklearn.decomposition import TruncatedSVD

# ML
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor

# Scoring
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

stations = pd.read_csv('data/station.csv')
trips = pd.read_csv('data/trip_train.csv', parse_dates=['start_date', 'end_date'], infer_datetime_format=True,low_memory=False)
weather = pd.read_csv('data/weather.csv', parse_dates=['date'], infer_datetime_format=True)
cities = pd.read_csv('data/cities.csv')


def preprocess_data(stations, weather, train_trips, test_trips):

    stations = pd.merge(stations, cities, on='city', how='left').loc[:, ['id', 'zip_code']]
    stations.columns = ['start_station_id', 'city_zip_code']

    train_trips = pd.merge(train_trips, stations, on='start_station_id', how='left')
    train_trips['date'] = pd.to_datetime(train_trips.start_date.dt.date)
    train_trips.start_date = train_trips.start_date.apply(lambda x: int((x - datetime.datetime(1994, 4, 29)).total_seconds()))
    train_trips.end_date = train_trips.end_date.apply(lambda x: int((x - datetime.datetime(1994, 4, 29)).total_seconds()))

    test_trips = pd.merge(test_trips, stations, on='start_station_id', how='left')
    test_trips['date'] = pd.to_datetime(test_trips.start_date.dt.date)
    test_trips.start_date = test_trips.start_date.apply(lambda x: int((x - datetime.datetime(1994, 4, 29)).total_seconds()))
    test_trips.end_date = test_trips.end_date.apply(lambda x: int((x - datetime.datetime(1994, 4, 29)).total_seconds()))

    weather.rename(columns={'zip_code': 'city_zip_code'}, inplace=True)
    weather.precipitation_inches = pd.to_numeric(weather.precipitation_inches, errors='coerse')
    weather.precipitation_inches.fillna(0, inplace=True)

    train_data = pd.merge(train_trips, weather, on=['date', 'city_zip_code'], how='inner')
    train_data.drop(labels=['id', 'start_station_name', 'end_station_name', 'date', 'zip_code', 'bike_id'], axis=1, inplace=True)

    test_data = pd.merge(test_trips, weather, on=['date', 'city_zip_code'], how='inner')
    test_data.drop(labels=['id', 'start_station_name', 'end_station_name', 'date', 'zip_code', 'bike_id'], axis=1, inplace=True)

    return train_data, test_data

def the_Hashing_Trick(data, column_labels):

    temp = []

    for row in data.loc[:, column_labels].iterrows():
        index, value = row
        for i in range(len(value)):
            value[i] = str(value[i])
        temp.append(value.tolist())

    h = FeatureHasher(n_features=10, input_type='string')
    f = h.transform(raw_X=temp)

    data = pd.concat([data, pd.DataFrame(f.todense())], axis=1)
    data.drop(labels=column_labels, axis=1, inplace=True)

    return data

def data_standarization(train_data, test_data = None):

    scaler = preprocessing.StandardScaler().fit(train_data)

    data = scaler.transform(train_data)

    if (test_data != None):
        test_data = scaler.transform(test_data)
        return data, test_data

    return data

def reduce_dimentions_to(data, n_dimensions = 5):

    svd = TruncatedSVD(n_components=n_dimensions, n_iter=7, random_state=42)

    return svd.fit_transform(data)


