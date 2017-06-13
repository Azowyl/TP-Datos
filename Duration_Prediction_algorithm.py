# Python 2
from __future__ import division

# Dataframe
import pandas as pd
import numpy as np

# Time measurement
import datetime
from time import time

# The Hashing Trick
from sklearn import preprocessing
from sklearn.feature_extraction import FeatureHasher

# Dimension Reduction
from sklearn.decomposition import TruncatedSVD

# ML
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression


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


def impute_values(data, strategy):

    return preprocessing.Imputer(missing_values='NaN', strategy=strategy, axis=0).fit_transform(data)


def train_model(x, y):
    return BayesianRidge(lambda_1=1.e-6, lambda_2=1.e-6, noramilize=True).fit(x, y)


#input
stations = pd.read_csv('data/station.csv')
trips = pd.read_csv('data/trip.csv', parse_dates=['start_date', 'end_date'], infer_datetime_format=True,low_memory=False)
weather = pd.read_csv('data/weather.csv', parse_dates=['date'], infer_datetime_format=True)
cities = pd.read_csv('data/cities.csv')

trip_train = pd.read_csv('data/trip_train.csv')
trip_test = pd.read_csv('data/trip_test.csv')


train_set, test_set = preprocess_data(stations, weather, trip_train, trip_test)

column_labels = ['start_station_id','end_station_id','subscription_type','city_zip_code','events']
train_set = the_Hashing_Trick(train_set, column_labels)
test_set = the_Hashing_Trick(test_set, column_labels)

target_values = train_set['duration']
train_set.drop(labels='duration', axis=1, inplace=True)

ids = test_set['id']
trip_test.drop(labels=['id'], axis=1, inplace=True)

train_set = impute_values(train_set)
test_set = impute_values(test_set)

regressor = train_model(train_set, target_values)

results = pd.DataFrame(data=regressor.predict(trip_test), index=ids).reset_index(inplace=True)
results.columns = ['id', 'duration']

results.to_csv('results.csv', index=False)
