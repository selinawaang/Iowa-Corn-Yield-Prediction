import os
import pandas as pd
import numpy as np
import time
import re

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import logging
import sys

# Set up logging
logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)

# Define global variables

# get current directory
path = os.getcwd()
# parent directory
parent = os.path.dirname(path)

# Helper functions:

def ts_split(df, test_years = 5):
    '''
    Helper function: split dataset into train and test based on years
    '''

    min_year = df['year'].min()
    total_years = df['year'].max() - df['year'].min() + 1

    train_years = total_years - test_years

    train_df = df[df['year'] < min_year + train_years]
    test_df = df[df['year'] >= min_year + train_years]

    return train_df, test_df

def make_encoder(df):
    '''
    Helper function: create encoder for transforming variables
    '''

    if 'county' in df.columns:
        onehot_ftrs = ['county']
        std_ftrs = df.columns.drop(onehot_ftrs).tolist()
    else:
        onehot_ftrs = []
        std_ftrs = df.columns.tolist()


    encoder = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(sparse_output=False), onehot_ftrs),
            ('std', StandardScaler(), std_ftrs),
            #'feature_selection', SelectFromModel(Lasso())
        ]
    )

    return encoder

# Main class:

class Preprocessing:
    '''
    Class: performs preprocessing, splitting feature engineering and feature selection
    '''
    def __init__(self, test_years=5, selection_method = 'lasso'):
        self.test_years = test_years
        self.selection_method = selection_method
        self.df = None

    def split(self, test_years):
        # split data into train and test
        self.train, self.test = ts_split(self.df, test_years)

    def add_lag_value(self):
        # add lag value for each county, shift 1 year
        self.df['lag_value'] = self.df.groupby('county')['Value'].shift(1)
        
        # remove null values after lag
        self.df.dropna(inplace=True)
    
    def add_year_trend(self):
        pass

    def feature_selection(self, method = 'lasso'):
        if method == 'lasso':
            # encode data for feature selection
            encoder = make_encoder(self.X_train)

            # implement lasso feature selection
            selector = SelectFromModel(Lasso())

            # connect steps using a pipeline
            pipe = Pipeline([('encoder', encoder),
                             ('selector', selector)])
            pipe.fit(self.X_train, self.y_train)

            # extract selected features
            X_train_reduced = pipe.transform(self.X_train)
            self.selected_features = list(map(lambda x: x.split('__')[1], pipe.get_feature_names_out()))
            
            # add county and year columns back in if they were not selected:
            if 'county' not in self.selected_features:
                self.selected_features.append('county')
            if 'year' not in self.selected_features:
                self.selected_features.append('year')
        else:
            raise ValueError('Unsupported feature selection method')
        
        self.X_train = self.X_train[self.selected_features]
        self.X_test = self.X_test[self.selected_features]
    
    def preprocess(self, df):
        
        # apply preprocessing, splitting, feature engineering and feature selection steps

        self.df = df.copy().sort_values('year')

        self.add_lag_value()
        self.split(5)

        # split data into X and y
        self.X_train = self.train.drop(['Value'], axis=1)
        self.y_train = self.train['Value']
        self.X_test = self.test.drop(['Value'], axis=1)
        self.y_test = self.test['Value']

        self.feature_selection()

        # save preprocessed data to CSV files
        self.X_train.to_csv(os.path.join(parent, 'data/X_train.csv'), index=False)
        self.X_test.to_csv(os.path.join(parent, 'data/X_test.csv'), index=False)
        self.y_train.to_csv(os.path.join(parent, 'data/y_train.csv'), index=False)
        self.y_test.to_csv(os.path.join(parent, 'data/y_test.csv'), index=False)

        print('Data preprocessing done')

        return self.X_train, self.X_test, self.y_train, self.y_test

