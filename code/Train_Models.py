import os
import pandas as pd
import numpy as np
import time
import re

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.ensemble import VotingRegressor

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import Preprocess_Data

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

data_dir = os.path.join(parent,'data/')

# Main class:
class MLYieldPredictor:
    '''
    Class: implements ML pipeline to predict yield

    inputs: 
    - model: ML algorithm to train
    - preprocessing_pipeline: preprocess pipeline
    - params: hyperparameters for tuning
    - exp_num: experiment number used to record different model experiments
    - n_splits: number of splits for cross-validation, default = 5

    outputs:
    - best_model: the model with the best validation score
    '''
    def __init__(self, model, preprocessing_pipeline, params, exp_num, n_splits = 5):
        self.model = model
        self.preprocessing_pipeline = preprocessing_pipeline
        self.params = params
        self.n_splits = n_splits
        self.best_model = None
        self.y = None
        self.X = None

        # make experiment folder
        self.exp_num = exp_num
        self.exp_folder = os.path.join(parent,f'experiments/experiment_{self.exp_num}/')
        # create experiment folder
        if not os.path.exists(self.exp_folder):
            os.makedirs(self.exp_folder)
    
    def train(self, X, y, cv_method = 'GridSearchCV'):
        # [todo] add bayesian hyper-parameter tuning


        # initialize algorithm
        algo = self.model()
        self.pipe = Pipeline([('preprocessor', self.preprocessing_pipeline), 
                         ('model', algo)])
        tsp = TimeSeriesSplit(n_splits = self.n_splits)
        # tune hyperparameters
        if cv_method == 'GridSearchCV':
            grid = GridSearchCV(self.pipe, 
                                param_grid = self.params, 
                                scoring = 'neg_mean_squared_error', 
                                cv = tsp, 
                                return_train_score = True, 
                                n_jobs=-1, 
                                verbose=True)            
            grid.fit(X, y)
            results = pd.DataFrame(grid.cv_results_)
            
            # save experiment results
            results.to_csv(os.path.join(self.exp_folder, 'cv_results.csv'))

            print('best model parameters:',grid.best_params_)
            print('validation score:',grid.best_score_)

            # save best model

            self.best_model = grid.best_estimator_

            save_file = open(os.path.join(self.exp_folder, 'model.save'), 'wb')
            pickle.dump(self.best_model, save_file)

    def predict(self, X):
        # make predictions using trained model on unseen or seen data
        # predict on test data
        self.y_pred = self.best_model.predict(X)
        return self.y_pred

    def evaluate(self, y_true, y_pred):
        # evaluate model using metrics like R2, MAE, RMSE
        # [to do:] output multiple metrics

        mse_overall = mean_squared_error(y_true, y_pred)

        return mse_overall


# Other training functions:

class VisualizePredictions():
    '''
    Class: visualizes predictions

    inputs:
    - X: matrix of features 
    - y_true: true yield values
    - y_pred: predicted yield values

    outputs:
    - displays visualizations
    '''

    def __init__(self, X, y_true, y_pred):
        self.X = X
        self.y_true = y_true
        self.y_pred = y_pred
        self.X['y_true'] = y_true
        self.X['y_pred'] = y_pred
        self.X['residuals'] = self.X['y_true'] - self.X['y_pred']
    def plot_predictions(self):

        # visualize predictioned values vs true values
        figure = plt.figure(figsize=(12, 8))
        sns.scatterplot(data=self.X, x = "y_true", y = 'y_pred', hue = 'year', alpha = 0.5)
        plt.show()


    def plot_residuals_county_year(self):

        # visualize residuals for each county and year as a heatmap
        pivot_true = self.X.pivot(index = 'county', columns = 'year', values = 'y_true')
        pivot_pred = self.X.pivot(index = 'county', columns = 'year', values = 'y_pred')
        pivot_residuals = pivot_true - pivot_pred
        sns.heatmap(data=pivot_residuals, cmap='coolwarm')
        plt.title('Residuals by County and Year')
        plt.show()       

    def plot_mse_county(self):
        # visualize residuals for each county as a bar plot

        # calculate mse for each county
        county_mse = self.X.groupby('county').apply(lambda df: mean_squared_error(df['y_true'], df['y_pred'])).reset_index(name='county_mse')
        X_merged = pd.merge(self.X, county_mse, on='county', how='left')
        figure = plt.figure(figsize=(20,20))
        sns.barplot(data = X_merged.sort_values(by='county_mse', ascending=True), 
                    x = 'county_mse',
                    y = 'county',
                    palette = 'rocket')
        plt.show()

    def plot_residuals_county(self):

        # visualize residuals for each county as an error chart
        plt.figure(figsize = (20,10))
        sns.pointplot(data = self.X.sort_values(by='residuals', ascending=True), 
                    x = 'county',
                    y = 'residuals',
                    errorbar = 'sd',
                    palette = 'rocket')

        plt.xlabel('County')
        plt.ylabel('Residuals')
        plt.title('Distribution of Residuals by County')
        plt.xticks(rotation=90)
        plt.show()

    
    def plot_mse_year(self):

        # calculate mse for each year
        year_mse = self.X.groupby('year').apply(lambda df: mean_squared_error(df['y_true'], df['y_pred'])).reset_index(name='year_mse')
        #print(year_mse)
        #X_merged = pd.merge(X, year_mse, on='county', how='left')

        plt.figure(figsize = (20,10))
        sns.barplot(data = year_mse.sort_values(by='year', ascending=True), 
                    x = 'year',
                    y = 'year_mse',
                    color = 'darkkhaki')

        plt.xlabel('Year')
        plt.ylabel('Mean Square Error')
        plt.title('Mean Squared Error by Year')

        plt.show()         

