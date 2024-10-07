# Clean and merge datasets

import os
import pandas as pd
import numpy as np
import time
import re
import logging
import sys

# Set up logging
logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
if (logger.hasHandlers()):
    logger.handlers.clear()

logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)


# Define global variables

# get current directory
path = os.getcwd()

# parent directory
parent = os.path.dirname(path)

station_data_path = os.path.join(parent,'data/station_data.csv')
corn_data_path = os.path.join(parent,'data/USDA_corn.csv')

# Main class:

class CleanAndMerge:
    '''
    Class: 
    - input: path to station data, path to corn data, level and method of aggreatating monthly data
    - output: merged data saved to 'data/merged_data.csv'
    - clean station data and corn data
    - merge station data with corn data
    '''
    def __init__(self, 
                 station_data_path, 
                 corn_data_path,
                 agg_level = None,
                 agg_aggs = None):
        self.station_data = pd.read_csv(station_data_path)
        self.corn_data = pd.read_csv(corn_data_path)
    
    def remove_missing_data(self):
        
        # remove rows with missing data
        self.station_data['day'] = pd.to_datetime(self.station_data['day'], format='%Y/%m/%d')
        self.station_data = self.station_data[ self.station_data['day'] < pd.to_datetime('2024-08-01') ]
    
        logger.debug('Missing data removed')
    def convert_data_types(self, column_types):
        # convert columns to specified data types
        for col, type in column_types.items():
            self.station_data[col] = self.station_data[col].astype(type)
        
        logger.debug('Data type converted')
    
    # Feature engineering methods:
    def extract_ymd(self):
        # extract year, month, week and day from 'day' column
        self.station_data['year'] = self.station_data['day'].dt.year
        self.station_data['month'] = self.station_data['day'].dt.month
        self.station_data['week'] = self.station_data['day'].dt.isocalendar()['week']
        self.station_data['day'] = self.station_data['day'].dt.day

        logger.debug('ymd extracted')
    def shift_months(self):
        # shift month = 12 data to the next year
        
        # change month into a string
        self.station_data['month'] = self.station_data['month'].astype(str)
       
        # for all columns with month == 12, add year by one
        self.station_data.loc[self.station_data['month'] == '12', 'year'] += 1

        # for all instances where month == '12', change value to '12_prev
        self.station_data.loc[self.station_data['month'] == '12', 'month'] = '12_prev'

        # drop data during harvest months
        months_to_drop = ['9', '10', '11']
        self.station_data = self.station_data[~self.station_data['month'].isin(months_to_drop)]

        logger.debug('month shifted')
        logger.debug(self.station_data['month'].unique())
    def aggregate_values(self, level = ['year', 'month'], aggs = ['mean','std','min','max']):
        # convert daily time series data into monthly aggregates

        logger.debug('aggregating values')
        logger.debug(level)
        # perform aggregation
        station_data_agg = self.station_data.groupby(['station'] + level).agg(
            {'gdd_50_86' : aggs,
                'high' : aggs,
                'low' : aggs,
                'precip' : aggs,
                'era5land_srad' : aggs,
                'era5land_soilt4_avg' : aggs,
                'era5land_soilm4_avg' : aggs
                }
        )

        # rename multi-index columns
        station_data_agg.columns = station_data_agg.columns.map(lambda cn : "_".join(cn))
        

        # change aggregating level into a string
        station_data_agg = station_data_agg.reset_index()
        station_data_agg[f'{level[1]}'] = station_data_agg[f'{level[1]}'].astype(str)

        # reshape data
        station_data_agg = station_data_agg.pivot(index = ['station', 'year'], columns =f'{level[1]}')
        station_data_agg.columns = station_data_agg.columns.map(lambda cn : "_".join(cn))
        station_data_agg.reset_index(inplace = True)
        
        # drop null values
        station_data_agg.dropna(axis = 1, inplace = True)
        
        logger.debug(station_data_agg.head())
        

        # combine station_data_agg with the rest of the columns in station_data
        self.station_data= pd.merge(station_data_agg, 
                                    self.station_data[['station',
                                                  'year',
                                                  'lat',
                                                  'lon',
                                                  'elev',
                                                  'county',
                                                  'COUNTYFP']].drop_duplicates(), 
                                    on = ['station',
                                          'year'], 
                                    how = 'left')
        logger.debug('values aggregated')
    def merge_datasets(self):

        # merge station data with corn data
        self.merged_data = pd.merge(self.station_data, 
                               self.corn_data[['County ANSI', 
                                               'Year', 
                                               'Value']], 
                               left_on = ['COUNTYFP', 
                                          'year'],         
                               right_on = ['County ANSI', 
                                           'Year'], 
                               how = 'inner')
        
        self.merged_data = (self.merged_data
                                .drop(['COUNTYFP','County ANSI','station','Year'], axis=1)
                                .sort_values('year'))
        X = self.merged_data.drop(['Value'], axis=1)
        y = self.merged_data['Value']

        self.merged_data.to_csv(os.path.join(parent, 'data/merged_data.csv'), index = False)

        logger.debug('datasets merged')

        return self.merged_data
    
    def clean_and_merge_datasets(self):

        # perform cleaning and merging steps
        self.remove_missing_data()
        columns_to_convert = {'day' : 'datetime64[ns]', 
                      'era5land_srad' : 'float64', 
                      'era5land_soilt4_avg' : 'float64', 
                      'era5land_soilm4_avg' : 'float64'}
        self.convert_data_types(columns_to_convert)
        self.extract_ymd()
        self.shift_months()
        self.aggregate_values(level=['year','week'])
        
        self.merge_datasets()

        return self.merged_data



