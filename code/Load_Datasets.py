'''
Use reverse Geocoding to get the county for each station in the IEM metadata
Match each county to its ANSI code
Run once at the beginning of the project (reverse geocoding takes ~ 9 minutes)
'''

import pandas as pd
import numpy as np
import geopandas as ge
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import great_circle
import os
import time

import cgi
import cgitb
import urllib.parse
import urllib.request
import csv
import os

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

def get_counties_center():
    '''
    Heper function: 
    get the centroid of each county in Iowa
    '''
    counties_shape = ge.read_file(os.path.join(parent, 'data/Iowa_County_Boundaries/IowaCounties.shp'))
    counties_shape = counties_shape.to_crs(epsg=4326)
    counties_shape['lon'] = counties_shape.centroid.x
    counties_shape['lat'] = counties_shape.centroid.y

    # change FIPS value to match the value in station data
    counties_shape['FIPS_INT'] = counties_shape['FIPS_INT'] - 19000

    return counties_shape[['FIPS_INT','CountyName', 'lon', 'lat']]

def find_nearest_station(county, stations):
    '''
    Helper function:
    find the nearest station for a county
    '''
    county_location = (county['lat'], county['lon'])
    stations['Distance'] = stations.apply(lambda row: great_circle(county_location, (row['lat'], row['lon'])).miles, axis=1)
    nearest_station = stations.loc[stations['Distance'].idxmin()]
    
    return nearest_station['stid'], nearest_station['Distance']

def change_counties(row, missing_counties_df):
    '''
    Helper functio:
    change the counties for each station 
    '''
    row['county'] = missing_counties_df.loc[missing_counties_df['stid'] == row['stid'], 'CountyName'].values[0]
    row['COUNTYFP'] = missing_counties_df.loc[missing_counties_df['stid'] == row['stid'], 'FIPS_INT'].values[0]
    return row

# Main classes:
# - CountyFinder
# - StationDataFetcher

class CountyFinder:
    '''
    Class to select stations from IEM metadata, one station for each county
    '''
    def __init__(self, meta_data_path, output_path):

        # import IEM metadata
        self.IEM_meta = pd.read_csv(meta_data_path, skiprows=9)
        self.output_path = output_path

        # initialize Nominatim geocoder
        self.geolocator = Nominatim(user_agent = "county_finder")

        # add delay to avoid overloading server
        self.geocode = RateLimiter(self.geolocator.reverse, min_delay_seconds=1)

    def get_county(self, lat, lon):
        '''
        Return county name for given latitude and longitude
        '''

        print(f'Getting county name for location {lat},{lon}')

        # get county name and county ANSI for given latitude and longitude
        try:
            location = self.geocode((lat,lon), language='en')
            if location and 'address' in location.raw:
                county = location.raw['address'].get('county', '')
                county_name = county.replace(' County', '').strip()
                return county_name
            else:
                return np.nan
        except Exception as e:
            print(f'Error in getting county for location {lat},{lon} : {str(e)}')
            return np.nan   

    def add_county_column(self):
        ''' 
        Add a column to the IEM metadata dataframe with county name for each station
        '''
        # implement reverse geocoding (takes 9 minutes)
        counties = self.IEM_meta.apply(lambda row: self.get_county(row['lat'], row['lon']), axis = 1)
        self.IEM_meta['county'] = counties
    
    def add_county_ansi_column(self):
        '''
        Add a column to the IEM metadata dataframe with county ANSI for each station
        '''
        # import county fips codes
        county_fips = pd.read_csv(os.path.join(parent,'references/county_fips_code.txt'), sep='|')
        county_fips['COUNTYNAME'] = county_fips['COUNTYNAME'].str.replace(' County','')

        # join county fips data with the filtered IEM_meta dataframe
        self.IEM_meta = pd.merge(self.IEM_meta, county_fips[['COUNTYFP','COUNTYNAME']], left_on = 'county', right_on = 'COUNTYNAME', how = 'right').drop(columns = 'COUNTYNAME')

    
    def __call__(self):

        self.add_county_column()
        self.add_county_ansi_column()
        self.IEM_meta.to_csv(self.output_path, index=False)


class StationDataFetcher:
    '''
    Class to download meteorology data for specified stations from website: https://mesonet.agron.iastate.edu 
    '''
    def __init__(self, 
                 start_date, 
                 end_date, 
                 IEM_meta_processed_path, 
                 output_csv_path,
                 missing_counties = 'Ignore'):
        
        self.start_date = start_date
        self.end_date = end_date
        self.output_csv_path = output_csv_path
        self.IEM_meta_processed = pd.read_csv(IEM_meta_processed_path)
        self.missing_counties = missing_counties
    def get_stations(self):
        '''
        filter IEM meta data for a specific time interval and unique counties
        '''
        logger.debug('Getting stations for time period and unique counties')

        data = self.IEM_meta_processed

        # Convert begints and endts to datetime
        data['begints'] = pd.to_datetime(data['begints'])
        data['endts'] = pd.to_datetime(data['endts'])

        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)

        # filter for the selected time period
        mask = (data['begints'] <= start_date) & ((data['endts'].isnull()) | (data['endts'] >= end_date))
        data = data[mask]

        # filter for unique counties
        if self.missing_counties == 'Ignore':
            data = data.drop_duplicates(subset=['county'])

        elif self.missing_counties == 'Nearest':
            # implement nearest neighbor approach to select nearest station for each county
            
            # extract list of missing counties
            unique_counties = self.IEM_meta_processed['COUNTYFP'].unique().astype(int)
            data_unique = data.drop_duplicates(subset=['county'])
            station_counties = data_unique['COUNTYFP'].unique()
            missing_counties = [county for county in unique_counties if county not in station_counties]

            # get coordinates of centroid for each county
            counties_center = get_counties_center()

            # match each county to nearest station
            missing_counties_df = counties_center[counties_center['FIPS_INT'].isin(missing_counties)]
            missing_counties_df[['stid', 'Distance']] = missing_counties_df.apply(find_nearest_station, stations=data, axis=1, result_type="expand")

            # create new dataframe with missing counties and their nearest stations in the same format as IEM_meta_processed
            data_missing = data[data['stid'].isin(missing_counties_df['stid'])]
            data_missing = data_missing.apply(lambda row: change_counties(row, missing_counties_df), axis=1)

            
            data = data.drop_duplicates(subset=['county'])
            data = pd.concat([data, data_missing])
            data.to_csv(os.path.join(parent, 'data/IEM_meta_processed_filtered.csv'))
        
        self.IEM_meta_processed_filtered = data[['stid', 'lat', 'lon', 'elev','county','COUNTYFP']]
        print(f'Number of unique counties: {len(data)}')
        self.stations = data['stid'].tolist()

    def get_station_data(self):
        
        logger.debug('Getting station data')

        # convert dates to the format required by the URL
        year1, month1, day1 = map(lambda x: x,self.start_date.split('-'))
        year2, month2, day2 = map(lambda x: x, self.end_date.split('-'))

        # Build the URL with the parameters
        url_head = "https://mesonet.agron.iastate.edu/cgi-bin/request/coop.py?network=IACLIMATE&stations="
        url_tail = "&vars%5B%5D=gdd_50_86&vars%5B%5D=high&vars%5B%5D=low&vars%5B%5D=precip&vars%5B%5D=era5land_srad&vars%5B%5D=era5land_soilt4_avg&vars%5B%5D=era5land_soilm4_avg&what=download&delim=comma&gis=no&scenario_year=2023"

        # construct the dates in the URL
        url_dates = f"&year1={year1}&month1={month1}&day1={day1}&year2={year2}&month2={month2}&day2={day2}"
        url = f"{url_head}{','.join(self.stations)}{url_dates}{url_tail}"

        # make the request to server
        try:
            with urllib.request.urlopen(url) as response:
                status_code = response.getcode()
                data = response.read().decode('utf-8')
            
                # Write data to CSV file
                with open(self.output_csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for line in data.splitlines():
                        writer.writerow(line.split(','))
                print(f"Data saved to {self.output_csv_path}")

                # Print status code for debugging
                print(f"Status Code: {status_code}")

        except urllib.error.HTTPError as e:
            print(f"HTTP Error: {e.code} {e.reason}")

        except urllib.error.URLError as e:
            print(f"URL Error: {e.reason}")
            
        except Exception as e:
            print(f"General Error: {e}")

    def merge_station_data(self):

        logger.debug('Merging station data with IEM metadata')

        station_data = pd.read_csv(self.output_csv_path, skiprows=4)
        merged_data = station_data.merge(self.IEM_meta_processed_filtered, how='inner', left_on='station', right_on='stid')
        merged_data.drop(columns=['stid'], inplace=True)
        merged_data.to_csv(self.output_csv_path, index=False)

    def __call__(self):
        self.get_stations()
        self.get_station_data()
        self.merge_station_data()

# Test:
# get current directory
# path = os.getcwd()
# # parent directory
# parent = os.path.dirname(path)


# IEM_meta_path = os.path.join(parent,'references/IEM_meta.csv')
# IEM_meta_processed_path = os.path.join(parent,'data/IEM_meta_processed.csv')
# station_data_path = os.path.join(parent,'data/station_data.csv')

# start_date = '1951-01-01'
# end_date = '2024-08-05'

# start = time.time()
# station_data_fetcher = StationDataFetcher('1951-01-01',
#                                                         '2024-08-05',
#                                                         IEM_meta_processed_path,
#                                                         station_data_path,
#                                                         missing_counties='Nearest')
# station_data_fetcher()


# end = time.time()
# print(f"Fetching IEM station data took {end - start} seconds.")