import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from .logging_wrappers import timeit_log, trace_log
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import linregress
from loguru import logger
from multiprocesspandas import applyparallel
from functools import reduce
import multiprocessing
import sys


class FeatureEngineering:
    def __init__(self):
        super().__init__()
        self.devices_water_temp = None
        self.seasonal_features = None
        self.county_data_1 = None
        self.county_data_2 = None
        self.hist_weather_data = None

    # FEATURE ENGINEERING
    @timeit_log
    def feature_engineering(self,
                            inference: bool,
                            data: pd.DataFrame,
                            recalc_seasonal=False,
                            num_proc=multiprocessing.cpu_count(),
                            devices_hist_temp_path: str = 'additional_files/devices_temperature.pickle') -> pd.DataFrame:
        """
        Add new features to dataset.

        :param inference:              whether inference stage or not (train)
        :param data:                   data frame with initial data
        :param recalc_seasonal:        whether recalculate seasonal features or take them from external file
        :param num_proc:               number of processes for parallelization
        :param devices_hist_temp_path: use ext. devices temperature file, calculated from db data (average & variance
                                       LOHTRTMP, UPHTRTMP in months for each mac_address)

        :return:                       data frame with collected features
        """
        if data is None:
            logger.opt(colors=True).error("<lr>Feature engineering was failed. The data are empty!</lr>")
            raise Exception('The data for feature engineering are empty')

        if self.devices_water_temp is None:
            self.devices_water_temp = self._get_devices_temp_data(devices_hist_temp_path)

        data = pd.merge(data,
                        self.devices_water_temp,
                        how='left',
                        on=['mac_address', 'product_type', 'month'])

        if self.hist_weather_data is None:
            self.hist_weather_data = self._get_weather_features()

        data = pd.merge(data,
                        self._get_county_features(data,
                                                  path_1='./additional_files/county_file_1.csv',
                                                  path_2='./additional_files/county_file_2.csv',
                                                  num_proc=num_proc,
                                                  ),
                        how='left',
                        on=['mac_address', 'product_type', 'state', 'city', 'zipcode'])

        if recalc_seasonal and not inference:
            self._calc_seasonal_decompose(data, path='./additional_files/seasonal_decompose_by_county.csv')

        data = pd.merge(data,
                        self._get_time_series_features(data,
                                                       path='./additional_files/seasonal_decompose_by_county.csv'),
                        on=['mac_address', 'product_type', 'state', 'county', 'year', 'month', 'day', 'datetime'])

        if inference:
            # For prediction delete useless data and parse weather from website
            data = pd.merge(data,
                            self.hist_weather_data.groupby(['state',
                                                            'county',
                                                            'month']).agg(avg_temp=('avg_temp', 'mean'),
                                                                          avg_prec=('avg_prec', 'mean'),
                                                                          avg_hdd=('avg_hdd', 'mean'),
                                                                          avg_cdd=('avg_cdd', 'mean'),
                                                                          hist_temp=('hist_temp', 'mean'),
                                                                          rank_temp=('rank_temp', 'mean'),
                                                                          anomaly_temp=('anomaly_temp', 'mean'),
                                                                          hist_prec=('hist_prec', 'mean'),
                                                                          rank_prec=('rank_prec', 'mean'),
                                                                          anomaly_prec=('anomaly_prec', 'mean'),
                                                                          anomaly_cdd=('anomaly_cdd', 'mean'),
                                                                          hist_hdd=('hist_hdd', 'mean'),
                                                                          rank_hdd=('rank_hdd', 'mean'),
                                                                          anomaly_hdd=('anomaly_hdd', 'mean'),
                                                                          hist_cdd=('hist_cdd', 'mean'),
                                                                          rank_cdd=('rank_cdd', 'mean'),
                                                                         ).reset_index(),
                            how='left',
                            on=['state', 'county', 'month'])

            # data = self._fill_weather_forecast(data, parse_weather)
        else:
            data = pd.merge(data,
                            self.hist_weather_data,
                            how='left',
                            on=['state', 'county', 'year', 'month'])

        # Fill NaN values in averages by mean values calculated for state, and NaN hist values by average ones
        data = pd.merge(data, data.groupby('state').agg(avg_temp_state=('avg_temp', 'mean'),
                                                        avg_prec_state=('avg_prec', 'mean'),
                                                        avg_hdd_state=('avg_hdd', 'mean'),
                                                        avg_cdd_state=('avg_cdd', 'mean')).reset_index(),
                        how='left', on='state')

        fill_na_features = ['avg_temp', 'avg_prec', 'avg_hdd', 'avg_cdd']
        for feature in fill_na_features:
            data[feature] = data[feature].fillna(data[f'{feature}_state'])
            data[feature.replace('avg', 'hist')] = data[feature.replace('avg', 'hist')].fillna(data[feature])
        data.drop(columns=['avg_temp_state', 'avg_prec_state', 'avg_hdd_state', 'avg_cdd_state'],
                  inplace=True,
                  errors='ignore')

        # The duplicates with different counties can appear, so, averaging it
        data = data.groupby(['mac_address',
                             'datetime',
                             'year',
                             'month',
                             'day',
                             'state',
                             'county',
                             'city',
                             'zipcode',
                             'region',
                             'time_zone',
                             'zipcode_group',
                             'product_type',
                             'macs_amount',
                             'weekday']).agg(energy_usage=('energy_usage', 'mean'),
                                             average_LOHTRTMP=('average_LOHTRTMP', 'mean'),
                                             average_UPHTRTMP=('average_UPHTRTMP', 'mean'),
                                             disperse_LOHTRTMP=('disperse_LOHTRTMP', 'mean'),
                                             disperse_UPHTRTMP=('disperse_UPHTRTMP', 'mean'),
                                             lat=('lat', 'mean'),
                                             lng=('lng', 'mean'),
                                             population_perc=('population_perc', 'mean'),
                                             seasonal=('seasonal', 'mean'),
                                             trend=('trend', 'mean'),
                                             avg_temp=('avg_temp', 'mean'),
                                             avg_prec=('avg_prec', 'mean'),
                                             avg_hdd=('avg_hdd', 'mean'),
                                             avg_cdd=('avg_cdd', 'mean'),
                                             hist_temp=('hist_temp', 'mean'),
                                             rank_temp=('rank_temp', 'mean'),
                                             anomaly_temp=('anomaly_temp', 'mean'),
                                             hist_prec=('hist_prec', 'mean'),
                                             rank_prec=('rank_prec', 'mean'),
                                             anomaly_prec=('anomaly_prec', 'mean'),
                                             anomaly_cdd=('anomaly_cdd', 'mean'),
                                             hist_hdd=('hist_hdd', 'mean'),
                                             rank_hdd=('rank_hdd', 'mean'),
                                             anomaly_hdd=('anomaly_hdd', 'mean'),
                                             hist_cdd=('hist_cdd', 'mean'),
                                             rank_cdd=('rank_cdd', 'mean'),
                                             ).reset_index()

        data = self._get_shifted_energy_data(data)

        return data

    @trace_log
    def _get_devices_temp_data(self, path: str) -> pd.DataFrame:
        return pd.read_pickle(path)

    @trace_log
    def _get_shifted_energy_data(self, df):
        """
        Method add shifted energy usage features
        """
        def get_shifted_energy(df: pd.DataFrame, shift: int) -> pd.DataFrame:
            shifted_energy = df.sort_values(['product_type', 'mac_address', 'county', 'datetime']).groupby(
                ['product_type', 'mac_address'])['energy_usage'].shift(shift)
            shifted_energy.name = f'energy_usage_lag{shift}'
            return shifted_energy

        df['timestamp'] = (df.datetime.astype(np.int64) // 10 ** 9).values
        df = pd.merge(df, get_shifted_energy(df, shift=1), how='left', left_index=True, right_index=True)
        df = pd.merge(df, get_shifted_energy(df, shift=2), how='left', left_index=True, right_index=True)
        return df

    @trace_log
    def _calc_seasonal_decompose(self, data: pd.DataFrame, path: str) -> pd.DataFrame:
        """
        Run only if you are going to recalc seasonal decompose features: seasonal and trend for each state
        """
        def calc_seasons_data(device_data: pd.DataFrame) -> pd.DataFrame:
            composed_data = pd.DataFrame()
            # For each county in states try get decompose for county elif - for state elif - for whole device
            for state in tqdm(device_data['state'].unique(), desc='States processed',
                              colour='GREEN', delay=0.001, file=sys.stdout):
                for county in device_data.loc[device_data['state'] == state, 'county'].unique():
                    try:
                        result = seasonal_decompose(
                            device_data.loc[(device_data['state'] == state) & (device_data['county'] == county),
                            ['datetime', 'energy_usage']].set_index('datetime').resample('D').mean().interpolate(limit_direction='both'),
                            period=365, model='multiplicative')
                    except ValueError:
                        try:
                            result = seasonal_decompose(
                                device_data.loc[device_data['state'] == state, ['datetime', 'energy_usage']].set_index(
                                    'datetime').resample('D').mean().interpolate(limit_direction='both'),
                                period=365, model='multiplicative')
                        except ValueError:
                            result = device_data.groupby('datetime').agg(energy_usage=('energy_usage', np.mean))
                            result = seasonal_decompose(result.resample('D').mean().interpolate(),
                                                        period=365, model='multiplicative')
                    temp_data = pd.concat([result.trend, result.seasonal], axis=1)
                    temp_data['state'] = state
                    temp_data['county'] = county
                    temp_data.reset_index(inplace=True)
                    temp_data['timestamp'] = (temp_data.datetime.astype(np.int64) // 10 ** 9).values
                    regression_data = temp_data[['timestamp', 'trend']].dropna()
                    x = regression_data['timestamp'].values
                    y = regression_data['trend'].values
                    slope, intercept, r, p, se = linregress(x, y)
                    temp_data['slope'] = slope
                    temp_data['intercept'] = intercept
                    composed_data = pd.concat([composed_data, temp_data], axis=0)

            composed_data.drop(columns=['timestamp', 'trend'], inplace=True)
            composed_data = composed_data.loc[
                            (composed_data['datetime'] > '2020-12-31') & (composed_data['datetime'] < '2022-01-01'), :]
            return composed_data

        def calc_device_seasons(df: pd.DataFrame) -> pd.DataFrame:
            for device in df['product_type'].unique():
                device_data = df.loc[df['product_type'] == device, ['datetime', 'state', 'county', 'energy_usage']]
                result_device = calc_seasons_data(device_data)
                result_device['product_type'] = device
                yield result_device

        df = data.groupby(['product_type', 'state', 'county', 'city', 'zipcode', 'datetime', 'year', 'month', 'day']).agg(
            energy_usage=('energy_usage', np.mean)).reset_index()
        df = df.groupby(['product_type', 'state', 'county', 'datetime', 'year', 'month', 'day']).agg(
            energy_usage=('energy_usage', np.mean)).reset_index()

        result = pd.concat([item for item in calc_device_seasons(df)], axis=0)
        result['month'] = result['datetime'].dt.month
        result['day'] = result['datetime'].dt.day
        result.drop(columns='datetime', inplace=True)
        result.reset_index(drop=True, inplace=True)
        result.to_csv(path)
        return result

    @trace_log
    def _get_time_series_features(self, data: pd.DataFrame, path: str) -> pd.DataFrame:
        """
        Load ext. time series features and merge with data frame
        """
        df = data.loc[:, ['mac_address', 'product_type', 'state', 'county', 'year', 'month', 'day', 'datetime']]
        df['weekday'] = df.apply(lambda row: row['datetime'].weekday() + 1, axis=1).astype('int8')
        if self.seasonal_features is None:
            self.seasonal_features = pd.read_csv(path, index_col=0)
        df = pd.merge(df, self.seasonal_features, how='left', on=['product_type', 'state', 'county', 'month', 'day'])
        df['timestamp'] = (df.datetime.view(np.int64) // 10 ** 9).values
        df['trend'] = df['slope'] * df['timestamp'] + df['intercept']
        df.drop(columns=['timestamp', 'slope', 'intercept'], inplace=True)
        return df

    @trace_log
    def _get_county_features(self,
                             data: pd.DataFrame,
                             path_1: str,
                             path_2: str,
                             num_proc: int = 1,
                             ) -> pd.DataFrame:
        """
        Method merged ext. files with counties in order to group data frame by counties
        """
        if self.county_data_1 is None:
            self.county_data_1 = pd.read_csv(path_1, index_col=0, dtype={'zipcode_group': 'int8',
                                                                         'zipcode': 'object',
                                                                         'lat': 'float32',
                                                                         'lng': 'float32',
                                                                         })

        self.county_data_1['zips_4'] = self.county_data_1['zipcode'].apply_parallel(lambda x: (str(x)[:-1]),
                                                                                    num_processes=num_proc)
        self.county_data_1['zips_3'] = self.county_data_1['zipcode'].apply_parallel(lambda x: (str(x)[:-2]),
                                                                                    num_processes=num_proc)
        if self.county_data_2 is None:
            self.county_data_2 = pd.read_csv(path_2, index_col=0)

        df = data.loc[:, ['mac_address', 'product_type', 'state', 'city', 'zipcode']]
        df['zips_4'] = df['zipcode'].apply_parallel(lambda x: (str(x)[:-1]), num_processes=num_proc)
        df['zips_3'] = df['zipcode'].apply_parallel(lambda x: (str(x)[:-2]), num_processes=num_proc)

        df = pd.merge(df, self.county_data_1[['state', 'county', 'zipcode', 'lat', 'lng', 'population_perc']].drop_duplicates('zipcode'),
                      how='left', on=['state', 'zipcode'])
        df = pd.merge(df, self.county_data_1[['state', 'county', 'city', 'lat', 'lng', 'population_perc']].drop_duplicates(),
                      how='left', on=['state', 'city'], suffixes=('', '_z'))
        df = pd.merge(df, self.county_data_2[['state', 'county', 'city']].drop_duplicates(),
                      how='left', on=['state', 'city'], suffixes=('', '_c'))
        df = pd.merge(df, self.county_data_1[['state', 'county', 'zips_4']].drop_duplicates('zips_4'),
                      how='left', on=['state', 'zips_4'], suffixes=('', '_z4'))
        df = pd.merge(df, self.county_data_1[['state', 'county', 'zips_3']].drop_duplicates('zips_3'),
                      how='left', on=['state', 'zips_3'], suffixes=('', '_z3'))
        df['lat'] = df['lat'].fillna(df['lat_z'])
        df['lng'] = df['lng'].fillna(df['lng_z'])
        df['population_perc'] = df['population_perc'].fillna(df['population_perc_z'])
        df['county'] = df['county'].fillna(df['county_z'])
        df['county'] = df['county'].fillna(df['county_c'])
        df['county'] = df['county'].fillna(df['county_z4'])
        df['county'] = df['county'].fillna(df['county_z3'])
        df = pd.merge(df, self.county_data_1.groupby(['state', 'county']).agg(lat_county=('lat', 'mean'),
                                                                              lng_county=('lng', 'mean')).reset_index(),
                      how='left', on=['state', 'county'])
        df['lat'] = df['lat'].fillna(df['lat_county'])
        df['lng'] = df['lng'].fillna(df['lng_county'])
        df.drop(columns=['zips_3', 'zips_4', 'county_c', 'county_z', 'county_z3', 'county_z4',
                         'population_perc_z', 'lat_z', 'lng_z', 'lat_county', 'lng_county'],
                inplace=True, errors='ignore')
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # SOME MANUAL CORRECTION FOR FURTHER PROCESSING
        name_replace = [('ME', 'St. John Plantation', 'Aroostook'),
                        ('ME', 'Tremont', 'Hancock'),
                        ('ME', 'Unorganized Territory Of Perkins', 'Sagadahoc'),
                        ('ME', 'Westport', 'Lincoln'),
                        ('ME', 'Chain Of Ponds', 'Franklin'),
                        ('ME', 'Freeman Township', 'Franklin'),
                        ('CA', 'New Pine Creek', 'Modoc'),
                        ('FL', "O'Brien", 'Suwannee'),
                        ('PA', "Feasterville-Trevose", 'Bucks'),
                        ('MI', "Sister Lakes", 'Van Buren'),
                        ('HI', "Kailua-Kona", 'Hawaii'),
                        ('NM', "Kingshill", 'Virgin Islands'),
                        ('VA', 'Lexington', 'Rockbridge')
                        ]
        for item in name_replace:
            df.loc[(df['state'] == item[0]) & (df['city'] == item[1]), 'county'] = item[2]

        df['county'] = df['county'].apply(lambda x: str(x).title())
        replace_df = [(' County', ''),
                      (' Borough', ''),
                      ('City ', ''),
                      (' Area', ''),
                      (' Census', ''),
                      (' Municipality', ''),
                      (' Parish', ''),
                      ('St.', 'Saint'),
                      ("'S", 's'),
                      ('-', ' '),
                      ('ñ', 'n'),
                      ('Dewitt', 'De Witt'),
                      ('Desoto', 'De Soto'),
                      ('DeSoto', 'De Soto'),
                      ('Dekalb', 'De Kalb'),
                      ('Lasalle', 'La Salle'),
                      ('Laporte', 'La Porte'),
                      ]

        for item in replace_df:
            df['county'] = df['county'].apply(lambda x: x.replace(item[0], item[1]))

        city_counties = ['Alexandria',
                         'Buena Vista',
                         'Charlottesville',
                         'Chesapeake',
                         'Danville',
                         'Fairlawn',
                         'Fredericksburg',
                         'Hampton',
                         'Harrisonburg',
                         'Lynchburg',
                         'Manassas',
                         'North Prince George',
                         'Newport News',
                         'Norfolk',
                         'Portsmouth',
                         'Staunton',
                         'Suffolk',
                         'Virginia Beach',
                         'Waynesboro',
                         'Williamsburg',
                         'Winchester']

        for city_county in city_counties:
            df.loc[(df['state'] == 'VA') & (df['city'] == city_county), 'county'] += ' City'
        df = pd.merge(df, self.county_data_1[['state',
                                              'region',
                                              'time_zone',
                                              'zipcode_group']].drop_duplicates().reset_index(drop=True),
                      how='left', on='state')
        df.loc[(df['state'] == 'HI') & (df['county'] == 'Nan'), 'county'] = 'Hawaii'

        # Fill missed coordinates by mean in counties
        df = pd.merge(df, df.groupby(['state', 'county']).agg(lat_county=('lat', 'mean'),
                                                              lng_county=('lng', 'mean'),
                                                              population_county=('population_perc', np.median)).reset_index(),
                      how='left', on=['state', 'county'])
        df['lat'] = df['lat'].fillna(df['lat_county'])
        df['lng'] = df['lng'].fillna(df['lng_county'])
        df['population_perc'] = df['population_perc'].fillna(df['population_county'])

        # Fill missed coordinates by mean in states
        df = pd.merge(df, df.groupby('state').agg(lat_state=('lat', 'mean'),
                                                  lng_state=('lng', 'mean'),
                                                  population_state=('population_perc', np.median)).reset_index(),
                      how='left', on='state')
        df['lat'] = df['lat'].fillna(df['lat_state'])
        df['lng'] = df['lng'].fillna(df['lng_state'])
        df['population_perc'] = df['population_perc'].fillna(df['population_state'])

        df.drop(columns=['lat_county', 'lng_county', 'lat_state', 'lng_state',
                         'population_county', 'population_state'], inplace=True, errors='ignore')
        return df

    @trace_log
    def _get_weather_features(self):
        """
        Method obtained county weather features from ext. file
        """
        def load_weather_parameter(path: str, suffix='') -> pd.DataFrame:
            historical_data = pd.DataFrame()
            for file in tqdm(glob(path), desc='Files reading', colour='GREEN', delay=0.01, file=sys.stdout):
                temp_data = pd.read_csv(file, skiprows=3)
                temp_data['state'] = temp_data['Location ID'].str.split('-', expand=True).drop(columns=1).values
                file_date = file.split('-')[-2]
                temp_data['year'] = int(file_date[:4])
                temp_data['month'] = int(file_date[4:])
                historical_data = pd.concat([historical_data, temp_data], axis=0)

            historical_data = historical_data.astype({'Rank': 'Int16',
                                                      'Value': 'float32',
                                                      'Anomaly (1901-2000 base period)': 'float32',
                                                      '1901-2000 Mean': 'float32',
                                                      'year': 'int16',
                                                      'month': 'int8',
                                                      })

            historical_data.rename(columns={'Value': f'hist_{suffix}',
                                            'Rank': f'rank_{suffix}',
                                            'Anomaly (1901-2000 base period)': f'anomaly_{suffix}',
                                            '1901-2000 Mean': f'avg_{suffix}'},
                                   inplace=True)
            return historical_data

        historical_temperature = load_weather_parameter('./additional_files/US_county_historical_temperature/*.csv',
                                                        suffix='temp')
        historical_precipitation = load_weather_parameter('./additional_files/US_county_historical_precipitation/*.csv',
                                                          suffix='prec')
        historical_hdd = load_weather_parameter('./additional_files/US_county_historical_HDD/*.csv', suffix='hdd')
        historical_cdd = load_weather_parameter('./additional_files/US_county_historical_CDD/*.csv', suffix='cdd')
        historical_weather = reduce(lambda left, right: pd.merge(left, right,
                                                                 on=['Location ID',
                                                                     'Location',
                                                                     'state',
                                                                     'year',
                                                                     'month']),
                                    [historical_temperature, historical_precipitation, historical_hdd, historical_cdd])
        historical_weather = historical_weather.drop(columns='Location ID').rename(columns={'Location': 'county'})
        replace_htd = [(' County', ''),
                       (' Borough', ''),
                       ('City ', ''),
                       (' Area', ''),
                       (' Census', ''),
                       (' Municipality', ''),
                       (' Parish', ''),
                       (' and', ''),
                       (' And', ''),
                       ('St.', 'Saint'),
                       ("'S", 's'),
                       ('-', ' '),
                       ('Dewitt', 'De Witt'),
                       ('Desoto', 'De Soto'),
                       ('Dekalb', 'De Kalb'),
                       ('Lasalle', 'La Salle'),
                       ('Laporte', 'La Porte'),
                       ('Lewis Clark', 'Lewis And Clark'),
                       ]

        for item in replace_htd:
            historical_weather['county'] = historical_weather['county'].apply(lambda x: x.replace(item[0], item[1]))

        historical_weather.loc[historical_weather['county'] == 'Washington, D.C.', ['state', 'county']] = ['DC', 'District Of Columbia']
        historical_weather['county'] = historical_weather['county'].apply(lambda x: str(x).title())

        # ADD Hawaii data
        hawaii_avg_temp = pd.DataFrame(data=[['HI', 1, 73],
                                             ['HI', 2, 73],
                                             ['HI', 3, 74.5],
                                             ['HI', 4, 76],
                                             ['HI', 5, 78],
                                             ['HI', 6, 80],
                                             ['HI', 7, 81],
                                             ['HI', 8, 82],
                                             ['HI', 9, 81.5],
                                             ['HI', 10, 80],
                                             ['HI', 11, 77.5],
                                             ['HI', 12, 74.5]],
                                       columns=['state', 'month', 'hist_temp']
                                       )
        hawaii_counties = pd.DataFrame([['HI', 'Hawaii'], ['HI', 'Honolulu'], ['HI', 'Kauai'], ['HI', 'Maui']],
                                       columns=['state', 'county'])
        hawaii_years = pd.DataFrame([['HI', 2020], ['HI', 2021], ['HI', 2022]], columns=['state', 'year'])
        hawaii = reduce(lambda left, right: pd.merge(left, right, how='left', on='state'),
                        [hawaii_avg_temp, hawaii_counties, hawaii_years])

        return pd.concat([historical_weather, hawaii], axis=0)


if __name__ == '__main__':
    INFERENCE = False
    PRODUCT_TYPE = 'hotspringWaterHeater'

    from data_collect import DataCollect

    dataset = DataCollect()
    dataset.combine_data_from_files(inference=False,
                                    devices_path='../big_data_dump/History_devices_energy_data/locations.csv',
                                    devices_hist_folder='../big_data_dump/History_devices_energy_data',
                                    inference_path='testing/prediction_dataset.csv')

    from dataset_processing import DatasetProcessing

    data_proc = DatasetProcessing()
    data = data_proc.preprocess_data(data=dataset.data,
                                     zipcode_scale=True,
                                     product_type=PRODUCT_TYPE,
                                     weather_date='2023-01-17',
                                     inference=INFERENCE)

    feat_eng = FeatureEngineering()
    data = feat_eng.feature_engineering(data=data, inference=INFERENCE, recalc_seasonal=False)
    print(data)
