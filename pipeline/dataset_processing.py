import pandas as pd
import numpy as np
from .logging_wrappers import timeit_log, trace_log
from sklearn.preprocessing import LabelEncoder
import datetime
import pickle


class DatasetProcessing:
    def __init__(self):
        super().__init__()
        self.zipcode_replace = None

    # DATA PROCESSING
    @timeit_log
    def preprocess_data(self,
                        data: pd.DataFrame,
                        product_type: str = None,
                        inference: bool = True,
                        prediction_period: int = 5,
                        zipcode_scale: bool = False,
                        ) -> pd.DataFrame:
        """
        Fill missed country values, rename few mistake values, create prediction period for inference data.

        :param data:              input data frame from DB contains time, spacial and energy usage features
        :param product_type:      if not None filter data frame by specific device
        :param inference:         bool var concerning whether inference or train df used
        :param prediction_period: prediction period in days for inference stage
        :param zipcode_scale:     whether group data frame to zip scale of leave at mac_address level

        :return:                  data frame with cleaned data
        """
        # Check all requirement columns in data frame
        assert np.isin(data.columns, ['mac_address',
                                      'product_type',
                                      'year',
                                      'month',
                                      'day',
                                      'energy_usage',
                                      'country',
                                      'state',
                                      'city',
                                      'zipcode']).all(), 'Dataset should be a dataframe with next columns: ' \
                                                         '[mac_address, product_type, year, month, day, energy_usage,' \
                                                         ' country, state, city, zipcode]'
        data['datetime'] = pd.to_datetime(data[['year', 'month', 'day']])
        if product_type is not None:
            data = data.loc[data['product_type'] == product_type, :]
        if inference:
            data = self._create_prediction_period(data, period=prediction_period)
        else:
            self._correct_time_periods(data)
            data.dropna(subset=['energy_usage'], inplace=True)
        data = self._correct_db_data(data)
        data = self._correct_zipcodes(data, path='additional_files/zipcode_replace.pickle')
        data = pd.merge(data,
                        data.groupby(['city', 'zipcode', 'datetime'])['mac_address'].count().reset_index().rename(
                            columns={'mac_address': 'macs_amount'}),
                        how='left',
                        on=['city', 'zipcode', 'datetime'])

        # Aggregating mac_addresses to zipcode level
        if zipcode_scale and not inference:
            data = data.groupby(['product_type',
                                 'state',
                                 'city',
                                 'zipcode',
                                 'macs_amount',
                                 'year',
                                 'month',
                                 'day',
                                 'datetime']).agg(energy_usage=('energy_usage', np.mean)).reset_index()
            le = LabelEncoder()

            # Fictive mac_address! In order to not change further script (could be optimized in future)
            data['mac_address'] = le.fit_transform(data['zipcode'].astype('str'))

        return data

    @trace_log
    def _correct_db_data(self,
                         data: pd.DataFrame) -> pd.DataFrame:
        """
        Method corrects location DB data
        """
        data = pd.merge(data,
                        data.loc[data['country'].isna(), ['country', 'state', 'city']].drop_duplicates().rename(
                        columns={'country': 'country_new'}),
                        how='left', on=['state', 'city'])
        data['country'] = data['country'].fillna(value=data['country_new'])
        data.drop(columns='country_new', inplace=True)
        data.loc[data['city'] == 'Селинсгроув', 'city'] = 'Selinsgrove'
        data.loc[(data['city'] == 'SF') & (data['state'] == 'CA'), 'city'] = 'San Francisco'
        data['city'] = data['city'].apply(lambda x: str(x).title())
        data.loc[(data['city'] == 'False') | (data['city'] == 'Nan'), 'city'] = np.nan
        data.loc[data['state'] == 'Kentucky', 'state'] = 'KY'
        data = data.loc[data['country'] == 'US', :]
        data.drop(data.loc[data['state'] == 'false', :].index, inplace=True)
        data.drop(columns='country', inplace=True)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=['state'], inplace=True)
        data.reset_index(drop=True, inplace=True)

        return data

    @trace_log
    def _correct_time_periods(self,
                              data: pd.DataFrame) -> pd.DataFrame:
        """
        Method clear damaged time intervals (low data, server problem periods etc.)
        """
        corrupted_intervals = [('heatpumpWaterHeaterGen4', '2020-11-23', '2020-12-03'),
                               ('heatpumpWaterHeaterGen4', '2021-05-15', '2021-06-04'),
                               ('hotspringWaterHeater', '2020-11-23', '2020-12-03'),
                               ('hotspringWaterHeater', '2021-05-15', '2021-06-04'),
                               ('heatpumpWaterHeaterGen5', '2020-11-23', '2020-12-03'),
                               ]
        data.drop(data[data['datetime'] <= '2020-07-03'].index, inplace=True)
        for item in corrupted_intervals:
            data.loc[(data['product_type'] == item[0]) & (data['datetime'] >= item[1]) & (data['datetime'] <= item[2]),
            'energy_usage'] = np.nan

        return data

    @trace_log
    def _create_prediction_period(self,
                                  data: pd.DataFrame,
                                  period: int = 7,
                                  start_date: str = None) -> pd.DataFrame:
        """
        Method creates prediction interval for inference
        """
        if start_date is None:
            start_date = data.datetime.dt.date.max()
        # prediction period shouldn't be less than 1 day
        if period < 1:
            period = 1
        date_period = pd.date_range(start=start_date + datetime.timedelta(days=1),
                                    end=start_date + datetime.timedelta(days=period),
                                    freq='D')
        df = data.copy()
        for date in date_period:
            gen_feats = data.drop(columns='energy_usage').drop_duplicates(
                ['mac_address', 'zipcode', 'city', 'state', 'country', 'product_type'])
            gen_feats['datetime'] = date
            df = pd.concat([df, gen_feats], axis=0)
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day

        return df

    @trace_log
    def _correct_zipcodes(self,
                          data: pd.DataFrame,
                          path: str) -> pd.DataFrame:
        """
        Method corrects weird zipcodes using ext. prepared file
        """
        if self.zipcode_replace is None:
            with open(path, 'rb') as file:
                self.zipcode_replace = pickle.load(file)
        for key, values in self.zipcode_replace.items():
            for item in values:
                if key == 'part_1':
                    data.loc[data['zipcode'] == item[0], 'zipcode'] = item[1]
                elif key == 'part_2':
                    data.loc[(data['city'] == item[0]) & (data['state'] == item[1]), 'zipcode'] = item[2]
                else:
                    data.loc[(data['city'] == item[0]) & (data['state'] == item[1]) & (data['zipcode'] == item[2]), 'zipcode'] = item[3]

        # To avoid the difference between 012345 and '012345' zipcodes
        data['zipcode'] = data['zipcode'].astype('int', errors='ignore')

        return data


if __name__ == '__main__':
    INFERENCE = False

    from data_collect import DataCollect
    dataset = DataCollect()
    dataset.combine_data_from_files(inference=INFERENCE,
                                    devices_path='../big_data_dump/History_devices_energy_data/locations.csv',
                                    devices_hist_folder='../big_data_dump/History_devices_energy_data',
                                    inference_path='testing/prediction_dataset.csv')

    dataset_processed = DatasetProcessing()
    result = dataset_processed.preprocess_data(dataset.data, inference=INFERENCE)

    pd.set_option('display.width', 320)
    pd.set_option('display.max_columns', 20)
    print(result)
