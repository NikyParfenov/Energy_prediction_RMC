import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from logging_wrappers import timeit_log, trace_log
from loguru import logger
import sys


class DataCollect:
    """
    Module is not for production but for testing pipelines using local files
    """
    def __init__(self):
        self.data = None

    # GET DATA FROM FILES
    @timeit_log
    def combine_data_from_files(self,
                                devices_path: str,
                                devices_hist_folder: str,
                                data_start_from: str = None,
                                ) -> None:
        """
        Call read methods and combine devices with locations

        :param inference:           whether train or inference data
        :param devices_path:        path for location data from 'devices' table
        :param devices_hist_folder: devices data from 'devices_history' table
        :param inference_path:      path to dataset for prediction
        :param data_start_from:     whether drop data before this date

        :return:                    None, save pandas DataFrame into class variable
        """
        locations = self._get_spacial_data_from_file(devices_path)
        self.data = self._get_time_series_data_from_multiple_files(devices_hist_folder,
                                                                   file_pattern='*_history_POWRWATT*')

        if data_start_from is not None:
            self.data.drop(self.data[pd.to_datetime(self.data[['year', 'month', 'day']]) < data_start_from].index,
                           inplace=True)

        self.data = pd.merge(self.data, locations, how='left', on='location_id')
        self.data.drop_duplicates(inplace=True)
        self.data.drop(columns='location_id', inplace=True)

    @trace_log
    def _get_spacial_data_from_file(self, path: str):
        """
        Read a file with geodata (from db table 'devices') and with 'location_id' key
        """
        try:
            df = pd.read_csv(path)
            assert np.isin('location_id', df.columns), "There is no 'location_id' column in the file"
        except FileNotFoundError as error:
            logger.opt(colors=True).error("<lr>File with locations data is not found!</lr>")
            raise error
        return df

    @trace_log
    def _get_time_series_data_from_multiple_files(self,
                                                  folder: str,
                                                  file_pattern: str = '*'):
        """
        Read and combine few *.csv files with time-series devices data
        """
        df = pd.DataFrame()
        for path in tqdm(glob(f'{folder}/{file_pattern}.csv'), desc='Files reading', colour='GREEN', delay=0.01,
                         file=sys.stdout):
            try:
                temp = pd.read_csv(path, dtype={'year': 'int16', 'month': 'int8', 'day': 'int8'})
                assert np.isin('location_id', temp.columns), logger.opt(colors=True).error("<lr>There is no "
                                                                                           "'location_id' column in "
                                                                                           "the file</lr>")
            except FileNotFoundError as error:
                logger.opt(colors=True).error("<lr>File with locations data is not found!</lr>")
                raise error

            df = pd.concat([df, temp], axis=0)
            df.dropna(subset=['location_id'], inplace=True)
        return df


if __name__ == '__main__':
    dataset = DataCollect()
    dataset.combine_data_from_files(devices_path='../../../../../media/nikita/T7/Orion_innovation/big_data_dump/History_devices_energy_data/locations.csv',
                                    devices_hist_folder='../../../../../media/nikita/T7/Orion_innovation/big_data_dump/History_devices_energy_data',
                                    )
    pd.set_option('display.width', 320)
    pd.set_option('display.max_columns', 20)
    print(dataset.data)
