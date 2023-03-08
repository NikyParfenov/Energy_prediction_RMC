import numpy as np
import pandas as pd
from .logging_wrappers import timeit_log, trace_log
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from datetime import timedelta
from tqdm import tqdm
import sys


class ModelTrain:
    def __init__(self):
        super().__init__()
        self.subtract_trend = False
        self.test_period = 61


    @timeit_log
    def model_train(self,
                    data: pd.DataFrame) -> tuple[dict[str, tuple[MinMaxScaler, LinearRegression]], dict[str, float], dict[str, float], list[str]]:
        """
        Runs training for each product type
        """
        features = ['year', 'month', 'day', 'macs_amount', 'weekday',
                    'average_LOHTRTMP', 'average_UPHTRTMP', 'disperse_LOHTRTMP',
                    'disperse_UPHTRTMP', 'lat', 'lng', 'population_perc', 'seasonal',
                    'trend', 'avg_temp', 'avg_prec', 'avg_hdd', 'avg_cdd', 'timestamp',
                    'energy_usage_lag1', 'energy_usage_lag2']

        model = {}
        r2score = {}
        conf_int = {}
        for product in data.product_type.unique():
            train_data, test_data, scaler = self._train_test_data_linear(data.loc[data.product_type == product, :],
                                                                         features)
            lr, score, stdev = self._run_training_linear(train_data, test_data, product)
            model[product] = (scaler, lr)
            r2score[product] = score
            conf_int[product] = 1.96 * stdev

        return model, conf_int, r2score, features

    @timeit_log
    def predict(self,
                df: pd.DataFrame,
                model: dict[str, tuple[MinMaxScaler, LinearRegression]],
                features: list[str],
                conf_int: dict[str, float]) -> pd.DataFrame:
        """
        Method makes forecast relying on input data frame
        """
        start_date = str((df.dropna(subset=['energy_usage']).datetime.max() + timedelta(days=1)).date())
        for mac in tqdm(df.mac_address.unique(), desc='Macs_predicted', colour='GREEN', file=sys.stdout):
            for date in df.loc[(df['mac_address'] == mac) & (df['datetime'] >= start_date), :].index:
                if len(df.loc[(df['mac_address'] == mac) & (df['datetime'] < start_date), :].index) > 1:
                    df.loc[(df['mac_address'] == mac) & (df.index == date), 'energy_usage_lag1'] = df.loc[(df['mac_address'] == mac) & (df.index == (date - 1)), 'energy_usage'].values[0]
                    df.loc[(df['mac_address'] == mac) & (df.index == date), 'energy_usage_lag2'] = df.loc[(df['mac_address'] == mac) & (df.index == (date - 2)), 'energy_usage'].values[0]
                    x_dataset = df.loc[(df['mac_address'] == mac) & (df.index == date), features].fillna(0)
                    product = df.loc[(df['mac_address'] == mac), 'product_type'].unique()[0]
                    x_dataset = model[product][0].transform(x_dataset)
                    df.loc[(df['mac_address'] == mac) & (df.index == date), 'energy_usage'] = model[product][1].predict(x_dataset)[0]

        # if some values are less than zero
        df.loc[df['energy_usage'] < 0, 'energy_usage'] = 0

        df['conf_int'] = df.datetime.apply(lambda x: max(0, (x - pd.to_datetime(start_date)).days + 1))
        # Manual estimation of error evolution, based on predicted data
        df['conf_int'] = df['conf_int'].astype('bool').astype('int') * (
                    0.01662 * df['conf_int'] ** 2 - 0.01721 * df['conf_int'] + 1.0)
        for device in df.product_type.unique():
            df.loc[df['product_type'] == device, 'conf_int'] *= conf_int[device]

        return df.loc[df['datetime'] >= start_date, ['mac_address',
                                                     'product_type',
                                                     'datetime',
                                                     'energy_usage',
                                                     'conf_int']]

    @trace_log
    def _train_test_data_linear(self,
                                df: pd.DataFrame,
                                features: list[str]) -> tuple[tuple[np.ndarray, pd.Series], tuple[np.ndarray, pd.Series], MinMaxScaler]:
        """
        Mrthod splits the data on train and test part relying on test_period class-variable
        """
        split_date = np.sort(df['datetime'].unique())[-self.test_period]
        test_dataset = df.loc[df['datetime'] >= split_date, :].dropna().reset_index(drop=True)
        train_dataset = df.loc[df['datetime'] < split_date, :].dropna().reset_index(drop=True)

        x_test = test_dataset[features]
        x_train = train_dataset[features]
        y_test = test_dataset['energy_usage']
        y_train = train_dataset['energy_usage']

        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        return (x_train, y_train), (x_test, y_test), scaler

    @trace_log
    def _run_training_linear(self,
                             train_data: tuple[np.ndarray, pd.Series],
                             test_data: tuple[np.ndarray, pd.Series],
                             product: str) -> tuple[LinearRegression, float, float]:
        """
        Method runs training procedure
        """
        x_train, y_train = train_data
        x_test, y_test = test_data
        lr = LinearRegression()
        lr.fit(x_train, y_train)

        r2score = lr.score(x_train, y_train)
        stdev = mean_squared_error(y_train, lr.predict(x_train), squared=False)
        print(f'{product} model estimation:')
        print(f'R2_train: {r2score:.3f}, '
              f'RMSE_train: {stdev:.2f}\n'
              f'R2_test: {lr.score(x_test, y_test):.3f}, '
              f'RMSE_train: {mean_squared_error(y_test, lr.predict(x_test), squared=False):.2f}')
        return lr, r2score, stdev
