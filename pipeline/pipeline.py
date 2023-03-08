"""
Main file that combines pipeline for data collecting, processing, model training and energy prediction
"""
import pandas as pd
from .dataset_processing import DatasetProcessing
from .feature_engineering import FeatureEngineering
from .train import ModelTrain
from loguru import logger
from datetime import timedelta
from time import perf_counter
import pickle


class EnergyPredictionModel(DatasetProcessing, FeatureEngineering, ModelTrain):
    def __init__(self, divisor=1., product_type=None):
        super().__init__()
        self.product_type = product_type
        self.divisor = divisor  # divisor of target value (optional)
        self.model = None
        self.metrics = None
        self.conf_int = None
        self.features = None

    # TRAIN BLOCK
    def train_pipeline(self,
                       input_dataset,
                       zipcode_scale=False,
                       recalc_seasonal=False,
                       save_model=False,
                       ):
        df = self.preprocess_data(data=input_dataset,
                                  zipcode_scale=zipcode_scale,
                                  product_type=self.product_type,
                                  inference=False,
                                  )
        df = self.feature_engineering(data=df,
                                      inference=False,
                                      recalc_seasonal=recalc_seasonal,
                                      )
        self.model, self.conf_int, self.metrics, self.features = self.model_train(df)

        if save_model:
            self.save_model()

    # PREDICT BLOCK
    def prediction_pipeline(self,
                            prediction_dataset: pd.DataFrame,
                            prediction_period: int = 5,
                            ):

        df = self.preprocess_data(data=prediction_dataset,
                                  zipcode_scale=False,
                                  product_type=self.product_type,
                                  prediction_period=prediction_period,
                                  inference=True,
                                  )
        df = self.feature_engineering(data=df,
                                      inference=True,
                                      recalc_seasonal=False,
                                      )

        return self.predict(df, self.model, self.features, self.conf_int)

    # SAVE-LOAD BLOCK
    def save_model(self, path='.'):
        with open(f'{path}/energy_model.pickle', 'wb') as file:
            pickle.dump(self, file)


if __name__ == '__main__':
    INFERENCE = False
    PRODUCT_TYPE = 'hotspringWaterHeater'

    logger.opt(colors=True).info(f'<lc>===== GLOBAL PIPELINE HAS BEEN STARTED =====</lc>')
    start = perf_counter()

    from data_collect import DataCollect

    dataset = DataCollect()
    dataset.combine_data_from_files(devices_path='../big_data_dump/History_devices_energy_data/locations.csv',
                                    devices_hist_folder='../big_data_dump/History_devices_energy_data')

    energy_model = EnergyPredictionModel()
    energy_model.train_pipeline(input_dataset=dataset.data,
                                save_model=True)

    end = perf_counter()
    logger.opt(colors=True).success(f'<lg>PIPELINE HAS BEEN FINISHED SUCCESSFULLY! '
                                    f'TOTAL PROCESSING TIME: {str(timedelta(seconds=end - start))}</lg>')
