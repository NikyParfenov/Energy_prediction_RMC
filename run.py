import pickle
import pandas as pd
import argparse
from loguru import logger
from time import time
import sys

# FOR RUN USE THE NEXT COMMAND: 'python3 run.py -f <path to .csv file>'
# IF YOU WANT TO WRITE LOGS IN FILE ADD: '-l 1'
# FOR EXAMPLE: 'python3 run.py -f data.csv -l 1'
try:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, type=str)
    parser.add_argument("-l", "--log", default=0, type=int)
    args = vars(parser.parse_args())

    if args['log']:
        logger.add(f"pipeline_run_{time()}.log")

    # Check that energy_model is in the folder and unzipped!
    with open(f'./energy_model.pickle', 'rb') as file:
        energy_model = pickle.load(file)

    # Input DataFrame with product_types: hotspringWaterHeater, heatpumpWaterHeaterGen4, heatpumpWaterHeaterGen5
    # Note: enery_usage is the sum of POWRWATT during the day for specific mac_address
    df = pd.read_csv(args['file'])

    # Call the 'prediction_pipeline' method from pickle object and put DataFrame inside the method
    result = energy_model.prediction_pipeline(prediction_dataset=df, prediction_period=5)

    # Change this to required output
    print(result)
    # result.to_json('prediction.json')
except:
    err = sys.exc_info()
    print(err)
