# Energy_prediction_RMC
Pipeline for daily energy prediction of devices (only code, without data and model)  
To run prediction use run.py: `python3 run.py -f <input_data.csv> -l/--log` (-l/--log is optional to print logs in file)  
  
Each block of pipeline is a class-mixin with appropriate method. Final class is in pipeline.py.  
**In ./pipeline folder:**
- data_collect.py - for manual combining data from disk
- dataset_processing.py - dataset correction and cleaning
- feature_engineering.py - creates new features
- train.py - file with model
- pipeline.py - full pipeline for train and predict
