# general imports
import os
import json
import argparse
from datetime import datetime
# compression
import pickle as pkl
import joblib # seems to work better with objects
# data transformations
import torch as torch
import pandas as pd
import numpy as np
# scale transformations
from sklearn.preprocessing import RobustScaler, StandardScaler
# logging
import logging
import logging.config
# dev
import matplotlib.pyplot as plt



# ------------------------------------------------------------------------
# initialize logger
logging.config.fileConfig(
    os.path.join(os.getcwd(), 'config', 'logging.conf')
)

# create logger
logger = logging.getLogger('atemteurer')

# Parse arguments
parser = argparse.ArgumentParser(
    description='Process csv respiration files from resmed and store them into pickles. NOT Tested, no error handling..:-)'
)
# get arguments
parser.add_argument(
    "--source_dir", 
    help="The folder where the respiration files are saved",
    default='./data/resmed/raw'
)

# get arguments
parser.add_argument(
    "--config", 
    help="The folder where the respiration files are saved",
    default='./config/resmed.json'
)

parser.add_argument(
    "--seq_len", 
    help="The folder where the respiration files are saved",
    default=750
)

# get arguments
parser.add_argument(
    "--preprocess_type", 
    help="The folder where the respiration files are saved",
    default='RobustScaler'
)

# get arguments
parser.add_argument(
    "--fit_new_scaler", 
    help="The folder where the respiration files are saved",
    default='0'
)


args = parser.parse_args()


if __name__ == '__main__':

    # get arguments
    source_dir = str(args.source_dir)
    transform_config = str(args.transform_config)
    seq_len = int(args.seq_len)
    preprocess_type = str(args.preprocess_type)
    fit_new_scaler = int(args.fit_new_scaler)

    # check if arguments are valid, i.e. if directory exists
    if not os.path.exists(source_dir):
        logger.error("Source_dir: {} is not a valid directory or does not exist".format(source_dir))
        raise FileNotFoundError

    # check if arguments are valid, i.e. if directory exists
    if not os.path.exists(transform_config):
        logger.error("Transform_config: {} does not exist".format(transform_config))
        raise FileNotFoundError

    # load config sheet
    with open(transform_config, 'r') as f:
        config = json.load(f)

    # list files in that directory
    files = os.listdir(source_dir)

    # debug
    f = files[0]
    data_subset = pd.read_pickle(os.path.join(source_dir, f))

    # get columns
    time_columns = [c['name'] for c in config['time_columns']]
    one_hot_columns = [c['name'] for c in config['categorical_columns']]
    categorical_columns = [c['name'] for c in config['one_hot_columns']]
    numerical_columns = [c['name'] for c in config['numerical_columns']]

    # validate columns
    if not set(time_columns + 

            one_hot_columns + 
            categorical_columns + 
            numerical_columns
        ).issubset(data_subset.columns):
        # columns are not correct 
        logger.error("Column specified in {} do not exist in data.frame laocated {}.".format(
            transform_config,
            f)
        )
        raise AttributeError

    # train transformer or load existing ones
    if 'transformer' not in config.keys() or fit_new_scaler == 1:
        # tbd. support multiple scaling options
        if preprocess_type == 'RobustScaler':
            transformer = RobustScaler()
        else:
            transformer = StandardScaler()
        
        # fit transfomer
        transformer = transformer.fit(data_subset[numerical_columns])

        # check if scaler object file already exists.
        file_name = os.path.join('models', 'preprocessing', preprocess_type + '.sav')
        # check if folder exists
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        # check if file exists
        elif os.path.exists(file_name):
            logger.warning('File {} already exists, it will be recreated.'.format(file_name))
            os.unlink(file_name)

        # dump new file
        joblib.dump(transformer, file_name)

        # update config
        config['transformer'] = file_name

        # backup old config
        os.rename(
            transform_config, 
            os.path.join(os.path.dirname(transform_config), datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.json')
        )

        # write down new config
        with open(transform_config, 'w') as f:
            f.write(json.dumps(config, indent=2))
    
    else:
        # todo catch wrong scaler object...
        transformer = joblib.load(config['transformer'])
    
    # scale variables
    data_subset[numerical_columns] = transformer.transform(data_subset[numerical_columns])

    # define input dimensions
    batch_size = data_subset.shape[0] - seq_len

    # seq_len already defined
    dataset = data_subset[numerical_columns + one_hot_columns]

    np.array(dataset)

    output_vars = [1]
    seq_len = 3
    input_size = 2
    window_shift = 1
    batch_size = dataset.shape[0]- seq_len

    train = None
    test = None

    #for i in range(1, batch_size):

    if train is None:
        train = data_subset.iloc[0:750, 3:].to_numpy()
    else:
        train = np.dstack((train, data_subset.iloc[0:750, 3:].to_numpy()))


    np.transpose(test)
    
    np.array(data_subset.iloc[750, 2:])

    
