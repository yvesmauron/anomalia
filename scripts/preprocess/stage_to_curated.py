# general imports
import os
import sys
sys.path.append(os.getcwd())

import json
import argparse
from datetime import datetime
import glob
import shutil
# compression
import pickle as pkl
# data transformations
import torch as torch
import pandas as pd
import numpy as np
# scale transformations
from sklearn.preprocessing import RobustScaler, StandardScaler
# logging
from tqdm import tqdm
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
logger = logging.getLogger('anomalia')

# ------------------------------------------------------------------------
# Parse arguments
parser = argparse.ArgumentParser(
    description='Process csv respiration files from resmed and store them into pickles. NOT Tested, no error handling..:-)'
)
# get arguments
parser.add_argument(
    "--source_dir", 
    help="The folder where the respiration files are saved",
    default='./data/resmed/staging/20200219/'
)
parser.add_argument(
    "--target_dir", 
    help="where you want to store them in pickled format, dir is created if it does no exist",
    default='./data/resmed/curated'
)
parser.add_argument(
    "--config", 
    help="The folder where the respiration files are saved",
    default='./config/resmed.json'
)

args = parser.parse_args()

if __name__ == '__main__':

    # get arguments
    source_dir = str(args.source_dir)
    target_dir = str(args.target_dir)
    data_config = str(args.config)

    # read respiration data
    # check if arguments are valid, i.e. if directory exists
    if not os.path.exists(source_dir):
        logger.error('Source directory: {} is not valid'.format(source_dir))
        raise FileNotFoundError

    # check if arguments are valid, i.e. if directory exists
    if not os.path.exists(data_config):
        logger.error("Config: {} does not exist".format(data_config))
        raise FileNotFoundError

    # read source data
    logger.info('Reading the files from source directory')
    source_dir = str(args.source_dir)
    file_list = glob.glob("{}*.csv".format(source_dir))

    if len(file_list) > 1:
        raw_df = []
        with tqdm(total=len(file_list), bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:100}{r_bar}", ascii=True) as pbar:
            for f in file_list:
                pbar.set_postfix(file=f)
                raw_df.append(pd.read_csv(f))
                pbar.update(1)
                
        raw_df = pd.concat(raw_df, ignore_index=True)
    else:
        raw_df = pd.read_csv(file_list[0])


    # load config sheet
    logger.info('Reading configuration file')
    with open(data_config, 'r') as f:
        config = json.load(f)

    # get columns
    # training columns
    time_columns = [c['name'] for c in config['time_columns']]
    one_hot_columns = [c['name'] for c in config['one_hot_columns']]
    categorical_columns = [c['name'] for c in config['categorical_columns']]
    numerical_columns = [c['name'] for c in config['numerical_columns']]
    train_columns = one_hot_columns + \
        numerical_columns
    # grouping and sorting columns
    grouping_column = [config['grouping_column']]
    event_time_column = [config['event_time_column']]
    sort_columns = grouping_column + event_time_column
    # all relevant columns
    all_columns = set(
        train_columns + 
        one_hot_columns +
        categorical_columns +
        time_columns +
        sort_columns)

    # check if grouping and event time colums exist
    if grouping_column[0] not in all_columns:
        # columns are not correct 
        logger.error("Grouping column specified in {} does not exist in source files.".format(
            data_config)
        )
        raise AttributeError      

    if event_time_column[0] not in all_columns:
        # columns are not correct 
        logger.error("Event time column specified in {} does not exist in source files.".format(
            data_config)
        )
        raise AttributeError      
    # ---------------------------------------------------------------------------
    # create additional columns, to be automated
    # get type from filename
    #logger.info('Extracting station name from file name')
    #raw_df['StationName'] = raw_df['file_name'].apply(lambda x: x.split('_')[2])
    # ---------------------------------------------------------------------------

    # validate columns
    if not set(all_columns).issubset(raw_df.columns):
        # columns are not correct 
        logger.error("Column specified in {} do not exist in source files.".format(
            data_config)
        )
        raise AttributeError

    # subset dataframe columns to relevant ones
    logger.info('Removing unnecessarry columns')
    train_df = raw_df[all_columns]
    del raw_df

    # get dummy variables for the station
    if len(categorical_columns) > 0:
        logger.info('Creating one hot encoded variables from categorical variables.')
        for col in config['categorical_columns']:
            logger.info('One hot encoding column: {}, prefixing with: {}'.format(
                col['name'],
                col['one_hot_prefix']
            ))
            train_df[col['name']] = pd.Categorical(train_df[col['name']])
            dummies = pd.get_dummies(train_df[col['name']], prefix=col['one_hot_prefix'])
            # get dummy variables
            train_df = pd.concat([train_df, dummies], axis=1)
            # add dummy columns to one_hot_columns 
            one_hot_columns = one_hot_columns + list(dummies.columns)

    # convert time column to timestamp
    logger.info('Converting timestamp to date time')
    train_df[config['time_columns'][0]['name']] = pd.to_datetime(
        train_df[config['time_columns'][0]['name']],
        format="%Y-%m-%d %H:%M:%S.%f"
    )

    # process 
    logger.info('Sorting dataset and creating groups')
    train_df = \
        train_df \
        .sort_values(grouping_column + event_time_column, ascending=[1, 1]) \
        .groupby(grouping_column)

    # save df per group -> grouped by file_name
    # easier to work with because of less memory
    if os.path.exists(target_dir):
        logger.warning('Target directory: {} already exists, it will be recreated.'.format(target_dir))
        shutil.rmtree(target_dir)

    os.makedirs(target_dir)

    logger.info('Start writing files')
    with tqdm(total=len(file_list), bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:100}{r_bar}", ascii=True) as pbar:
        for name, group in train_df:
            pbar.set_postfix(file=name)
            train_tensor = torch.Tensor(group[numerical_columns + one_hot_columns].to_numpy())
            torch.save(train_tensor, os.path.join(target_dir, name + '.pt'))
            pbar.update(1)

    logger.info('Successfully written files')

    logger.info('Process terminated')