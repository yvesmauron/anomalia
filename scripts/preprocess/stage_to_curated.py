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
from anomalia.resmed.preprocess import *

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
    default='./data/resmed/staging/BBett_idle/'
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
    train_df, numerical_columns, one_hot_columns = preprocess_resmed_from_config(
        raw_df,
        data_config
    )

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