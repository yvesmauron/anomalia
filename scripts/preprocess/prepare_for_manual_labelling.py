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
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
# logging
import logging
import logging.config
# atemreich imports
from atemteurer import utils 
# dev
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------
# initialize logger
logging.config.fileConfig(
    os.path.join(os.getcwd(), 'config', 'logging.conf')
)

# create logger
logger = logging.getLogger('atemteurer')

# ------------------------------------------------------------------------
# Parse arguments
parser = argparse.ArgumentParser(
    description='create train and test set for lstm'
)
# get arguments
parser.add_argument(
    "--source_device", 
    help="The folder where the respiration files are saved",
    default='resmed'
)

parser.add_argument(
    "--seq_len", 
    help="length of input sequence (number of timesteps)",
    default=750
)

args = parser.parse_args()

if __name__ == '__main__':

    source_device = str(args.source_device)
    seq_len = int(args.seq_len)
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y%m%d")

    raw_path = os.path.join('data', source_device)
    unclassified_path = os.path.join('data', source_device, 'classification', date_time, 'unclassified')
    normal_path = os.path.join('data', source_device, 'classification', date_time, 'normal')
    anomaly_path = os.path.join('data', source_device, 'classification', date_time, 'anomaly')
    # read respiration data
    # check if arguments are valid, i.e. if directory exists
    if not os.path.exists(raw_path):
        logger.error('Source directory: {} is not valid'.format(raw_path))
        raise FileNotFoundError

    # check if arguments are valid, i.e. if directory exists
    if os.path.exists(unclassified_path):
        logger.warning('Unclassified directory: {} already exists, it will be recreated.'.format(unclassified_path))
        shutil.rmtree(unclassified_path)
        os.makedirs(unclassified_path)
    else:
        logger.info('Unclassified directory: {} it will be created.'.format(unclassified_path))
        os.makedirs(unclassified_path)

    # check if arguments are valid, i.e. if directory exists
    if os.path.exists(normal_path):
        logger.warning('Normal directory: {} already exists, it will be recreated.'.format(normal_path))
        shutil.rmtree(normal_path)
        os.makedirs(normal_path)
    else:
        logger.info('Normal directory: {} it will be created.'.format(normal_path))
        os.makedirs(normal_path)

    # check if arguments are valid, i.e. if directory exists
    if os.path.exists(anomaly_path):
        logger.warning('Anomaly directory: {} already exists, it will be recreated.'.format(anomaly_path))
        shutil.rmtree(anomaly_path)
        os.makedirs(anomaly_path)
    else:
        logger.info('Anomaly directory: {} it will be created.'.format(anomaly_path))
        os.makedirs(anomaly_path)

    file_list = glob.glob("{}/*.pt".format(os.path.join(raw_path, 'curated')))

    tensor_list = []

    i = 0
    for f in file_list:
        logger.debug('Processing file {}'.format(f))
        # load tensor
        train_tensor = torch.load(f)
        # get all values that are not default ones
        start, end = utils.get_id_bounds(train_tensor[:, 0], -3276.8000)
        # subset tensor using default_value_idx and split
        train_tensor_croped = train_tensor[start:end]
        # dimensions 
        # [samples, featues]
        dims = train_tensor_croped.shape
        # n sequences
        n_seq = dims[0] // seq_len
        # crop train tensor to match
        train_tensor_croped = train_tensor_croped[:(n_seq * seq_len),:]
        # reshape tensor
        train_tensor_reshaped = train_tensor_croped.view(n_seq, seq_len, dims[1])
        # appent tensor
        train_tensor = train_tensor_reshaped.numpy()

        for idx in range(train_tensor.shape[0]):
            np.savetxt(os.path.join(unclassified_path, str(i) + '.csv'), train_tensor[idx,:,:], delimiter=',', fmt='%4f')
            i += 1

