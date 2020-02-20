# general imports
import os
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
    "--test_size", 
    help="The percentage that should remain in the train samples (0.9 = 0.9 train, 0.1 test)",
    default=0.2
)

parser.add_argument(
    "--seq_len", 
    help="length of input sequence (number of timesteps)",
    default=750
)

parser.add_argument(
    "--window_shift_step_size", 
    help="how much the input window should be shifted (allows for a higher sample count)",
    default=300
)

args = parser.parse_args()

if __name__ == '__main__':

    source_device = str(args.source_device)
    seq_len = int(args.seq_len)
    test_size = float(args.test_size)
    window_shift_step_size = int(args.window_shift_step_size)

    raw_path = os.path.join('data', source_device, 'raw')
    train_path = os.path.join('data', source_device, 'train')
    test_path = os.path.join('data', source_device, 'test')
    # read respiration data
    # check if arguments are valid, i.e. if directory exists
    if not os.path.exists(raw_path):
        logger.error('Source directory: {} is not valid'.format(raw_path))
        raise FileNotFoundError

    # check if arguments are valid, i.e. if directory exists
    if os.path.exists(train_path):
        logger.warning('Train directory: {} already exists, it will be recreated.'.format(train_path))
        shutil.rmtree(train_path)
        os.makedirs(train_path)
    else:
        logger.info('Train directory: {} it will be created.'.format(train_path))
        os.makedirs(train_path)

    # check if arguments are valid, i.e. if directory exists
    if os.path.exists(test_path):
        logger.warning('Test directory: {} already exists, it will be recreated.'.format(test_path))
        shutil.rmtree(test_path)
        os.makedirs(test_path)
    else:
        logger.info('Test directory: {} it will be created.'.format(test_path))
        os.makedirs(test_path)


    file_list = glob.glob("{}/*.pt".format(raw_path))

    tensor_list = []

    for f in file_list:
        logger.debug('Processing file {}'.format(f))
        # load tensor
        train_tensor = torch.load(f)
        # get all values that are not default ones
        start, end = utils.get_id_bounds(train_tensor[:, 0], -3276.8000)
        # subset tensor using default_value_idx and split
        train_tensor_croped = train_tensor[start:end]
        #
        train = train_tensor_croped[1:, 2]
        train_lag = train_tensor_croped[:-1, 2]
        #
        inhale_starts = ((train > 0) & (train_lag == 0) & (train > train_lag)).nonzero()
        #
        for start, end in zip(inhale_starts[:-1], inhale_starts[1:]):
            tensor_list.append(train_tensor_croped[(start + 1):end, :])

    #lstm_input_tensor = torch.cat(tensor_list, dim=0)
    #del tensor_list

    train, test, = train_test_split(tensor_list, test_size=test_size)

    torch.save(train, os.path.join(train_path, 'train_' + source_device + '.pt'))
    torch.save(test, os.path.join(test_path, 'test_' + source_device + '.pt'))
