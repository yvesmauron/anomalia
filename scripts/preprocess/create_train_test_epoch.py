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
# logging
import logging
import logging.config
# atemreich imports
from anomalia import utils 
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
    description='create train and test set for lstm'
)
# get arguments
parser.add_argument(
    "--source_device", 
    help="The folder where the respiration files are saved",
    default='resmed'
)

# get arguments
parser.add_argument(
    "--classifcation_date", 
    help="The folder where the respiration files are saved",
    default=datetime.now().strftime('%Y%m%d')
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

args = parser.parse_args()


def samples_to_tensor_list(dir):
    file_list = glob.glob("{}/*.csv".format(dir))

    tensor_list = []

    for f in file_list:
        logger.debug('Processing file {}'.format(f))
        # load tensor
        train_tensor = torch.FloatTensor(np.genfromtxt(f, delimiter=','))
        # get all values that are not default ones
        tensor_list.append(train_tensor)
    
    return tensor_list


if __name__ == '__main__':

    source_device = str(args.source_device)
    classifcation_date = str(args.classifcation_date)
    seq_len = int(args.seq_len)
    test_size = float(args.test_size)

    normal_path = os.path.join('data', source_device, 'classification', str(classifcation_date), 'normal')
    anomaly_path = os.path.join('data', source_device, 'classification', str(classifcation_date), 'anomaly')

    eval_path = os.path.join('data', source_device, 'eval')
    train_path = os.path.join('data', source_device, 'train')
    test_path = os.path.join('data', source_device, 'test')

    # read respiration data
    # check if arguments are valid, i.e. if directory exists
    if not os.path.exists(normal_path):
        logger.error('Normal directory: {} is not valid'.format(normal_path))
        raise FileNotFoundError

    # check if arguments are valid, i.e. if directory exists
    if not os.path.exists(anomaly_path):
        logger.error('Anomaly directory: {} is not valid'.format(anomaly_path))
        raise FileNotFoundError

    # check if arguments are valid, i.e. if directory exists
    if os.path.exists(eval_path):
        logger.warning('Eval directory: {} already exists, it will be recreated.'.format(eval_path))
        shutil.rmtree(eval_path)
        os.makedirs(eval_path)
    else:
        logger.info('Eval directory: {} it will be created.'.format(eval_path))
        os.makedirs(eval_path)
    
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


    train_test_list = samples_to_tensor_list(normal_path)
    val_list = samples_to_tensor_list(anomaly_path)

    #lstm_input_tensor = torch.cat(tensor_list, dim=0)
    #del tensor_list

    train, test, = train_test_split(train_test_list, test_size=test_size)

    torch.save(train, os.path.join(train_path, 'train_' + source_device + '.pt'))
    torch.save(test, os.path.join(test_path, 'test_' + source_device + '.pt'))
    torch.save(val_list, os.path.join(eval_path, 'eval_' + source_device + '.pt'))