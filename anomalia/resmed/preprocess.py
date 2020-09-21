from .. import utils
import pyarrow.parquet as pq
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
import json
import logging.config
import logging
import numpy as np
import shutil
import glob
import pandas as pd
import sys
import os


# general imports


# logging
# atemreich imports


# ------------------------------------------------------------------------
# initialize logger
logging.config.fileConfig(
    os.path.join(os.getcwd(), 'config', 'logging.conf')
)

# create logger
logger = logging.getLogger('anomalia')


def train_data(
    train_configs="data/normal",
    data_dir="data/resmed/staging/20200914",
    seq_len=750
):

    configs = [p for p in Path(train_configs).iterdir()]

    tensor_list = []

    for config in configs:
        # read configuration file
        with open(config, "r") as f:
            config = json.load(f)
        print(config["data_file"])
        # create parquet table
        df = pq.read_table(
            os.path.join(data_dir, os.path.basename(config["data_file"]))
        ).to_pandas()

        # get normal range from config
        if config["range"]["from"] is not None and config["range"]["to"] is not None:
            start_range = float(config["range"]["from"] * 60)
            end_range = float(config["range"]["to"] * 60)

            # filter data frame
            df = df[(df.time_offset >= start_range)
                    & (df.time_offset <= end_range)]

            # get tensor
            train_tensor = torch.Tensor(df.iloc[:, :3].values)
            # dims
            dims = train_tensor.shape
            # n sequences
            n_seq = dims[0] // seq_len
            # crop train tensor to match
            train_tensor = train_tensor[:(n_seq * seq_len), :]
            # reshape tensor
            train_tensor = train_tensor.view(n_seq, seq_len, dims[1])

            tensor_list.append(train_tensor)

    return torch.cat(tensor_list)


def to_categorical(df, col):
    # just do ignore warning...
    if df[col].dtype.name != 'category':
        df.loc[:, col] = df[col].astype('category')
    df.loc[:, col] = df[col].cat.codes.astype("int16")
    df.loc[:, col] -= df[col].min()
    return df


def preprocess_resmed_from_config(raw_df, data_config, grouped_output=True):

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
        logger.info(
            'Creating one hot encoded variables from categorical variables.')
        for col in config['categorical_columns']:
            logger.info(
                f"Preparing catecorical column: {col['name']} for embedding")
            train_df = to_categorical(train_df, col['name'])

    # convert time column to timestamp
    logger.info('Converting timestamp to date time')
    train_df.loc[:, config['time_columns'][0]['name']] = pd.to_datetime(
        train_df[config['time_columns'][0]['name']],
        format="%Y-%m-%d %H:%M:%S.%f"
    )

    # process
    logger.info('Sorting dataset and creating groups')
    train_df = \
        train_df \
        .sort_values(grouping_column + event_time_column, ascending=[1, 1])

    if grouped_output:
        train_df = \
            train_df \
            .groupby(grouping_column)

    return train_df, numerical_columns, one_hot_columns


def read_data(source_dir, data_config, seq_len=750, scoring=False):

    # -----------------------------------
    # verify folder paths
    if not os.path.exists(source_dir):
        logger.error('Source directory: {} is not valid'.format(source_dir))
        raise FileNotFoundError

    # check if arguments are valid, i.e. if directory exists
    if not os.path.exists(data_config):
        logger.error("Config: {} does not exist".format(data_config))
        raise FileNotFoundError

    # -----------------------------------
    # read source data
    logger.info('Reading the files from source directory')
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

    # -------------------------------------
    # preprocess df
    train_df, numerical_columns, one_hot_columns = preprocess_resmed_from_config(
        raw_df,
        data_config
    )

    # convert to tensor
    tensor_list = []
    train = []

    with tqdm(total=train_df.ngroups, bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:100}{r_bar}", ascii=True) as pbar:
        for name, group in train_df:
            pbar.set_postfix(file=name)
            # train tensor
            train_tensor = torch.Tensor(
                group[numerical_columns + one_hot_columns].to_numpy())
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
            train_tensor_croped = train_tensor_croped[:(n_seq * seq_len), :]
            # reshape tensor
            train_tensor_reshaped = train_tensor_croped.view(
                n_seq, seq_len, dims[1])
            # add to tensor list
            tensor_list.append(train_tensor_reshaped)
            # crop df for scoring
            if scoring:
                croped_df = group.iloc[int(start):int(end), :]
                croped_df = group.iloc[:(n_seq * seq_len), :]
                if train_df.ngroups > 1:
                    train.append(croped_df)
                else:
                    train = croped_df
            pbar.update(1)

    if train_df.ngroups > 1 and scoring:
        train = pd.concat(train, ignore_index=True)

    return tensor_list, train
