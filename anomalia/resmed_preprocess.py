# general imports
import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
# logging
import logging
import logging.config
# atemreich imports
import json

# ------------------------------------------------------------------------
# initialize logger
logging.config.fileConfig(
    os.path.join(os.getcwd(), 'config', 'logging.conf')
)

# create logger
logger = logging.getLogger('anomalia')

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
        logger.info('Creating one hot encoded variables from categorical variables.')
        for col in config['categorical_columns']:
            logger.info('One hot encoding column: {}, prefixing with: {}'.format(
                col['name'],
                col['one_hot_prefix']
            ))
            train_df.loc[: ,col['name']] = pd.Categorical(train_df[col['name']])
            dummies = pd.get_dummies(train_df[col['name']], prefix=col['one_hot_prefix'])
            # get dummy variables
            train_df = pd.concat([train_df, dummies], axis=1)
            # add dummy columns to one_hot_columns 
            one_hot_columns = one_hot_columns + list(dummies.columns)

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