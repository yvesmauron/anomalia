# general imports
import os
import sys
sys.path.append(os.getcwd())

from azureml.core import Workspace, Datastore, Dataset
from azureml.data.data_reference import DataReference
import argparse
import json

# ------------------------------------------------------------------------
# initialize logger
import logging
import logging.config

logging.config.fileConfig(
    os.path.join(os.getcwd(), 'config', 'logging.conf')
)

# create logger
logger = logging.getLogger('anomalia')

# ------------------------------------------------------------------------
# Parse arguments
parser = argparse.ArgumentParser(
    description='create train and restset'
)
# get arguments
parser.add_argument(
    "--sp_config", 
    help="Configuration for service principle",
    default="./config/sp_config.json"
)

parser.add_argument(
    "--ws_config", 
    help="Configuration for azure ml workspace",
    default="./config/ws_config.json"
)

parser.add_argument(
    "--ds_config", 
    help="Configuration for azure ml workspace",
    default="./config/ds_config.json"
)

args = parser.parse_args()

if __name__ == "__main__":
    
    # get input parameters
    sp_config_path = str(args.sp_config)
    ws_config_path = str(args.ws_config)
    ds_config_path = str(args.ds_config)

    # basic error handling
    # check if arguments are valid, i.e. if directory exists
    if not os.path.exists(sp_config_path):
        logger.error('Normal directory: {} is not valid'.format(sp_config_path))
        raise FileNotFoundError

    # check if arguments are valid, i.e. if directory exists
    if not os.path.exists(ws_config_path):
        logger.error('Normal directory: {} is not valid'.format(ws_config_path))
        raise FileNotFoundError

    # check if arguments are valid, i.e. if directory exists
    if not os.path.exists(ds_config_path):
        logger.error('Normal directory: {} is not valid'.format(ds_config_path))
        raise FileNotFoundError

    # load config sheet
    with open(sp_config_path, 'r') as f:
        sp_config = json.load(f)

    # load config sheet
    with open(ds_config_path, 'r') as f:
        ds_config = json.load(f)

    # load config sheet
    ws = Workspace.from_config(path=ws_config_path)

    datastore = Datastore.register_azure_data_lake(
        workspace=ws,
        datastore_name=ds_config['data_store_name'],
        store_name=ds_config['data_store_name'], # ADLS Gen2 account name
        tenant_id=sp_config['tenant_id'], # tenant id of service principal
        client_id=sp_config['adls_client_id'], # client id of service principal
        client_secret=sp_config['adls_client_secret']
    ) # the secret of service principal

    # register datasets
    for ds in ds_config['datasets']:

        dataset = Dataset.File.from_files((datastore, ds['path'] + '*.pt'))

        dataset.register(
            workspace=ws,
            name=ds['name'],
            description=ds['description'],
            create_new_version=True
        )
        logger.info('created dataset with name: {} in data store: {}'.format(
            ds['name'], 
            ds_config['data_store_name']
            )
        )
    
    # datasets should now be available in the ml.azure.com studio > datasets

