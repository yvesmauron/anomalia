# general imports
import os
import sys
import argparse
# add working directory to python path for later imports
sys.path.append(os.getcwd())
# deep learning libraries
import torch
# project package dependencies
from anomalia.smavra import SMAVRA
from anomalia.datasets import ResmedDatasetEpoch, TestDataset
from anomalia.logging.logging_azurml import AzureLogger
from anomalia.logging.logging_mlflow import MLFlowLogger
from anomalia.smavra_trainer import SmavraTrainer
# azrue dependencies
from azureml.core import Workspace, Dataset


# ------------------------------------------------------------------------
# Parse arguments
parser = argparse.ArgumentParser(
    description='create train and test set for lstm'
)

# get arguments
parser.add_argument(
    "--ws_config", 
    help="Configuration for azure ml workspace",
    default="./config/ws_config.json"
)

# get arguments
parser.add_argument(
    "--output_dir", 
    help="output dir to log",
    default='./outputs'
)

# get arguments
parser.add_argument(
    "--ds_name", 
    help="Configuration for azure ml workspace",
    default="resmed_train"
)

# get arguments
parser.add_argument(
    "--batch_size", 
    help="The batch size that should be used to train the model",
    default=64
)

# get arguments
parser.add_argument(
    "--n_epochs", 
    help="The number of epochs to train the neural network",
    default=300
)

parser.add_argument(
    "--compute_node", 
    help="The file path that holds the environment file",
    default='local'
)

args = parser.parse_args()

# --------------------------------------------------------
# set globals
TEST_MODE = False
EXPERIMENT_NAME = 'SMAVRA'
USE_CUDA = True

if __name__ == "__main__":

    ws_config_path = str(args.ws_config)
    batch_size = int(args.batch_size)
    n_epochs = int(args.n_epochs)
    ds_name = str(args.ds_name)
    output_dir = str(args.output_dir)
    compute_node = str(args.compute_node)

    # --------------------------------------------------------
    # init workspace
    if not os.path.exists(ws_config_path):
        raise FileNotFoundError

    ws = Workspace.from_config(path=ws_config_path)

    train_files = Dataset.get_by_name(ws, name=ds_name)
    train_paths = train_files.download(target_path='.', overwrite=True)

    # --------------------------------------------------------
    # define learner
    smarva_input_params = {
        'input_size':1 if TEST_MODE else 3,
        'hidden_size':10 if TEST_MODE else 30,
        'latent_size':1 if TEST_MODE else 3,
        #'attention_size':1 if TEST_MODE else 3, # not supported anymore
        'output_size':1 if TEST_MODE else 3,
        'num_layers':1 if TEST_MODE else 2,
        'n_heads':1 if TEST_MODE else 3,
        'dropout':0.25,
        'batch_first':True,
        'cuda': USE_CUDA,
        'mode':'static',
        'rnn_type':'LSTM',
        'use_variational_attention':False
    }

    smavra = SMAVRA(**smarva_input_params)

    if USE_CUDA:
        smavra.cuda()

    # --------------------------------------------------------
    # define dataset

    if not TEST_MODE:
        dataset = ResmedDatasetEpoch(train_paths[0], batch_size)
    else:
        dataset = TestDataset(200, 200)

    # --------------------------------------------------------
    # define logger
    if compute_node is 'local':
        logger = MLFlowLogger(EXPERIMENT_NAME, "model", './environment.yml', ['./anomalia'])
    else:
        logger = AzureLogger(ws, EXPERIMENT_NAME, output_dir)
    

    # --------------------------------------------------------
    # define optimizer
    lr=0.0005
    optimizer = torch.optim.Adam(smavra.parameters(), lr=lr)

    # --------------------------------------------------------
    # let the trainer take care of training
    trainer = SmavraTrainer(model=smavra, dataset=dataset, optimizer=optimizer, logger=logger)

    # --------------------------------------------------------
    # Start run including logging
    logger.start_run()
    logger.log('lr', lr) # think about a better way to do this
    trainer.fit(n_epochs=n_epochs, batch_size=batch_size)
    # start run
    logger.end_run()