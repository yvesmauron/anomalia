import json
import argparse
import torch
from anomalia.smavra import SMAVRA
from anomalia.datasets import ResmedDatasetEpoch, TestDataset
from anomalia.logging.logging_azurml import AzureLogger
from anomalia.logging.logging_mlflow import MLFlowLogger
from anomalia.smavra_trainer import SmavraTrainer
from azureml.core import Workspace, Dataset
import os
import sys
# add working directory to python path for later imports
sys.path.append(os.getcwd())
# general imports

# deep learning libraries
# project package dependencies
# azrue dependencies


# ------------------------------------------------------------------------
# Parse arguments
parser = argparse.ArgumentParser(
    description='Train SMAVRA'
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
    "--source_dir",
    help="The batch size that should be used to train the model",
    default="data/resmed/staging/BBett_idle/"
)

# get arguments
parser.add_argument(
    "--data_config",
    help="The batch size that should be used to train the model",
    default="config/resmed.json"
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
    default=1000
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
    source_dir = str(args.source_dir)
    data_config = str(args.data_config)
    n_epochs = int(args.n_epochs)
    ds_name = str(args.ds_name)
    output_dir = str(args.output_dir)
    compute_node = str(args.compute_node)

    # --------------------------------------------------------
    # init workspace
    if not os.path.exists(ws_config_path):
        raise FileNotFoundError

    if compute_node == 'local':
        train_paths = ['data/resmed/train/train_resmed.pt']
    else:
        ws = Workspace.from_config(path=ws_config_path)
        train_files = Dataset.get_by_name(ws, name=ds_name)
        train_paths = train_files.download(target_path='.', overwrite=True)

    # --------------------------------------------------------
    # define learner
    smarva_input_params = {
        'input_size': 1 if TEST_MODE else 3,
        'hidden_size': 10 if TEST_MODE else 30,
        'latent_size': 1 if TEST_MODE else 15,
        # 'attention_size':1 if TEST_MODE else 3, # not supported anymore
        'output_size': 1 if TEST_MODE else 3,
        'num_layers': 1 if TEST_MODE else 2,
        'n_heads': 1 if TEST_MODE else 3,
        'dropout': 0.25,
        'batch_first': True,
        'cuda': USE_CUDA,
        'mode': 'static',
        'rnn_type': 'LSTM',
        'use_variational_attention': True,
        'use_proba_output': True
    }

    smavra = SMAVRA(**smarva_input_params)
    smavra.float()

    if USE_CUDA:
        smavra.cuda()

    # --------------------------------------------------------
    # define dataset

    if not TEST_MODE:
        dataset = ResmedDatasetEpoch(
            device='cuda' if USE_CUDA else 'cpu',
            batch_size=batch_size
        )
    else:
        dataset = TestDataset(200, 200)

    # --------------------------------------------------------
    # define logger
    if compute_node is 'local':
        logger = MLFlowLogger(EXPERIMENT_NAME, "model",
                              './environment.yml', ['./anomalia'])
    else:
        logger = AzureLogger(ws, EXPERIMENT_NAME, output_dir)

    # --------------------------------------------------------
    # log input values for lstm
    logger.start_run()
    for key, value in smarva_input_params.items():
        logger.log(key, str(value))

    # --------------------------------------------------------
    # define optimizer
    lr = 0.0005
    optimizer = torch.optim.Adam(smavra.parameters(), lr=lr)

    # --------------------------------------------------------
    # let the trainer take care of training
    trainer = SmavraTrainer(
        model=smavra,
        dataset=dataset,
        optimizer=optimizer,
        logger=logger,
        checkpoint_interval=100
    )

    # --------------------------------------------------------
    # Start run including logging

    logger.log('lr', lr)  # think about a better way to do this
    trainer.fit(
        n_epochs=n_epochs,
        batch_size=batch_size,
        clip=True,
        max_grad_norm=5,
        kld_annealing_start_epoch=0,
        kld_annealing_max=0.6,
        kld_annealing_intervals=[15, 25, 5],
        kld_latent_loss_weight=.6,
        kld_attention_loss_weight=.01
    )

    # log important artifacsts, that are used for inference
    with open("config/data_config.json", "w") as f:
        json.dump(dataset.get_train_config(), f, indent=4)

    logger.log_artifact("config/data_config.json", "config")
    logger.log_artifact("config/resmed.json", "config")

    logger.end_run()
