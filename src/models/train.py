# utils
import click
import json
from dotenv import find_dotenv, load_dotenv
import logging
from pathlib import Path

# dl
import torch

# azure
# from azureml.core import Workspace, Dataset

# model specific
from src.models.anomalia.smavra import SMAVRA
from src.models.anomalia.datasets import ResmedDatasetEpoch, TestDataset
# from src.models.tracking.azurml import AzureTracker
from src.models.tracking.mlflow import MLFlowTracker
from src.models.anomalia.smavra_trainer import SmavraTrainer

# Set the logging level for all azure-* libraries
azure_logger = logging.getLogger('azure')
azure_logger.setLevel(logging.WARNING)

# --------------------------------------------------------
# set globals
TEST_MODE = False
EXPERIMENT_NAME = 'SMAVRA'
USE_CUDA = True


@click.command()
@click.option(
    '--batch_size',
    type=click.INT,
    default=64,
    help="Batch size to be used."
)
@click.option(
    '--train_data',
    type=click.Path(),
    default="data/processed/resmed/train/train.pt",
    help="Path to training data."
)
@click.option(
    '--n_epochs',
    type=click.INT,
    default=2265,
    help="Numbers of epochs the model should be trained."
)
@click.option(
    '--ds_name',
    type=click.Path(),
    default="resmed_dataset",
    help="Dataset name in Azure Machine Leaning Services, if used."
)
@click.option(
    '--output_dir',
    type=click.Path(),
    default="./outputs",
    help="Output directory to be used in Azure ML Services."
)
@click.option(
    '--compute_node',
    type=click.Path(),
    default="local",
    help="Where it should be computed; not supported at the moment."
)
def train_smavra(
    batch_size: int,
    train_data: str,
    n_epochs: int,
    ds_name: str,
    output_dir: str,
    compute_node: str
):
    """Train smavra model

    Args:
        batch_size (int): Batch size to be used.
        train_data (str): Path to training data.
        n_epochs (int): Numbers of epochs the model should be trained.
        ds_name (str): Dataset name in Azure Machine Leaning Services.
        output_dir (str): Output directory to be used in Azure ML Services.
        compute_node (str): Where it should be computed; not supported.
    """
    # --------------------------------------------------------
    # init workspace
    # if not os.path.exists(os.environ.get("ML_WORKSPACE_CONFIG")):
    #     raise FileNotFoundError

    # if compute_node == 'local':
    #     train_paths = ['data/resmed/train/train_resmed.pt']
    # else:
    #     ws = Workspace.from_config(
    #       path=os.environ.get("ML_WORKSPACE_CONFIG")
    #     )
    #     train_files = Dataset.get_by_name(ws, name=ds_name)
    #     train_paths = train_files.download(target_path='.', overwrite=True)

    # --------------------------------------------------------
    # define learner
    smarva_input_params = {
        'input_size': 1 if TEST_MODE else 3,
        'hidden_size': 10 if TEST_MODE else 64,
        'latent_size': 1 if TEST_MODE else 16,
        'attention_size': 1 if TEST_MODE else 2,
        'output_size': 1 if TEST_MODE else 3,
        'num_layers': 1 if TEST_MODE else 2,
        'n_heads': 1 if TEST_MODE else 1,
        'dropout': 0.5,
        'batch_first': True,
        'cuda': USE_CUDA,
        'mode': 'static',
        'rnn_type': 'LSTM',
        'seq_len': 750,
        'use_epoch_latent': True,
        'use_variational_attention': True,
        'use_proba_output': False
    }

    smavra = SMAVRA(**smarva_input_params)

    smavra.float()

    if USE_CUDA:
        smavra.cuda()

    # --------------------------------------------------------
    # define dataset
    if not TEST_MODE:
        dataset = ResmedDatasetEpoch(
            data=train_data,
            device='cuda' if USE_CUDA else 'cpu',
            batch_size=batch_size
        )
    else:
        dataset = TestDataset(200, 200)

    # --------------------------------------------------------
    # define logger
    if compute_node == 'local':
        logger = MLFlowTracker(EXPERIMENT_NAME, "model",
                               './environment.yml', ['./src/models'])
    # else:
    #     logger = AzureTracker(ws, EXPERIMENT_NAME, output_dir)

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
        kld_annealing_max=0.5,
        kld_annealing_intervals=[15, 25, 5],
        kld_latent_loss_weight=.0025,
        kld_attention_loss_weight=.03
    )

    # log important artifacsts, that are used for inference
    with open("config/preprocessing_config.json", "w") as f:
        json.dump(dataset.get_train_config(), f, indent=4)

    logger.log_artifact("config/preprocessing_config.json", "config")

    logger.end_run()


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    train_smavra()
