# -*- coding: utf-8 -*-
import click
import logging
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import shutil
from tqdm import tqdm
import json
from src.features.build_features import (
    process_resmed_train,
    process_resmed_score
)

# Azure keyvault dependencies
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

# Required for Azure Data Lake Storage Gen1 filesystem management
from azure.datalake.store import core, lib

# Set the logging level for all azure-* libraries
azure_logger = logging.getLogger('azure')
azure_logger.setLevel(logging.WARNING)


@click.command()
@click.argument(
    'input_user_data_path',
    type=click.Path(),
    default="exploration/video-analysis/normal"
)
@click.argument(
    'input_data_path',
    type=click.Path(),
    default="exploration/video-analysis/data"
)
@click.argument(
    'output_user_data_path',
    type=click.Path(),
    default="data/external/user_classification/normal"
)
@click.argument(
    'output_data_path',
    type=click.Path(),
    default="data/raw/resmed"
)
def sync_azure_datalake(input_user_data_path, input_data_path,
                        output_user_data_path, output_data_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    logger.info("connecting to azure datalake...")
    # get data from ADLS
    # Get credentials
    credentials = DefaultAzureCredential()

    # Create a secret client
    secret_client = SecretClient(
        os.environ.get("KEY_VAULT_URL"),  # Your KeyVault URL
        credentials
    )

    adlCreds = lib.auth(
        tenant_id=secret_client.get_secret("tenantid").value,
        client_id=secret_client.get_secret("spclientid").value,
        client_secret=secret_client.get_secret("spclientsecret").value,
        resource="https://datalake.azure.net/"
    )

    # Create a filesystem client object
    adlsFileSystemClient = core.AzureDLFileSystem(
        adlCreds, store_name=os.environ.get("DATALAKE_NAME"))

    user_data_path = Path(output_user_data_path)
    data_path = Path(output_data_path)

    # remove directories if they exist
    if user_data_path.exists():
        shutil.rmtree(user_data_path)

    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)

    user_data_path.mkdir(parents=True, exist_ok=True)

    user_classifications = adlsFileSystemClient.ls(input_user_data_path)
    train_data_paths = []

    logger.info("downloading user classification data...")
    with tqdm(
        total=len(user_classifications),
        bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:100}{r_bar}",
        ascii=True
    ) as pbar:
        for file_path in user_classifications:
            pbar.set_postfix(file=str(os.path.basename(file_path)))
            dst_path = os.path.join(
                user_data_path, os.path.basename(file_path)
            )
            adlsFileSystemClient.get(
                file_path,
                dst_path
            )

            # check if downloaded config is valid for use training
            with open(dst_path, "r") as f:
                config = json.load(f)

            if config["range"]["from"] is not None \
                    and config["range"]["to"] is not None:
                train_data_paths.append(os.path.basename(config["data_file"]))
            else:
                Path(dst_path).unlink()

            pbar.update(1)

    # get all data paths
    data_paths = adlsFileSystemClient.ls(input_data_path)
    existing_data_paths = [
        os.path.basename(p)
        for p in Path(os.path.join(output_data_path)).iterdir()
    ]

    logger.info("downloading sensor data if not exists...")
    with tqdm(
        total=len(data_paths),
        bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:100}{r_bar}",
        ascii=True
    ) as pbar:
        for file_path in data_paths:
            # set posix for info
            pbar.set_postfix(file=str(os.path.basename(file_path)))

            # only download data that we didn't download yet
            if os.path.basename(file_path) not in existing_data_paths:
                adlsFileSystemClient.get(
                    file_path,
                    os.path.join(data_path, os.path.basename(file_path))
                )

            pbar.update(1)

    logger.info("processing training dataset")
    process_resmed_train(
        user_data_path=user_data_path,
        data_path=data_path
    )

    logger.info("processing scoring dataset")
    process_resmed_score(
        data_path=data_path
    )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    ##
    sync_azure_datalake()
