import click
import logging
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import tempfile
import os
import sys
from src.data.utils import edf, azure as azure_utils
import re

sys.path.append(os.getcwd())

# ------------------------------------------------------------------------
azure_logger = logging.getLogger('azure')
azure_logger.setLevel(logging.WARNING)


@click.command()
@click.argument(
    '--input_path',
    type=click.Path(),
    default="raw/manual-extraction/resmed/bianca/2020-12-15/DATALOG"
)
@click.argument(
    '--output_path',
    type=click.Path(),
    default="exploration/video-analysis/data"
)
@click.argument(
    '--station',
    type=click.STRING,
    default="bia"
)
def stage_edf_file(input_path: str, output_path: str, station: str):
    """convert edf file to parquet file

    Args:
        input_path (str): input path in adls lake
        output_path (str): output path in adls lake
        station (str): station as this info cannot be found in edf files

    Raises:
        FileNotFoundError: if file does not exist on adls
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # read respiration data
    adlsFileSystemClient = azure_utils.adls_client(
        os.environ.get("KEY_VAULT_URL"),
        os.environ.get("DATALAKE_NAME"))

    data_paths = adlsFileSystemClient.ls(input_path)

    # only data paths
    data_paths = [i for i in data_paths if re.search(r"_HRD.edf", i)]

    for file_name in data_paths:
        file_name = str(file_name)

        target_path = os.path.join(output_path,
                                   os.path.basename(file_name))

        if not adlsFileSystemClient.exists(target_path):
            logger.info(f"processing file {file_name}")
            # download edf file to temp storage and convert it to df
            with tempfile.NamedTemporaryFile(suffix=".edf") as f:
                adlsFileSystemClient.get(
                    file_name,
                    f.name
                )

                edf_df = edf.process_resmed(
                    file_path=str(f.name), station=station)

            table = pa.Table.from_pandas(edf_df)

            with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
                pq.write_table(table, f.name)
                adlsFileSystemClient.put(f.name, f"{target_path}.parquet")

        else:
            logger.info(f"file {target_path} already exists, not processing.")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    # convert edf-files into paquet format
    stage_edf_file()
