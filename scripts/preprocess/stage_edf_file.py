# general imports
from anomalia import utils
import matplotlib.pyplot as plt
import logging.config
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pickle as pkl
import shutil
import pyarrow.parquet as pq
import pyarrow as pa
import mne
from pathlib import Path
import argparse
import json
import os
import sys
sys.path.append(os.getcwd())

#import glob
# compression
# data transformations
# logging
# atemreich imports
# dev

# ------------------------------------------------------------------------
# initialize logger
logging.config.fileConfig(
    os.path.join(os.getcwd(), 'config', 'logging.conf')
)

# create logger
logger = logging.getLogger('anomalia')


def stage_edf_file(source_folder: str, target_folder: str, station: str):
    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y%m%d")
    staging_path = os.path.join(target_folder, date_time)

    # read respiration data
    # check if arguments are valid, i.e. if directory exists
    if not os.path.exists(source_folder):
        logger.error(f'Source directory: {source_folder} is not valid')
        raise FileNotFoundError

    # check if arguments are staging_path, i.e. if directory exists
    if os.path.exists(staging_path):
        logger.warning(
            'Unclassified directory: {} already exists, it will be recreated.'.format(staging_path))
        shutil.rmtree(staging_path)
        os.makedirs(staging_path)
    else:
        logger.info(
            'Unclassified directory: {} it will be created.'.format(staging_path))
        os.makedirs(staging_path)

    source_path = Path(source_folder)

    source_files = list(source_path.glob("*HRD.edf"))

    for file_name in source_files:
        file_name = str(file_name)
        logger.info(f"processing file {file_name}")

        edf_file = mne.io.read_raw_edf(file_name)

        channels = edf_file.ch_names

        cols = len(channels)
        rows = len(edf_file[0][0][0])

        edf_data = np.zeros((rows, cols))

        for i in range(len(channels)):
            edf_data[:, i] = edf_file[channels[i]][0][0]

        edf_df = pd.DataFrame(edf_data)
        edf_df.columns = [c.lower() for c in channels]

        edf_df["timestamp"] = edf_file.info["meas_date"]
        edf_df["time_offset"] = edf_file[0][1] - min(edf_file[0][1])
        edf_df["timestamp"] = edf_df["timestamp"] + \
            pd.to_timedelta(edf_file[0][1], "s")
        edf_df["timestamp"] = edf_df["timestamp"].dt.strftime(
            "%Y-%m-%d %H:%M:%S.%f")
        edf_df["station"] = station

        table = pa.Table.from_pandas(edf_df)
        source_file_name = os.path.basename(str(file_name))
        pq.write_table(table, os.path.join(
            staging_path, f"{source_file_name}.parquet"))


# ------------------------------------------------------------------------
# Parse arguments
parser = argparse.ArgumentParser(
    description='Convert edf files to parquet'
)
# get arguments
parser.add_argument(
    "--source_folder",
    help="The folder where the respiration files are saved",
    default='data/resmed/raw/Bianca'
)

# get arguments
parser.add_argument(
    "--target_folder",
    help="The folder where converted files will be saved",
    default='data/resmed/staging'
)

# get arguments
parser.add_argument(
    "--station",
    help="The station from which the data was recorded",
    default='BIA'
)

args = parser.parse_args()

if __name__ == '__main__':

    source_folder = str(args.source_folder)
    target_folder = str(args.target_folder)
    station = str(args.station)
    stage_edf_file(source_folder, target_folder, station)
