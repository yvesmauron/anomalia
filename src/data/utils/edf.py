import mne
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("mne")
logger.setLevel(logging.ERROR)


def process_resmed(file_path: str, station: str) -> pd.DataFrame:
    edf_file = mne.io.read_raw_edf(file_path)

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

    return edf_df
