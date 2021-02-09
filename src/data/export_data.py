import pyarrow.parquet as pq
import os
from pathlib import Path
import shutil


def to_csv(input_path: str = "data/output/score/4d8ddb41e7f340c182a6a62699502d9f",
           output_path: str = "data/export"):
    """Export data to csv format format

    Args:
        input_path (str): Input Directory
        output_path (str, optional): Output directory, this directory will be
            truncated. Defaults to "data/export".

    Raises:
        FileNotFoundError: [description]
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError

    if output_path.exists():
        shutil.rmtree(output_path)

    output_path.mkdir()

    for f in input_path.iterdir():

        pred_df = pq.read_table(
            os.path.join(f)
        ).to_pandas()

        pred_df.to_csv(
            os.path.join(
                output_path,
                f"{ os.path.basename(f)[:-12]}_full.csv"),
            index=False,
            float_format='%.15f'
        )

        col_names = [
            col for col in pred_df.columns if not col.endswith("_scaled")]

        pred_df = pred_df.loc[:, col_names].to_csv(
            os.path.join(
                output_path,
                f"{ os.path.basename(f)[:-12]}_reduced.csv"),
            index=False,
            float_format='%.15f'
        )


to_csv()
