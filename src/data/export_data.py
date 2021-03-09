import pandas as pd
from src.visualization import visualize as viz
from datetime import datetime
import pyodbc as odbc
import uuid
import pyarrow.parquet as pq
import os
from pathlib import Path
import shutil
from dotenv import find_dotenv, load_dotenv


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


def generate_segment_markers(df: pd.DataFrame, qualifier_text: str) -> pd.DataFrame:

    df = (
        df
        .groupby("epoch_id")
        .agg(
            StartOfSegment=pd.NamedAgg(column='timestamp', aggfunc='min'),
            EndOfSegment=pd.NamedAgg(column='timestamp', aggfunc='max'))
        .reset_index()
        .loc[:, ["StartOfSegment", "EndOfSegment"]]
    )
    n_rows = len(df.index)
    create_date = [datetime.now() for _ in range(n_rows)]
    df = pd.DataFrame(dict(
        EntityId=[uuid.uuid4() for _ in range(n_rows)], QualifierText=[qualifier_text for _ in range(n_rows)], CreatedDate=create_date, UpdatedDate=create_date, StartOfSegment=df.StartOfSegment, EndOfSegment=df.EndOfSegment, CorrectedByUser=0, IsCorrect=0, YAxis=3
    ))

    return df


def insert_markers(df: pd.DataFrame):
    load_dotenv(find_dotenv())

    cnxn = odbc.connect(
        f'DSN={os.getenv("DSN")};' +
        f'UID={os.getenv("User")};' +
        f'PWD={os.getenv("Password")};' +
        f'Database={os.getenv("Database")}'
    )
    # (?,?,?,?,?,?,?,?,?)

    cursor = cnxn.cursor()

    for index, row in df.iterrows():
        cursor.execute("""
            INSERT INTO [dbo].[SegmentMarkers]
            (
                [EntityId],
                [QualifierText],
                [CreatedDate],
                [UpdatedDate],
                [StartOfSegment],
                [EndOfSegment],
                [CorrectedByUser],
                [IsCorrect],
                [YAxis]
            )
            VALUES (?,?,?,?,?,?,?,?,?)""",
                       row.EntityId,
                       row.QualifierText,
                       row.CreatedDate,
                       row.UpdatedDate,
                       row.StartOfSegment,
                       row.EndOfSegment,
                       row.CorrectedByUser,
                       row.IsCorrect,
                       row.YAxis
                       )

    cnxn.commit()
    cursor.close()


def export_aitainer(
    run_id: str = '4d8ddb41e7f340c182a6a62699502d9f',
    test_lookback: int = 202012
):

    # run_id = '4d8ddb41e7f340c182a6a62699502d9f'
    # test_lookback = 202012
    test, components, explained, labels, probs = viz.latent_pca_data(
        run_id=run_id,
        test_lookback=test_lookback
    )
    test["labels"] = labels

    label_dict = (
        test.loc[test.labels == -1, ["file_name", "epoch"]]
        .groupby("file_name")
        .agg({"epoch": list})
        .to_dict()["epoch"]
    )

    markers = None

    for file_name, epochs in label_dict.items():
        score_file = os.path.join(
            "data/output/score", run_id, f"{file_name}_0_HRD.edf.parquet")

        df = pq.read_table(score_file, columns=[
            "epoch_id", "timestamp", "epoch_class", "epoch_mse"]).to_pandas()

        default_values = generate_segment_markers(
            df.loc[df.epoch_class == -1, :], "Default Wert"
        )

        disconnection = generate_segment_markers(
            df.loc[df.epoch_id.isin(epochs) & (
                df.epoch_mse > 2.2), :], "Diskonnektion"
        )

        if markers is not None:
            markers = pd.concat([markers, default_values, disconnection])
        else:
            markers = pd.concat([default_values, disconnection])

    insert_markers(markers)
