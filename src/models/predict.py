from mlflow.tracking import MlflowClient
import mlflow
import tempfile
import json
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import click
import logging
from dotenv import find_dotenv, load_dotenv

from src.models.anomalia.datasets import ResmedDatasetEpoch
from src.features.build_features import reshape_resmed_tensor
from src.models.anomalia.smavra import SMAVRA
from torch.utils.data import DataLoader
import torch
import pyarrow.parquet as pq
import pyarrow as pa


@click.command()
@click.option('--run_id', type=click.STRING, help="run id from mlflow experiment. check mlflow ui.")
@click.option('--input_dir', type=click.Path(), default="data/processed/resmed/score", help="input directory holding the data to be predicted.")
@click.option('--output_dir', type=click.Path(), default="data/scored/resmed", help="the directory the predictions should be written to.")
@click.option('--preprocessing_config', type=click.Path(), default="config/preprocessing_config.json", help="location of preprocessing config in mlflow.")
@click.option('--seq_len', type=click.INT, default=750, help="sequence lengths -> workaround. Defaults to 750.")
def predict_smavra(
    run_id: str,
    input_dir: str = "data/processed/resmed/score",
    output_dir: str = "data/scored/resmed",
    preprocessing_config: str = "config/preprocessing_config.json",
    seq_len: int = 750
):
    """predict using trained smavra network

    Args:
        run_id (str): run_id from mlflow experiment
        input_dir (str, optional): input directory holding the data to be predicted. Defaults to "data/processed/resmed/score".
        output_dir (str, optional): the directory the predictions should be written to. Defaults to "data/scored/resmed".
        preprocessing_config (str, optional): location of preprocessing config in mlflow. Defaults to "config/preprocessing_config.json".
        seq_len (int, optional): sequence lengths -> workaround. Defaults to 750.
    """
    column_order = ["mask_press", "resp_flow", "delivered_volum"]

    logger = logging.getLogger(__name__)

    logger.info("Preparing directory structure.")
    # clean up output_dir
    output_dir = Path(output_dir)

    # remove directories if they exist
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Setting up model.")
    mlflow_client = MlflowClient()

    # get processing info
    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(mlflow_client.download_artifacts(
            run_id=run_id,
            path="config/preprocessing_config.json",
            dst_path=tmp_dir
        ), "r") as f:
            preprocessing_config = json.load(f)

    # load model
    smavra = mlflow.pytorch.load_model('runs:/' + run_id + '/model')

    smavra.eval()
    # TODO: Move the model to the cpu (there is a bug somewhere)
    smavra.cuda()

    logger.info("Start with predicition.")
    for score_file_path in Path(input_dir).iterdir():

        pred_df = pq.read_table(
            os.path.join(score_file_path)
        ).to_pandas()

        pred_tensor = torch.Tensor(pred_df.iloc[:, :3].values)
        pred_tensor = reshape_resmed_tensor(pred_tensor, seq_len)

        score_dataset = ResmedDatasetEpoch(
            data=pred_tensor,
            batch_size=1,
            device="cuda",
            means=torch.Tensor(preprocessing_config["means"]),
            stds=torch.Tensor(preprocessing_config["stds"])
        )

        score_loader = DataLoader(
            score_dataset,
            batch_size=1,
            shuffle=False
        )

        preds = []

        with tqdm(total=len(score_loader), bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:100}{r_bar}", ascii=True) as pbar:
            pbar.set_postfix(file=score_file_path)
            for i, epoch in enumerate(score_loader):
                # get prediction
                mu_scaled, _ = smavra(epoch)
                # reshape params
                batch, seq, fe = mu_scaled.shape
                # move it back to cpu
                # mse of epoch
                epoch_mse = np.repeat(
                    torch.pow(
                        (mu_scaled - epoch),
                        2
                    ).mean(axis=(1, 2)).cpu().detach().numpy(),
                    seq_len
                )
                epoch_mse = epoch_mse.reshape(batch * seq, 1)
                # mse of timestep
                t_mse = torch.pow(
                    (mu_scaled - epoch),
                    2
                ).mean(axis=(2)).cpu().detach().numpy()
                t_mse = t_mse.reshape(batch * seq, 1)

                # se of timestep and measure
                m_se = torch.pow((mu_scaled - epoch), 2)
                m_se = m_se.view(batch * seq, fe).cpu().detach().numpy()

                mu = score_dataset.backtransform(mu_scaled.view(
                    batch * seq, fe).cpu().detach()).squeeze(0).numpy()

                mu_scaled = mu_scaled.view(
                    batch * seq, fe).cpu().detach().numpy()

                predictions = np.concatenate(
                    [mu_scaled, mu, epoch_mse, t_mse,  m_se], axis=1)

                colnames = [f"{_}_mu_scaled" for _ in column_order] \
                    + [f"{_}_mu" for _ in column_order] \
                    + ["epoch_mse", "t_mse"] \
                    + [f"{_}_se" for _ in column_order]

                predictions = pd.DataFrame(predictions, columns=colnames)

                preds.append(predictions)

                pbar.update(1)

        if len(preds) > 1:
            preds = pd.concat(preds, ignore_index=True)
        else:
            preds = preds[0]

        preds = pd.concat([pred_df, preds], axis=1)

        table = pa.Table.from_pandas(preds)
        file_name = os.path.basename(score_file_path)
        pq.write_table(
            table,
            os.path.join(output_dir, file_name)
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
    predict_smavra()
