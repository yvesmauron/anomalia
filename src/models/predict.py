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
from torch.utils.data import DataLoader, Dataset
import torch
import pyarrow.parquet as pq
import pyarrow as pa
import pickle as pk


def predict_file(
    model: torch.nn.Module,
    dataset: Dataset,
    file_path: str,
    explain_latent,
    explain_attention,
    seq_len: int,
    column_order: list
):
    score_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    preds = []
    latents = []
    attention_weights = {}

    with tqdm(
        total=len(score_loader),
        bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:100}{r_bar}",
        ascii=True
    ) as pbar:
        pbar.set_postfix(file=file_path)

        for i, epoch in enumerate(score_loader):
            # get prediction
            h_t, latent, attention_weight, attention, lengths = \
                model.encode(epoch)

            if explain_attention:
                _, n_heads, _, _ = attention_weight.shape
                attention_w = {
                    j: attention_weight[0, j, :, :].cpu().detach().numpy()
                    for j in range(n_heads)
                }
                attention_weights[i] = attention_w
            # ----------------------------------------------------
            # decoder
            mu_scaled, _ = model.decode(h_t, latent, attention, lengths)
            # reshape params
            batch, seq, fe = mu_scaled.shape

            # mse of epoch
            epoch_mse = torch.pow(
                (mu_scaled - epoch),
                2
            ).mean(axis=(1, 2)).cpu().detach().numpy()

            epoch_original_scale = dataset.backtransform(epoch.view(
                batch * seq, fe).cpu().detach()
            )

            if explain_latent:
                latents.append(
                    np.append(
                        arr=latent.squeeze().cpu().detach().numpy(),
                        values=[
                            epoch_mse,
                            i,
                            -1
                            if any(epoch_original_scale[:, 2] < -32760)
                            else 0
                        ]
                    )
                )

            epoch_mse = np.repeat(
                epoch_mse,
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

            mu = dataset.backtransform(mu_scaled.view(
                batch * seq, fe).cpu().detach()).squeeze(0).numpy()

            mu_scaled = mu_scaled.view(
                batch * seq, fe).cpu().detach().numpy()

            epoch_id = np.repeat([[i]], repeats=batch * seq, axis=0)

            epoch_class = -1 if any(
                epoch_original_scale[:, 2] < -32760) else 0
            epoch_class = np.repeat(
                [[epoch_class]], repeats=batch * seq, axis=0)

            predictions = np.concatenate(
                [epoch_id, mu_scaled, mu, epoch_mse, t_mse, m_se, epoch_class],
                axis=1
            )

            colnames = ["epoch_id"] \
                + [f"{_}_mu_scaled" for _ in column_order] \
                + [f"{_}_mu" for _ in column_order] \
                + ["epoch_mse", "t_mse"] \
                + [f"{_}_se" for _ in column_order] \
                + ["epoch_class"]

            predictions = pd.DataFrame(predictions, columns=colnames)

            preds.append(predictions)

            pbar.update(1)

    return preds, latents, attention_weights


@click.command()
@click.option(
    '--run_id',
    type=click.STRING,
    help="run id from mlflow experiment. check mlflow ui.",
    default="4d8ddb41e7f340c182a6a62699502d9f"
)
@click.option(
    '--input_dir',
    type=click.Path(),    
    default="data/processed/resmed/score",
    help="input directory holding the data to be predicted."
)
@click.option(
    '--output_dir',
    type=click.Path(),   
    default="data/output/",
    help="the directory the predictions should be written to."
)
@click.option(
    '--preprocessing_config',
    type=click.Path(),
    default="config/preprocessing_config.json",
    help="location of preprocessing config in mlflow."
)
@click.option(
    '--seq_len',
    type=click.INT,
    default=750,
    help="sequence lengths -> workaround. Defaults to 750."
)
@click.option(
    '--device',
    type=click.STRING,
    default="cuda",
    help="device to run inference on."
)
@click.option(
    '--explain_latent',
    type=click.BOOL,
    default=True,
    help="if latent space should be explained or not."
)
@click.option(
    '--explain_attention',
    type=click.BOOL,
    default=False,
    help="if attention space should be explained or not." +
         "(could lead to very large files)"
)
@click.option(
    '--score_file_pattern',
    type=click.STRING,
    default="*",
    help="Files prefix to be used for scoring."
)
def predict_smavra(
    run_id: str,
    input_dir: str = "data/processed/resmed/score",
    output_dir: str = "data/scored/resmed",
    preprocessing_config: str = "config/preprocessing_config.json",
    seq_len: int = 750,
    device: str = "cuda",
    explain_latent: bool = True,
    explain_attention: bool = False,
    score_file_pattern: str = "*"
):
    """predict using trained smavra network

    Args:
        run_id (str): run_id from mlflow experiment
        input_dir (str, optional): input directory holding the data to be
            predicted. Defaults to "data/processed/resmed/score".
        output_dir (str, optional): the directory the predictions should be
            written to. Defaults to "data/scored/resmed".
        preprocessing_config (str, optional): location of preprocessing config
            in mlflow. Defaults to "config/preprocessing_config.json".
        seq_len (int, optional): sequence lengths -> workaround.
            Defaults to 750.
        device (str, optional): device to run inference on.
    """
    column_order = ["mask_press", "resp_flow", "delivered_volum"]

    logger = logging.getLogger(__name__)

    logger.info("Preparing directory structure.")
    # clean up output_dir
    output_score_dir = Path(os.path.join(output_dir, "score", run_id))
    output_explain_latent_dir = Path(
        os.path.join(output_dir, "explain", "latent", run_id))
    output_explain_attention_dir = Path(
        os.path.join(output_dir, "explain", "attention", run_id))

    # remove directories if they exist
    if output_score_dir.exists():
        shutil.rmtree(output_score_dir)

    output_score_dir.mkdir(parents=True, exist_ok=True)

    # remove directories if they exist
    if output_explain_latent_dir.exists():
        shutil.rmtree(output_explain_latent_dir)

    output_explain_latent_dir.mkdir(parents=True, exist_ok=True)

    # remove directories if they exist
    if output_explain_attention_dir.exists():
        shutil.rmtree(output_explain_attention_dir)

    output_explain_attention_dir.mkdir(parents=True, exist_ok=True)

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
    smavra = mlflow.pytorch.load_model(
        'dbfs:/databricks/mlflow-tracking/1480730348517746/92bd93c5895144558545f21c6e5b2e08/artifacts/model', # 'runs:/' + run_id + '/model',
        map_location="cuda:0"
    )

    smavra.eval()
    if device == "cuda":
        smavra.cuda()
    else:
        smavra.cpu()

    logger.info("Start with predicition.")
    for score_file_path in Path(input_dir).glob(score_file_pattern):

        pred_df = pq.read_table(
            os.path.join(score_file_path)
        ).to_pandas()

        pred_tensor = torch.Tensor(pred_df.iloc[:, :3].values)
        pred_tensor = reshape_resmed_tensor(pred_tensor, seq_len)

        score_dataset = ResmedDatasetEpoch(
            data=pred_tensor,
            batch_size=1,
            device=device,
            means=torch.Tensor(preprocessing_config["means"]),
            stds=torch.Tensor(preprocessing_config["stds"])
        )
        preds, latents, attention_weights = predict_file(
            model=smavra,
            dataset=score_dataset,
            file_path=score_file_path,
            explain_latent=explain_latent,
            explain_attention=explain_attention,
            seq_len=seq_len,
            column_order=column_order
        )

        if len(preds) > 1:
            preds = pd.concat(preds, ignore_index=True)
        else:
            preds = preds[0]

        preds = pd.concat([
            pred_df,
            preds
        ], axis=1)

        # write predicitons
        table = pa.Table.from_pandas(preds)
        file_name = os.path.basename(score_file_path)
        pq.write_table(
            table,
            os.path.join(output_score_dir, file_name)
        )

        # EXPLAINABILITY
        # write latent
        if explain_latent:
            latents = np.stack(latents, 0)
            latent_cols = [
                f"latent_{i}" for i in range(
                    latents.shape[1] - 3)
            ]
            df = pd.DataFrame(
                latents, columns=latent_cols +
                ["epoch_loss", "epoch", "epoch_class"]
            )

            df["file_name"] = os.path.basename(score_file_path)[:15]
            table = pa.Table.from_pandas(df)

            file_name = os.path.join(
                output_explain_latent_dir,
                os.path.basename(score_file_path)
            )
            pq.write_table(
                table,
                file_name
            )

        # write attention
        if explain_attention:
            with open(
                os.path.join(
                    output_explain_attention_dir,
                    os.path.basename(score_file_path) + ".pkl"
                ),
                "wb"
            ) as f:
                pk.dump(attention_weights, f)


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
