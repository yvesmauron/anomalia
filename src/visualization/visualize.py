
import tempfile
import json
import os
from pathlib import Path
import logging

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import mlflow

from src.features.build_features import reshape_resmed_tensor
from src.models.anomalia.datasets import ResmedDatasetEpoch
from src.visualization.cluster import hdbscan_cluster


import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from mlflow.tracking import MlflowClient
import torch

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger("visualization")


def plot_signals(
    session: str,
    df: pd.DataFrame,
    color_palette: dict = {
        "resp_flow": "rgba(247, 201, 77, 1)",
        "deli_volu": "rgba(64, 145, 182, 1)",
        "mask_pres": "rgba(105, 173, 82, 1)",
        "true": "rgba(0, 0, 0, 1)",
        "se_resp_flow": "rgba(247, 201, 77, 1)",
        "se_deli_volu": "rgba(64, 145, 182, 1)",
        "se_mask_pres": "rgba(105, 173, 82, 1)",
    }
):

    # subplot -----
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_titles=("Resp Flow", "Delivered Volume",
                    "Mask Pressure", "Anomaly Score")
    )

    # error -----
    fig.add_trace(
        go.Scatter(
            x=df.timestamp,
            y=df.mask_press_se,
            mode='lines',
            name='Scaled SE MaskPressure',
            line=dict(
                color=color_palette["se_mask_pres"]
            )
        ),
        row=4,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.timestamp,
            y=df.delivered_volum_se,
            mode='lines',
            name='Scaled SE Deliveredvolume',
            line=dict(
                color=color_palette["se_deli_volu"]
            )
        ),
        row=4,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.timestamp,
            y=df.resp_flow_se,
            mode='lines',
            name='Scaled SE RespFlow',
            line=dict(
                color=color_palette["se_resp_flow"]
            )
        ),
        row=4,
        col=1
    )

    # Mask Pressure -----
    fig.add_trace(
        go.Scatter(
            x=df.timestamp,
            y=df.mask_press,
            mode='lines',
            name='Mask Pressure (True)',
            line=dict(
                color=color_palette["true"]
            )
        ),
        row=3,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.timestamp,
            y=df.mask_press_mu,
            mode='lines',
            name='Mask Pressure (Predicted)',
            line=dict(
                color=color_palette["mask_pres"]
            )
        ),
        row=3,
        col=1
    )

    # Delivered Volume -----
    fig.add_trace(
        go.Scatter(
            x=df.timestamp,
            y=df.delivered_volum,
            mode='lines',
            name='Delivered Volume (True)',
            line=dict(
                color=color_palette["true"]
            )
        ),
        row=2,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.timestamp,
            y=df.delivered_volum_mu,
            mode='lines',
            name='Delivered Volume (Predicted)',
            line=dict(
                color=color_palette["deli_volu"]
            )
        ),
        row=2,
        col=1
    )

    # Respiration Flow -----
    fig.add_trace(
        go.Scatter(
            x=df.timestamp,
            y=df.resp_flow,
            mode='lines',
            name='Resp Flow (True)',
            line=dict(
                color=color_palette["true"]
            )
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.timestamp,
            y=df.resp_flow_mu,
            mode='lines',
            name='Resp Flow (Predicted)',
            line=dict(
                color=color_palette["resp_flow"]
            )
        ),
        row=1,
        col=1
    )

    fig.update_layout(
        title_text=f"Session example: {session}",
        legend_title=None,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-.15,
            xanchor="right",
            x=1
        )
        # font=dict(
        # family="Courier New, monospace",
        # size=18
        # )
    )

    return fig


def latent_pca_data(
    run_id: str,
    pca_components: int = 3,
    test_lookback: str = 20201020
):
    logger = logging.getLogger(__name__)

    # get mlflow client
    mlflow_client = MlflowClient()
    # get run to be explained
    data = mlflow_client.get_run(run_id).data
    # get latent size
    latent_size = int(data.params["latent_size"])

    # read from directory
    latent_dir = os.path.join("data/output/explain/latent", run_id)
    latents = []

    # for p in Path(latent_dir).iterdir():
    #     df = pq.read_table(p).to_pandas()
    #     latents.append(df)

    # train = pd.concat(latents, axis=0)
    # train = train.loc[train["epoch_class"] >= 0, :]
    #
    # train_scaled = scaler.transform(train.iloc[:, :latent_size])

    latents = []
    test_paths = [
        p for p in Path(latent_dir).iterdir()
        if int(os.path.basename(p)[:8]) > test_lookback
    ]

    for p in test_paths:
        df = pq.read_table(p).to_pandas()
        latents.append(df)

    test = pd.concat(latents, axis=0)
    test = test.loc[test["epoch_class"] >= 0, :]
    scaler = RobustScaler().fit(test.iloc[:, :latent_size])
    test_scaled = scaler.transform(test.iloc[:, :latent_size])

    print(test.shape)

    labels, probs = hdbscan_cluster(test_scaled, min_cluster_size=5)

    # PCA
    logger.info(f"Creating PCA with {pca_components} components.")

    # fit pca
    pca = PCA(n_components=pca_components)
    pca.fit(test_scaled)

    components = pca.transform(test_scaled)

    # create df for visualization
    pca_columns = [f"PC{i+1}" for i in range(pca_components)]
    components = pd.DataFrame(
        components, columns=pca_columns

    ).reset_index()

    explained = pca.explained_variance_ratio_.sum() * 100

    return test, components, explained, labels, probs


def plot_latent(
    run_id: str,
    pca_components: int = 3,
    tsne_components: int = 2,
    compute_tsne: bool = False
):
    logger = logging.getLogger(__name__)

    # get mlflow client
    mlflow_client = MlflowClient()
    # get run to be explained
    data = mlflow_client.get_run(run_id).data
    # get latent size
    latent_size = int(data.params["latent_size"])

    # read from directory
    latent_dir = os.path.join("data/output/explain/latent", run_id)
    latents = []

    for p in Path(latent_dir).iterdir():
        table = pq.read_table(p)
        latents.append(table.to_pandas())

    df = pd.concat(latents, axis=0)

    # PCA
    logger.info(f"Creating PCA with {pca_components} components.")

    # fit pca
    pca = PCA(n_components=pca_components)
    components = pca.fit_transform(df.iloc[:, :latent_size])

    # create df for visualization
    pca_columns = [f"PC{i+1}" for i in range(pca_components)]
    components = pd.DataFrame(
        components, columns=pca_columns

    ).reset_index()
    components = pd.concat(
        [df.iloc[:, latent_size:].reset_index(), components], axis=1)

    total_var = pca.explained_variance_ratio_.sum() * 100

    labels = {str(i): f"PC {i+1}" for i in range(pca_components)}
    labels['color'] = 'log(epoch_loss)'

    # fit latent
    pca_fig = px.scatter_matrix(
        components,
        color=np.log(df.epoch_loss.astype("float")),
        dimensions=pca_columns,
        labels=labels,
        title=f'Run: {run_id}; Total Explained Variance: {total_var:.2f}%',
        hover_name="file_name",
        hover_data=["epoch_loss", "epoch"]

    )
    pca_fig.update_traces(diagonal_visible=False)

    if compute_tsne:
        # TSNE
        logger.info(f"Creating TSNE with {tsne_components} components.")
        tsne = TSNE(n_components=tsne_components, random_state=0)
        projections = tsne.fit_transform(df.iloc[:, :latent_size])

        projections = pd.DataFrame(
            projections, columns=[
                "P1", "P2"]).reset_index()
        projections = pd.concat(
            [df.iloc[:, latent_size:].reset_index(), projections], axis=1)

        tsne_fig = px.scatter(
            projections, x="P1", y="P2",
            color=np.log(df.epoch_loss.astype("float")),
            labels={'color': 'loc(epoch_loss)'},
            title=f'Run: {run_id}',
            hover_name="file_name",
            hover_data=["epoch_loss", "epoch"]
        )

        return pca_fig, tsne_fig
    else:
        return pca_fig, None


def epoch_attention(
    run_id: str,
    session: str,
    epoch_nr: int,
    input_dir: str = "data/processed/resmed/score",
    preprocessing_config: str = "config/preprocessing_config.json",
    seq_len: int = 750,
    device: str = "cuda"
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

    logger = logging.getLogger(__name__)

    logger.info(f"Reading file {session}.")
    # clean up output_dir
    session_file = Path(
        os.path.join(
            input_dir,
            f"{session}_0_HRD.edf.parquet"))

    if not session_file.exists():
        logger.error(f"file {str(session_file)} does not exist.")

    attention_path = Path(
        os.path.join(
            "reports/figures",
            run_id,
            "attention"
        )
    )
    # remove directories if they exist
    if not attention_path.exists():
        attention_path.mkdir(parents=True, exist_ok=True)

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
    if device == "cuda":
        smavra.cuda()
    else:
        smavra.cpu()

    logger.info("Getting attention.")

    pred_df = pq.read_table(
        session_file
    ).to_pandas()

    pred_tensor = torch.Tensor(pred_df.iloc[:, :3].values)
    pred_tensor = reshape_resmed_tensor(pred_tensor, seq_len)

    dataset = ResmedDatasetEpoch(
        data=pred_tensor,
        batch_size=1,
        device=device,
        means=torch.Tensor(preprocessing_config["means"]),
        stds=torch.Tensor(preprocessing_config["stds"])
    )

    epoch = dataset[epoch_nr].unsqueeze(0).to(device)

    # get prediction
    h_t, latent, attention_weight, attention, lengths = \
        smavra.encode(epoch)

    _, n_heads, _, _ = attention_weight.shape
    attention_weights = {
        j: attention_weight[0, j, :, :].cpu().detach().numpy()
        for j in range(n_heads)
    }

    return(attention_weights)
