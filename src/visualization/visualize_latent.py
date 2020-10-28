import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
import numpy as np
import pandas as pd
import os
import pyarrow.parquet as pq
import click
import logging
from dotenv import find_dotenv, load_dotenv
from mlflow.tracking import MlflowClient
import shutil


@click.command()
@click.option(
    '--run_id',
    type=click.STRING,
    help="run id from mlflow experiment. check mlflow ui.",
    default="d61422fcea4c4e6e913049b9149fbe68"
)
def explain_latent(
    run_id: str,
    pca_components: int = 3,
    tsne_components: int = 2
):

    logger = logging.getLogger(__name__)

    latent_path = Path(
        os.path.join("reports", "figures", run_id, "latent"))

    # remove directories if they exist
    if latent_path.exists():
        shutil.rmtree(latent_path)

    latent_path.mkdir(parents=True, exist_ok=True)

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
    fig = px.scatter_matrix(
        components,
        color=np.log(df.epoch_loss.astype("float")),
        dimensions=pca_columns,
        labels=labels,
        title=f'Run: {run_id}; Total Explained Variance: {total_var:.2f}%',
        hover_name="file_name",
        hover_data=["epoch_loss", "epoch"]

    )
    fig.update_traces(diagonal_visible=False)

    fig.write_html(
        os.path.join(
            latent_path,
            f"pca_{pca_components}.html"
        )
    )

    # TSNE
    logger.info(f"Creating TSNE with {tsne_components} components.")
    tsne = TSNE(n_components=tsne_components, random_state=0)
    projections = tsne.fit_transform(df.iloc[:, :latent_size])

    projections = pd.DataFrame(projections, columns=["P1", "P2"]).reset_index()
    projections = pd.concat(
        [df.iloc[:, latent_size:].reset_index(), projections], axis=1)

    fig = px.scatter(
        projections, x="P1", y="P2",
        color=np.log(df.epoch_loss.astype("float")),
        labels={'color': 'loc(epoch_loss)'},
        title=f'Run: {run_id}',
        hover_name="file_name",
        hover_data=["epoch_loss", "epoch"]
    )
    fig.write_html(
        os.path.join(
            latent_path,
            f"tsne_{tsne_components}.html"
        )
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
    explain_latent()
