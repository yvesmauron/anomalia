from pathlib import Path
import os
import click
import logging
from dotenv import find_dotenv, load_dotenv
import shutil
from src.visualization import visualize as viz


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

    latent_path = Path(
        os.path.join("reports", "figures", run_id, "latent"))

    # remove directories if they exist
    if latent_path.exists():
        shutil.rmtree(latent_path)

    latent_path.mkdir(parents=True, exist_ok=True)

    pca_fig, tsne_fig = viz.plot_latent(
        run_id,
        pca_components,
        tsne_components
    )

    pca_fig.write_html(
        os.path.join(
            latent_path,
            f"pca_{pca_components}.html"
        )
    )

    tsne_fig.write_html(
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
