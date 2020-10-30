import pyarrow.parquet as pq
import shutil
from pathlib import Path
import os
import logging
import click
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
import src.visualization.visualize_attention as viz
atts = viz.visualize_attention(
    run_id="4377a3ad68e84162827255bc1a0b7e40",
    session="20200930_120001",
    epoch_nr=361
)


@click.command()
@click.option(
    '--run_id',
    type=click.STRING,
    help="run id from mlflow experiment. check mlflow ui.",
    default="d61422fcea4c4e6e913049b9149fbe68"
)
def visualize_predictions(
    run_id: str
):

    logger = logging.getLogger(__name__)

    logger.info("Setting up directory structure")
    report_figures_path = Path(
        os.path.join("reports", "figures", run_id, "predictions"))

    # remove directories if they exist
    if report_figures_path.exists():
        shutil.rmtree(report_figures_path)

    report_figures_path.mkdir(parents=True, exist_ok=True)

    # get mlflow client
    # mlflow_client = MlflowClient()
    # get run to be explained
    # data = mlflow_client.get_run(run_id).data

    scored_path = Path(os.path.join("data", "output", "score", run_id))

    color_palette = {
        "resp_flow": "rgba(247, 201, 77, 1)",
        "deli_volu": "rgba(64, 145, 182, 1)",
        "mask_pres": "rgba(105, 173, 82, 1)",
        "true": "rgba(0, 0, 0, 1)",
        "se_resp_flow": "rgba(247, 201, 77, 1)",
        "se_deli_volu": "rgba(64, 145, 182, 1)",
        "se_mask_pres": "rgba(105, 173, 82, 1)",
    }

    sessions = [str(os.path.basename(p))[:15] for p in scored_path.iterdir()]

    logger.info(
        "Starting to procuce prediction plots. This could take a while.")
    # sessions = sessions[-1:]
    with tqdm(
        total=len(sessions),
        bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:100}{r_bar}",
        ascii=True
    ) as pbar:
        for session in sessions:
            pbar.set_postfix(file=session)

            df = pq.read_table(
                os.path.join(scored_path, f"{session}_0_HRD.edf.parquet")
            ).to_pandas()
            df = df.loc[df["delivered_volum"] > -32768, :]

            fig = viz.plot_signals(
                session,
                df,
                color_palette
            )

            fig.write_html(
                os.path.join(
                    report_figures_path,
                    f"{session}.html"))

            pbar.update(1)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    #
    _ = visualize_predictions()
