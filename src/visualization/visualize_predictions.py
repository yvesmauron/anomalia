import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyarrow.parquet as pq
import shutil
from pathlib import Path
import os
import logging
import click
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm


@click.command()
@click.option(
    '--run_id',
    type=click.STRING,
    help="run id from mlflow experiment. check mlflow ui.",
    default="d61422fcea4c4e6e913049b9149fbe68"
)
def visualize_latent(
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

            # subplot -----
            fig = make_subplots(
                rows=4,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                subplot_titles=("Resp Flow", "Delivered Volume",
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
                legend_title="Legend Title",
                font=dict(
                    family="Courier New, monospace",
                    size=18
                )
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

    ##
    visualize_latent()
