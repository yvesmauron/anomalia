from pathlib import Path
import click
import logging
from dotenv import find_dotenv, load_dotenv

from src.visualization import visualize as viz


@click.command()
@click.option(
    '-r', '--run_id', 'run_id',
    type=click.STRING,
    help="run id from mlflow experiment. check mlflow ui.",
    default="4377a3ad68e84162827255bc1a0b7e40"
)
@click.option(
    '-s', '--session', 'session',
    type=click.STRING,
    default="20200930_120001",
    help="the directory the predictions should be written to."
)
@click.option(
    '-e', '--epoch_nr', 'epoch_nr',
    type=click.INT,
    default=361,
    help="input directory holding the data to be predicted."
)
@click.option(
    '--input_dir',
    type=click.Path(),
    default="data/processed/resmed/score",
    help="input directory holding the data to be predicted."
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
def visualize_attention(
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
    attention = viz.epoch_attention(
        run_id,
        session,
        epoch_nr,
        input_dir,
        preprocessing_config,
        seq_len,
        device
    )

    return(attention)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    ##
    atts = visualize_attention(
        run_id="4377a3ad68e84162827255bc1a0b7e40",
        session="20200930_120001",
        epoch_nr=361
    )
    atts = visualize_attention(
        "4377a3ad68e84162827255bc1a0b7e40",
        "20200724_120001",
        1039
    )

    import plotly.express as px

    fig = px.imshow(atts[1])
    fig.show()
