# anomalia

Anomaly Detection for Sequence Data (time series)

## Project setup

The project is a python project. We suggest to create a
conda environment (anaconda or miniconda) and to avtivate it, e.g.

```sh
conda create newenv 
conda activate newenv
```

After setting up your environment, clone/pull the project to your machine, like:

```sh
git clone https://github.com/yvesmauron/time-series-anomaly-detection
```

and install the required packages (will consume some storage space) using the `requirements.txt`.

```sh
pip install -r requirements.txt
```

The only additional file you need is the `.env` file containing the environment varaibales necessary to connect to azure ressources. To get the file, please contact the project owner. 

> Please be careful with this file and do not share with others not authorized to work on the project. Also, do not upload it to servers that you do not control.

## Usage

To process incoming edf files, navigate to your project directory and activate the environment you created above:

```sh
python src/data/stage_resmed.py --input_path /path/ --output_path /path/ --station bia
```

To create download the data for training or prediction on your machine, call `make_dataset` as follows:

```sh
python src/data/make_dataset.py
```

You can now start training the model as follows:

```sh
python src/models/train.py
```

The parameters used for the training and the training progress can be investigated with mlflow; you can start the mlflow ui as follows:

```sh
mlflow ui
```

After successfully training the model you can start with the prediction process. To predict the anomaly score you must provide the `run_id` (you can guid from the mlflow ui), also you can specify i.e. with the `score_file_pattern` to e.g. only predict data from December 2020  with the following pattern `202012*`

This would then look similar to that.

```sh
python src/models/predict.py --run_id=4d8ddb41e7f340c182a6a62699502d9f --score_file_pattern=202012*
```

To analyze your predictions and better understand your model, you can use the explain ui implemented with dash. To run it, you can use the following command:

```sh
python src/visualization/explain.py
```

> You can get all the available parameters for all of the above scripts by calling `python path/to/script.py --help`

## For developers

Please note that this project is currently under active development and by no means finished. Additional features, such as workflow tool (like e.g. Airflow) etc. are prioritized, planned and implemented by the project team. Please contact the project team if you plan to contribute, there are lots of things to do :-).

### Project Organization

The project follows the data science cookiecutter template. Thus the code is structured as follows:

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
