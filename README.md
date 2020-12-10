anomalia
==============================
Project for 
Anomaly Detection for Sequence Data (time series)


------------------------
Description
------------------------


Preparation
------------------------
The project is a python project. Best is to create an own
conda environment (anaconda or miniconda) and to avtivate this, e.g.
commands:
    conda create newenv 
    conda activate newenv


Installation 
------------------------
1. Git Project 
    get the whole git project by 
    
    command:
        git clone https://github.com/maurony/ts-vrae

2. Required python packages
    the file requirements.txt contains all needed python libraries.
    (this will consume a frew GBs)
    
    command:
        pip install -r requirements.txt

3. Setup environment variaples
    You to put the file .env into your project's root directory (installation path).
    It contains all neccessary access rights and can be queried by the project care takers
    Yvves Mauro or Martin Mosisch. 

4. Download data base
    To download the data sources you need to download about 600 Megabytes (10. Dec. 2020)
    !! Madatory: You need the file .env file !!     
    
    command:
        python src/data/make_dataset.py
    
5. Start training 
    command:
        python src/models/train.py

6. GUI
    Web-base grafical User Interface
    to start the server 
    command:
        mlflow ui

( if needed ...)
7. Convert data (may your own) into .paquet file format
    command:
        python stage_edf_file.py



Project Organization
------------------------

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
