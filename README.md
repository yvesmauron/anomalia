# Anomaly detection algorithms for time series

This repository aims at collecting different approaches of detecting anomalies in time-series; both univariate and multi-variate. Experiments can run either on your local computer/server or on azure machine learning services. 

## Setup

### Installation

The current implementation uses conda for package management; which can be installed as follows (for Windows or more information about the installation process, please check [minicondas documentation site](https://docs.conda.io/en/latest/miniconda.html)):

```shell
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Thereafter you can install all dependencies used for this project with the follwoing conda command:

```shell
conda env create -f environment.yml
```

If everything went fine, you should be now able to run and/or develop experiments.

### Configuration

As azure services can be used, there is some additional configuration necessary, to e.g. the azure machine learning workspace, service principle etc. At the moment, 4 configurtion files are used in the code; that contain all the necessary information to use this services. If you want to use Azure to train your models, you need to create a `./config` directory and place the following files in this folder.

`ws_config.json`; which is workspace configuration of the azure machine learning services workspace (this can be exported from the Azure portal directly and has the follwoing format):

```json
{
    "subscription_id": "YOUR_SUBSCRIPTION_ID",
    "resource_group": "YOUR_RESOURCE_GROUP",
    "workspace_name": "YOUR_WORKSPACE_NAME"
}
```

`sp_config.json`; which holds information of the service principle so that the application can access the data lake storage:

```json
{
    "tenant_id": "YOUR_TENANT_ID",
    "adls_accountname": "YOUR_ADLS_ACCOUNT_NAME",
    "adls_client_id": "YOUR_ADLS_CLIENT_ID",
    "adls_client_secret": "YOUR_ADLS_CLIENT_SECRET"
}
```

*Optionally*, you can also define `ds_config.json` which lets you directly create file datasets within the azure machine learning servcies workspace from a folder stored in the azure data lake storage. The structure to

```json
{
    "data_store_name":"DATA_STORE_NAME",
    "datasets":[
        {
            "name":"NAME_OF_THE_DATASET",
            "description":"DESCRIPTION",
            "path":"PATH_WITHIN_THE_ADLS"
        },
        ...
    ]
}
```

> Note: if you are registering the dataset within azure machine learning services, you can then also connect to this dataset ML designer.

## Training Process

![model_fitting](img/animated.gif)

## Running experiments on Azure

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fmaurony%2Fts-vrae%2Fmaster%2Fazuredeploy.json)
