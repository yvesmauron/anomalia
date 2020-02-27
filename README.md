# Anomaly detection algorithms for time series

This repository aims at collecting different approaches of detecting anomalies in time-series; both univariate and multi-variate. Experiments can run either on your local computer/server or on azure machine learning services. 

## Installation

The current implementation uses conda for package management; which can be installed as follows:

```shell
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Thereafter you can install all dependencies used for this project with the follwoing conda command:

```shell
conda env create -f environment.yml
```

## Running experiments on Azure

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fmaurony%2Fts-vrae%2Fmaster%2Fazuredeploy.json)
