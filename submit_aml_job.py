import os
import shutil
import argparse

from azureml.core.workspace import Workspace
from azureml.core import Experiment

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import RunConfiguration
from azureml.train.dnn import PyTorch

import logging
import logging.config

# ------------------------------------------------------------------------
# initialize logger
logging.config.fileConfig(
    os.path.join(os.getcwd(), 'config', 'logging.conf')
)

# create logger
logger = logging.getLogger('atemteurer')

# ------------------------------------------------------------------------
# Parse arguments
parser = argparse.ArgumentParser(
    description='create train and test set for lstm'
)
# get arguments
parser.add_argument(
    "--aml_config", 
    help="The file path that holds the machine learning configuration file",
    default='./config/aml_config.json'
)

parser.add_argument(
    "--aml_project_folder", 
    help="The folder name that will then contain all resources to run the experiment on azure machine learning services",
    default='./dist'
)

parser.add_argument(
    "--env_file", 
    help="The file path that holds the environment file",
    default='./environment.yml'
)

parser.add_argument(
    "--window_shift_step_size", 
    help="how much the input window should be shifted (allows for a higher sample count)",
    default=300
)

args = parser.parse_args()

if __name__ == '__main__':

    # get input arguments
    aml_config = str(args.aml_config)
    aml_project_folder = str(args.aml_project_folder)
    environment = str(args.environment)


# ------------------------------------------------------------------------
# initialize logger
ws = Workspace.from_config(path=aml_config)
vms = AmlCompute.supported_vmsizes(workspace=ws)


# ------------------------------------------------------------------------
# Creating Azure Machine learning train dir with relevant scripts

logger.info('Creating Azure Machine learning train dir with relevant scripts')

# check if arguments are valid, i.e. if directory exists
if os.path.exists(aml_project_folder):
    logger.warning('Create directory for AML training: {} already exists, it will be recreated.'.format(aml_project_folder))
    shutil.rmtree(aml_project_folder)
    os.makedirs(aml_project_folder)
    os.makedirs(os.path.join(aml_project_folder, 'config'))
else:
    logger.info('Create directors for AML training: {} it will be created.'.format(aml_project_folder))
    os.makedirs(aml_project_folder)
    os.makedirs(os.path.join(aml_project_folder, 'config'))

# copy train scripts to 
shutil.copytree(src='./atemteurer', dst=os.path.join(aml_project_folder, 'atemteurer'))
shutil.copyfile(src='./scripts/training/aml_train.py', dst=os.path.join(aml_project_folder, 'aml_train.py'))
shutil.copyfile(src='./data/resmed/train/train_resmed.pt', dst=os.path.join(aml_project_folder, 'train_resmed.pt'))
shutil.copyfile(src='./environment.yml', dst=os.path.join(aml_project_folder, 'environment.yml'))
shutil.copyfile(src='./config/logging.conf', dst=os.path.join(aml_project_folder, 'config', 'logging.conf'))

# ------------------------------------------------------------------------
# Compute target

# choose a name for your cluster
compute_node = "local"

try:
    compute_target = ComputeTarget(workspace=ws, name=compute_node)
    logger.info('Found existing compute target.')
except ComputeTargetException:
    logger.info('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(
        vm_size='Standard_NC6', 
        max_nodes=1
    )

    # create the cluster
    compute_target = ComputeTarget.create(ws, compute_node, compute_config)

    compute_target.wait_for_completion(show_output=True)

# use get_status() to get a detailed status for the current cluster. 
print(compute_target.get_status().serialize())
# ------------------------------------------------------------------------
# Define experiment

experiment_name = 'SMAVRA'
experiment = Experiment(ws, name=experiment_name)

script_params = {}

estimator = PyTorch(
    source_directory=aml_project_folder, 
    script_params=script_params,
    compute_target=compute_target,
    entry_script='aml_train.py',
    use_gpu=True,
    conda_dependencies_file='environment.yml'
)

run = experiment.submit(estimator)

