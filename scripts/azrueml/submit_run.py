import os
import sys
import shutil
import argparse
# add working directory to python path for later imports
sys.path.append(os.getcwd())

from azureml.core import Experiment, Datastore
from azureml.core.workspace import Workspace
from azureml.core.runconfig import RunConfiguration
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
logger = logging.getLogger('anomalia')

# ------------------------------------------------------------------------
# Parse arguments
parser = argparse.ArgumentParser(
    description='create train and test set for lstm'
)
# get arguments
parser.add_argument(
    "--ws_config", 
    help="The file path that holds the machine learning configuration file",
    default='./config/ws_config.json'
)

parser.add_argument(
    "--experiment_folder", 
    help="The folder name that will then contain all resources to run the experiment on azure machine learning services",
    default='./dist'
)

parser.add_argument(
    "--env_file", 
    help="The file path that holds the environment file",
    default='./environment.yml'
)

parser.add_argument(
    "--compute_node", 
    help="The file path that holds the environment file",
    default='local'
)
# get arguments
parser.add_argument(
    "--output_dir", 
    help="output dir to log",
    default='./outputs'
)

# get arguments
parser.add_argument(
    "--ds_name", 
    help="Configuration for azure ml workspace",
    default="resmed_train"
)

# get arguments
parser.add_argument(
    "--batch_size", 
    help="The batch size that should be used to train the model",
    default=64
)

# get arguments
parser.add_argument(
    "--n_epochs", 
    help="The number of epochs to train the neural network",
    default=1000
)

# get arguments
parser.add_argument(
    "--data_store_name", 
    help="Configuration for azure ml workspace",
    default="atemlake"
)

args = parser.parse_args()

if __name__ == '__main__':

    # get input arguments
    ws_config_path = str(args.ws_config)
    experiment_folder = str(args.experiment_folder)
    env_file = str(args.env_file)
    compute_node = str(args.compute_node)
    ompute_node_name = compute_node.replace('_', '')
    batch_size = int(args.batch_size)
    n_epochs = int(args.n_epochs)
    ds_name = str(args.ds_name)
    output_dir = str(args.output_dir)
    data_store_name = str(args.data_store_name)


    # ------------------------------------------------------------------------
    # load workspace and check compute_node input
    if not os.path.exists(ws_config_path):
        logger.error('Workspace config not found in path: {}'.format(ws_config_path))
        raise FileNotFoundError

    ws = Workspace.from_config(path=ws_config_path)
    vms = [_['name'] for _ in AmlCompute.supported_vmsizes(workspace=ws)]

    if compute_node not in vms and compute_node is not 'local':
        logger.error('compute node: {} not supported by workspace'.format(compute_node))
        raise ValueError

    # ------------------------------------------------------------------------
    # Creating Azure Machine learning train dir with relevant scripts
    logger.info('Creating Azure Machine learning train dir with relevant scripts')

    # check if arguments are valid, i.e. if directory exists
    if os.path.exists(experiment_folder):
        logger.warning('Create directory for AML training: {} already exists, it will be recreated.'.format(experiment_folder))
        shutil.rmtree(experiment_folder)
        os.makedirs(experiment_folder)
        os.makedirs(os.path.join(experiment_folder, 'config'))
        os.makedirs(os.path.join(experiment_folder, 'outputs'))
    else:
        logger.info('Create directors for AML training: {} it will be created.'.format(experiment_folder))
        os.makedirs(experiment_folder)
        os.makedirs(os.path.join(experiment_folder, 'config'))
        os.makedirs(os.path.join(experiment_folder, 'outputs'))

    # copy train scripts to 
    shutil.copytree(src='./anomalia', dst=os.path.join(experiment_folder, 'anomalia'))
    shutil.copyfile(src='./scripts/training/train.py', dst=os.path.join(experiment_folder, 'train.py'))
    shutil.copyfile(src='./config/ws_config.json', dst=os.path.join(experiment_folder, 'config', 'ws_config.json'))
    shutil.copyfile(src='./environment.yml', dst=os.path.join(experiment_folder, 'environment.yml'))
    shutil.copyfile(src='./config/logging.conf', dst=os.path.join(experiment_folder, 'config', 'logging.conf'))

    # ------------------------------------------------------------------------
    # Compute target

    if ompute_node_name is not 'local':
        try:
            # check if ComputeTarget is valid here, as it seems to be abstract class
            # see: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-pytorch
            compute_target = ComputeTarget(workspace=ws, name=ompute_node_name)
            logger.info('Found existing compute target.')
        except ComputeTargetException:
            logger.info('Creating a new compute target...')
            compute_config = AmlCompute.provisioning_configuration(
                vm_size=compute_node, 
                max_nodes=1
            )

            # create the cluster
            compute_target = ComputeTarget.create(ws, ompute_node_name, compute_config)

            compute_target.wait_for_completion(show_output=True)
    else:
        compute_target = 'local'

    # ------------------------------------------------------------------------
    # Define experiment
    experiment_name = 'SMAVRA'
    experiment = Experiment(ws, name=experiment_name)

    script_params = {
        '--ws_config': ws_config_path,
        '--output_dir': output_dir,
        '--ds_name':ds_name,
        '--batch_size':batch_size,
        '--n_epochs':n_epochs,
        '--compute_node':compute_node
    }

    #datastore = Datastore.get(ws, data_store_name)

    estimator = PyTorch(
        source_directory=experiment_folder, 
        script_params=script_params,
        compute_target=compute_target,
        entry_script='train.py',
        use_gpu=True,
        use_docker=True if compute_target is not 'local' else False,
        user_managed=False if compute_target is not 'local' else True,
        conda_dependencies_file='environment.yml' if compute_target is not 'local' else None
        #source_directory_data_store=datastore
    )

    run = experiment.submit(estimator)

