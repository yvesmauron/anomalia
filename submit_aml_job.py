import os
import shutil

from azureml.core.workspace import Workspace
from azureml.core import Experiment

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
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
# initialize logger
ws = Workspace.from_config(path=os.path.join('config', 'aml_config.json'))

# ------------------------------------------------------------------------
# Creating Azure Machine learning train dir with relevant scripts

logger.info('Creating Azure Machine learning train dir with relevant scripts')
aml_project_folder = './dist'
environment_file = 'environment.yml'
# check if arguments are valid, i.e. if directory exists
if os.path.exists(aml_project_folder):
    logger.warning('Create directory for AML training: {} already exists, it will be recreated.'.format(aml_project_folder))
    shutil.rmtree(aml_project_folder)
    os.makedirs(aml_project_folder)
else:
    logger.info('Create directors for AML training: {} it will be created.'.format(aml_project_folder))
    os.makedirs(aml_project_folder)

# copy train scripts to 
shutil.copytree(src='./atemteurer', dst=os.path.join(aml_project_folder, 'atemteurer'))
shutil.copyfile(src='./scripts/training/aml_train.py', dst=os.path.join(aml_project_folder, '/aml_train.py'))
shutil.copyfile(src='./data/resmed/train/train_resmed.pt', dst=os.path.join(aml_project_folder, '/train_resmed.pt'))
shutil.copyfile(src='./environment.yml', dst=os.path.join(aml_project_folder, '/environment.yml'))


# ------------------------------------------------------------------------
# Compute target

# choose a name for your cluster
cluster_name = "gpu-cluster"

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target.')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6', 
                                                           max_nodes=4)

    # create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    compute_target.wait_for_completion(show_output=True)

# use get_status() to get a detailed status for the current cluster. 
print(compute_target.get_status().serialize())
# ------------------------------------------------------------------------
# Define experiment

experiment_name = 'SMAVRA'
experiment = Experiment(ws, name=experiment_name)

script_params = {}

estimator = PyTorch(source_directory=aml_project_folder, 
                    script_params=script_params,
                    compute_target=compute_target,
                    entry_script='pytorch_train.py',
                    use_gpu=True,
                    pip_packages=['pillow==5.4.1'],
                    conda_dependencies_file_path='environment.yml'
            )