import os
from abc import ABC, abstractmethod
import mlflow
import mlflow.pytorch
import torch
from azureml.core.run import Run


class MLLogger(ABC):
    """Base class for logging ml experiments
    """
    @abstractmethod
    def __init__(self):
        """constructor
        """
        pass

    @abstractmethod
    def log(self):
        """Log a metric of parameter
        """
        pass

    @abstractmethod
    def save_model(self):
        """Save trained model  
        """
        pass


class MLFlowLogger(MLLogger):
    """Logger for MLFlow
    """
    def __init__(self, experiment_name, artifact_path, conda_env, code_paths):
        """Constructor
        
        Arguments:
            experimentName {string} -- name of the experiment
            artifact_path {string} -- path to artifact
            conda_env {string} -- path to conda envrionment file
            code_paths {string} -- path to coda directory
        """
        super(AzureLogger, self).__init__()
        self.experiment_name = experiment_name
        self.artifact_path = artifact_path 
        self.env = conda_env
        self.code_paths = code_paths

        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run()

    def log(self, key, value, step=None):
        """log a metric of parameter
        
        Arguments:
            key {string} -- name of metric of parameter
            value {string} -- value to be logged
        
        Keyword Arguments:
            step {int} -- e.g. which epoch this loss was recorded (default: {None})
        """ 
        if step == None:
            mlflow.log_param(key, value)
        else:
            mlflow.log_param(key, value, step)

    def save_model(self, model, flavor='pytorch'):
        """Save trained model  
        
        Arguments:
            model {pytorch.model} -- model to be saved
            flavor {string} -- flavor of model (pytroch, tensorflow)
        """
        if flavor == 'pytorch':
            mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=self.artifact_path, 
                    conda_env=self.env, 
                    code_paths=self.code_paths
                )
        else:
            NotImplementedError


class AzureLogger(MLLogger):
    """Logger class for azure
    """
    def __init__(self, experiment_name, artifact_path):
        """Constructor
        
        Arguments:
            experimentName {string} -- name of the constructor
            artifact_path {string} -- path to artifact
        """
        super(AzureLogger, self).__init__()
        self.experiment_name = experiment_name
        self.artifact_path = artifact_path

        # start run
        self.run = Run.get_context()


    def log(self, key, value, step=None):
        """Log metric or parameter
        
        Arguments:
            key {string} -- name or metric to be logged
            value {int/string} -- value to be logged    
        
        Keyword Arguments:
            step {int} -- not used (default: {None})
        """
        if step == None:
            self.run.log_param(key, value)


    def save_model(self, model, flavor='pytorch'):
        """Save trained model  
        
        Arguments:
            model {pytorch.model} -- model to be saved
            flavor {string} -- flavor of model (pytroch, tensorflow)
        """
        if flavor == 'pytorch':
            # note that repeated saving could be implemented
            torch.save(model. os.path.join(self.artifact_path, 'model.pt'))
        else:
            NotImplementedError

