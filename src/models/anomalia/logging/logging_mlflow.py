import mlflow
import mlflow.pytorch

from .logging import MLLogger

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
        self.experiment_name = experiment_name
        self.artifact_path = artifact_path
        self.env = conda_env
        self.code_paths = code_paths

        mlflow.set_experiment(self.experiment_name)

    def start_run(self):
        """Indicate start of run
        """
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
            mlflow.log_metric(key, value, step)
    
    def log_artifact(self, local_path, artifact_path):
        """log artifacts

        Arguments:
            local_path {strinf} -- local path of artifact
            artifact_path {string} -- artifactpath
        """
        mlflow.log_artifact(local_path, artifact_path)

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

    def end_run(self):
        """Indicate end of run
        """
        mlflow.end_run()
