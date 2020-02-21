import os
import sys
sys.path.append(os.getcwd())

from scripts.training import train_model
from scripts.training import mlflow_logger

test_mode = True

name = "SMARVA"
if test_mode:
    name = 'SMARVA_Test'

logger = mlflow_logger.MLFlowLogger(name, "smarva", "model", './environment.yml', ['./atemteurer'])

train = train_model.ModelTrainer(logger)
train.test_mode = test_mode

train.device = 'cpu'

model = train.run('data/resmed/train/train_resmed.pt')