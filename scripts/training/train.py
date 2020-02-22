import os
import sys
sys.path.append(os.getcwd())

#from scripts.training
import model_trainer
#from scripts.training 
import mlflow_logger

test_mode = False

name = "SMARVA"
if test_mode:
    name = 'SMARVA_Test'

logger = mlflow_logger.MLFlowLogger(name, "model", './environment.yml', ['./atemteurer'])

train = model_trainer.ModelTrainer(logger)
train.test_mode = test_mode

train.device = 'cuda'

model = train.run('data/resmed/train/train_resmed.pt')