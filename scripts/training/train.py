import os
import sys
sys.path.append(os.getcwd())
from anomalia.smavra import SMAVRA
from anomalia.datasets import ResmedDatasetEpoch
from anomalia.logging import MLFlowLogger, AzureLogger

test_mode = False
batch_size = 64
train_path = 'data/resmed/train/train_resmed.pt'
experiment_name = 'SMAVRA'
use_cuda = True

smarva_input_params = {
    'input_size':1 if test_mode else 3,
    'hidden_size':10 if test_mode else 30,
    'latent_size':1 if test_mode else 3,
    'attention_size':1 if test_mode else 3,
    'output_size':1 if test_mode else 3,
    'num_layers':1 if test_mode else 2,
    'n_heads':1 if test_mode else 3,
    'dropout':0.25,
    'batch_first':True,
    'cuda': True,
    'mode':'static',
    'rnn_type':'LSTM'
}

smavra = SMAVRA(**smarva_input_params)

if use_cuda:
    smavra.cuda()

logger = MLFlowLogger(experiment_name, "model", './environment.yml', ['./atemteurer'])

dataset = ResmedDatasetEpoch(train_path, batch_size)


#train = model_trainer.ModelTrainer(logger)
train.test_mode = test_mode

train.device = 'cuda'

model = train.run('data/resmed/train/train_resmed.pt')