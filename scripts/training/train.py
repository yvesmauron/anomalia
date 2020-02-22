import os
import sys
sys.path.append(os.getcwd())
from anomalia.smavra import SMAVRA
from anomalia.datasets import ResmedDatasetEpoch, TestDataset
from anomalia.logging import MLFlowLogger, AzureLogger
from anomalia.smavra_trainer import SmavraTrainer
import torch

# --------------------------------------------------------
# set globals
TEST_MODE = False
BATCH_SIZE = 64
N_EPOCHS = 400
TRAIN_PATH = 'data/resmed/train/train_resmed.pt'
EXPERIMENT_NAME = 'SMAVRA'
USE_CUDA = True

# --------------------------------------------------------
# define learner
smarva_input_params = {
    'input_size':1 if TEST_MODE else 3,
    'hidden_size':10 if TEST_MODE else 30,
    'latent_size':1 if TEST_MODE else 3,
    'attention_size':1 if TEST_MODE else 3,
    'output_size':1 if TEST_MODE else 3,
    'num_layers':1 if TEST_MODE else 2,
    'n_heads':1 if TEST_MODE else 3,
    'dropout':0.25,
    'batch_first':True,
    'cuda': USE_CUDA,
    'mode':'static',
    'rnn_type':'LSTM'
}

smavra = SMAVRA(**smarva_input_params)

if USE_CUDA:
    smavra.cuda()

# --------------------------------------------------------
# define dataset

if not TEST_MODE:
    dataset = ResmedDatasetEpoch(TRAIN_PATH, BATCH_SIZE)
else:
    dataset = TestDataset(200, 200)

# --------------------------------------------------------
# define logger
logger = MLFlowLogger(EXPERIMENT_NAME, "model", './environment.yml', ['./anomalia'])

# --------------------------------------------------------
# define optimizer
lr=0.0005
optimizer = torch.optim.Adam(smavra.parameters(), lr=lr)

# --------------------------------------------------------
# let the trainer take care of training
trainer = SmavraTrainer(model=smavra, dataset=dataset, optimizer=optimizer, logger=logger)

# --------------------------------------------------------
# Start run including logging
logger.start_run()
logger.log('lr', lr) # think about a better way to do this
trainer.fit(n_epochs=N_EPOCHS, batch_size=BATCH_SIZE)
# start run
logger.end_run()