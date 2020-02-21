import os
import sys
sys.path.append(os.getcwd())

test_mode = False

from scripts.training import train_model
from scripts.training import azure_logger

logger = azure_logger.AzureLogger("smarva")

train = train_model.ModelTrainer(logger)

model = train.run('./train_resmed.pt')