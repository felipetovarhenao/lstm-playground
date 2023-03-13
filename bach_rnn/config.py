import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

DATASET_DIR = os.path.join(os.path.dirname(__file__), '../data/bach_inventions')
SEQ_LENGTH = 12
VOCAB_SIZE = 128
BATCH_SIZE = 100
LEARNING_RATE = 5e-3
MIN_DELTA = 0
PATIENCE = 25
EPOCHS = 1000
TEMPERATURE = 1.0
NUM_PREDICTIONS = 30
