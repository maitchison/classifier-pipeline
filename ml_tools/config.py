"""
Handles configuration file for classifier pipeline.

config settigns can be accessed via the config module.  For example
print(config.DEFAULT_BASE_PATH)

"""

import configparser
import os

# the default ini file created if no ini file exists
DEFAULT_INI = """
[DEFAULT]
; default base path
DEFAULT_BASE_PATH = c:/cac
; folder to put tensor board logs into
TENSORBOARD_LOG_FOLDER = c:/cac/logs/
; dataset folder to use
DATASET_FOLDER = c:/cac/datasets/fantail
; name of the file to write results to.
TRAINING_RESULTS_FILENAME = experiments.txt
; folder to store completed experiments in
EXPERIMENTS_FOLDER = c:/cac/experiments
; folder to save model while it's training.  Make sure this isn't on a dropbox folder and it will cause a crash.
TENSORFLOW_CHECKPOINT_FOLDER = c:/cac/checkpoints

    """

if not os.path.isfile('config.ini'):
    f = open('config.ini','w')
    f.write(DEFAULT_INI)
    f.close()

parser = configparser.ConfigParser()
try:
    parser.read('config.ini')
    DEFAULT_BASE_PATH = parser['DEFAULT']['DEFAULT_BASE_PATH']
    TENSORBOARD_LOG_FOLDER = parser['DEFAULT']['TENSORBOARD_LOG_FOLDER']
    DATASET_FOLDER = parser['DEFAULT']['DATASET_FOLDER']
    TRAINING_RESULTS_FILENAME = parser['DEFAULT']['TRAINING_RESULTS_FILENAME']
    EXPERIMENTS_FOLDER = parser['DEFAULT']['EXPERIMENTS_FOLDER']
    TENSORFLOW_CHECKPOINT_FOLDER = parser['DEFAULT']['TENSORFLOW_CHECKPOINT_FOLDER']
except Exception as e:
    print("Ini file not found or is corrupt.  Please create 'config.ini' in script path.",e)
    exit()
