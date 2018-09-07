"""
    Script to run batch experiments for research paper.
    Uses a job system to work through job list, which could be very long.
"""


"""
Experiments to run:
Check what keep_prob should be
Does L2 Regulatization really not work?
Effect of memory units

Do we need a seperate dense layer?  Looks like I used one in the best performing model...

Looks like L2 regulatization was broken on LSTM units, so try this again...

Question: use standard dataset or the cropped one?  Standard is right I think

Maybe reduce ram a bit, as 10 GB is a bit too much, say by halving the buffers?
"""

#this is needed to render infomational plots during training.
import matplotlib
matplotlib.use('Agg') # enable canvas drawing


import logging
import pickle
import os
import datetime
import numpy as np

import tensorflow as tf

from model_crnn import ModelCRNN_HQ, ModelCRNN_LQ

# folder to put tensor board logs into
LOG_FOLDER = "c:/cac/logs/"

# dataset folder to use
DATASET_FOLDER = "c:/cac/datasets/fantail"

# name of the file to write results to.
RESULTS_FILENAME = "experiments.txt"

# list of jobs to process.
job_list = []

class Job:
    """ Defines an experiment to process. """
    def __init__(self, name, params):
        self.name = name
        self.params = params

def train_model(rum_name, epochs=30.0, limit_training_segments=None, **kwargs):
    """ Trains a model with the given hyper parameters. """

    logging.basicConfig(level=0)
    tf.logging.set_verbosity(3)

    # a little bit of a pain, the model needs to know how many classes to classify during initialisation,
    # but we don't load the dataset till after that, so we load it here just to count the number of labels...
    dataset_name = os.path.join(DATASET_FOLDER, 'datasets.dat')
    dsets = pickle.load(open(dataset_name,'rb'))
    labels = dsets[0].labels

    model = ModelCRNN_HQ(labels=len(labels), **kwargs)

    if limit_training_segments is not None:
        # setting the seed will make sure that examples included in '100' samples will also be included in '200' samples.
        prev_seed = np.random.seed
        np.random.seed = 20180907
        model.limit_training_segments = limit_training_segments
        model.import_dataset(dataset_name)
        np.random.seed = prev_seed
    else:
        model.import_dataset(dataset_name)

    model.log_dir = LOG_FOLDER

    # display the data set summary
    print("Training on labels",labels)
    print()
    print("{:<20} {:<20} {:<20} {:<20} (segments/tracks/bins/weight)".format("label","train","validation","test"))
    for label in labels:
        print("{:<20} {:<20} {:<20} {:<20}".format(
            label,
            "{}/{}/{}/{:.1f}".format(*model.datasets.train.get_counts(label)),
            "{}/{}/{}/{:.1f}".format(*model.datasets.validation.get_counts(label)),
            "{}/{}/{}/{:.1f}".format(*model.datasets.test.get_counts(label)),
        ))
    print()

    for dataset in dsets:
        print(dataset.labels)

    print("Training started")
    print("---------------------")
    print('Hyper parameters')
    print("---------------------")
    print(model.hyperparams_string)
    print()
    print("Found {0:.1f}K training examples".format(model.rows / 1000))
    print()
    model.train_model(epochs=epochs, run_name=rum_name+" "+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    model.save()
    model.close()

    # this shouldn't be nessesary, but unfortunately my model.close isn't cleaning up everything.
    # I think it's because i'm adding everything to the default graph?
    tf.reset_default_graph()

    return model

def has_job(job_name):
    """ Returns if this job has been processed before or not. """
    f = open(RESULTS_FILENAME, "r")
    for line in f:
        words = line.split(",")
        job = words[0] if len(words) >= 1 else ""
        if job == job_name:
            f.close()
            return True
    f.close()
    return False

def log_job_complete(job_name, score,params = None, values = None):

    """ Log reference to job being complete
    :param job_name The name of the job
    :param score The final evaluation score
    :param params [optional] A list of the parameters used to train this model
    :param values [optional] A corresponding list of the parameter values used to train this model

    """
    f = open(RESULTS_FILENAME, "a")
    f.write("{}, {}, {}, {}\n".format(job_name, str(score), params if params is not None else "", values if values is not None else ""))
    f.close()

def run_job(job_name, **kwargs):
    """ Run a job with given hyper parameters, and log its results. """

    # check if we have done this job...
    if has_job(job_name):
        return

    print("-" * 60)
    print("Processing job '{}' with params {}".format(job_name, kwargs))
    print("-" * 60)

    model = train_model("experiments/" + job_name, **kwargs)

    log_job_complete(job_name, model.eval_score, list(kwargs.keys()), list(kwargs.values()))


def process_job_list():

    if not os.path.exists(RESULTS_FILENAME):
        open(RESULTS_FILENAME, "w").close()

    print()
    print("Found {} jobs.".format(len(job_list)))
    print()

    # go through each job and process it.  Would be nice to be able to start a job and pick up where we left of.
    for i in range(5):
        # we repeat the set of jobs 5 times to get an indicator of variation.
        for job in job_list:
            # build the job
            run_job("{}-{}".format(job.name,i+1), **job.params)


def add_job(name, params):
    """
    Adds a job to the list of jobs to process.
    :param name:
    :param params:
    :return:
    """
    job_list.append(Job(name, params))

def main():

    # create our joblist
    # add_job("default",{})
    # process_job_list()
    #run_job("100 Epochs", epochs=100)

    # investigate the effect of data sample count
    # also check track level...
    for i in range(1,20+1):
        segments_to_use = i * 100
        # we want to train each model for the same equiv time.
        equiv_epochs = 20000/segments_to_use
        run_job("Segments {}".format(segments_to_use), epochs=equiv_epochs, limit_training_segments=segments_to_use)




if __name__ == "__main__":
    # execute only if run as a script
    main()