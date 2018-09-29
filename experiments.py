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
import shutil
from ml_tools import config

import tensorflow as tf

from model_crnn import ModelCRNN

# list of jobs to process.
job_list = []

class Job:
    """ Defines an experiment to process. """
    def __init__(self, name, params):
        self.name = name
        self.params = params

def train_model(run_name, epochs=30.0, limit_training_segments=None, limit_training_tracks=None,
                resample_segments=None, early_stop=False, **kwargs):
    """ Trains a model with the given hyper parameters. """

    logging.basicConfig(level=0)
    tf.logging.set_verbosity(3)

    # a little bit of a pain, the model needs to know how many classes to classify during initialisation,
    # but we don't load the dataset till after that, so we load it here just to count the number of labels...
    dataset_name = os.path.join(config.DATASET_FOLDER, 'datasets.dat')
    dsets = pickle.load(open(dataset_name,'rb'))
    labels = dsets[0].labels

    model = ModelCRNN(labels=len(labels), **kwargs)

    if "segment_length" in kwargs:
        print("Using segment length of {} frames".format(kwargs["segment_length"]))
        model.training_segment_frames = kwargs["segment_length"]
        model.testing_segment_frames = kwargs["segment_length"]

    # setting the seed will make sure that examples included in '100' samples will also be included in '200' samples.
    prev_seed = np.random.seed
    np.random.seed = 20180907
    model.import_dataset(
        dataset_name,
        limit_segments=limit_training_segments,
        limit_tracks=limit_training_tracks,
        resample_segments=resample_segments

    )
    np.random.seed = prev_seed

    model.log_dir = config.TENSORBOARD_LOG_FOLDER
    model.save_epoch_references = False

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
    print('Hyperparameters')
    print("---------------------")
    print("\n".join(["{:<24}{}".format(param, value) for param, value in model.params.items()]))
    print()
    print("Found {0:.1f}K training examples".format(model.rows / 1000))
    print()
    model.train_model(epochs=epochs, run_name=run_name+" "+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), early_stop=early_stop)
    model.save()
    model.close()

    # this shouldn't be necessarily, but unfortunately my model.close isn't cleaning up everything.
    # I think it's because i'm adding everything to the default graph?
    tf.reset_default_graph()

    return model

def has_job(job_name):
    """ Returns if this job has been processed before or not. """

    try:
        f = open(os.path.join(config.EXPERIMENTS_FOLDER,config.TRAINING_RESULTS_FILENAME), "r")
    except:
        return False

    for line in f:
        words = line.split(",")
        job = words[0] if len(words) >= 1 else ""
        if job == job_name:
            f.close()
            return True
    f.close()
    return False

def copy_folder(src, dst, symlinks=False, ignore=None):
    # modified from
    # https://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth

    try:
        os.mkdir(dst)
    except FileExistsError:
        # this is fine... don't look before you leap ;)
        pass

    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def get_lock_name(job_name):
    return os.path.join(config.EXPERIMENTS_FOLDER, "_" + job_name + ".lock")

def create_lock(job_name):
    """ Creates a lock for given job. """
    open(get_lock_name(job_name),'w').close()

def has_lock(job_name):
    return os.path.isfile(get_lock_name(job_name))

def remove_lock(job_name):
    """ Removes lock for given job"""
    try:
        os.remove(get_lock_name(job_name))
    except Exception as e:
        print("Error removing lock, ",e)

def log_job_complete(model, job_name, score,params = None, values = None):

    """ Log reference to job being complete, and move important files to a folder
    :param model The model
    :param job_name The name of the job
    :param score The final evaluation score
    :param params [optional] A list of the parameters used to train this model
    :param values [optional] A corresponding list of the parameter values used to train this model

    """

    f = open(os.path.join(config.EXPERIMENTS_FOLDER,config.TRAINING_RESULTS_FILENAME), "a")
    f.write("{}, {}, {}, {}\n".format(job_name, str(score), params if params is not None else "", values if values is not None else ""))
    f.close()

    remove_lock(job_name)

    # copy completed log to folder
    try:
        copy_folder(os.path.join(config.TENSORBOARD_LOG_FOLDER, model.log_id), os.path.join(config.EXPERIMENTS_FOLDER, job_name))
    except Exception as e:
        print("Error copying job to experiments folder:",e)


def run_job(job_name, **kwargs):
    """ Run a job with given hyper parameters, and log its results. """

    # check if we have done this job...
    if has_job(job_name) or has_lock(job_name):
        return

    create_lock(job_name)
    
    # override default epochs
    if 'epochs' not in kwargs:
        kwargs['epochs'] = 20

    print("-" * 60)
    print("Processing job '{}' with params {}".format(job_name, kwargs))
    print("-" * 60)

    model = train_model("experiments/" + job_name, **kwargs)

    log_job_complete(model, job_name, model.eval_score, list(kwargs.keys()), list(kwargs.values()))


def process_job_list():

    if not os.path.exists(os.path.join(config.EXPERIMENTS_FOLDER,config.TRAINING_RESULTS_FILENAME)):
        open(os.path.join(config.EXPERIMENTS_FOLDER,config.TRAINING_RESULTS_FILENAME), "w").close()

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

    run_job("test", epochs=0.1)

    for retina_size in [1,2,4,8,16,32,48,64,96,128]:
        run_job("retina_size={}-{}".format(retina_size, 1), retina_size=retina_size)


    for filters in [128, 256, 512, 1024, 2048]:
        # reduce batch due to memory constraints with large number of filters.
        run_job("filters={}-{}".format(filters, 1), filters=filters, batch_size=8192//filters)

    # some basic settings
    for run in range(1, 5 + 1):
        run_job("default-{}".format(run))
        run_job("batch_norm=Off-{}".format(run), batch_norm=False)
        run_job("Flow=Off-{}".format(run), enable_flow=False)
        run_job("Augmentation=Off-{}".format(run), augmentation=False)
        run_job("Flow=Off Augmentation=Off-{}".format(run), augmentation=False, enable_flow=False)
        run_job("use_filtered=On-{}".format(run), use_filtered=True)


    for run in range(1, 1 + 1):
        for thermal_threshold in [-100, -50, -20, -10, 0, 10, 20, 50, 100]:
            run_job("use_filtered=On with thermal threshold={}-{}".format(thermal_threshold, run), use_filtered=True, thermal_threshold=thermal_threshold)
        for batch_size in [1, 2, 4, 8, 16, 32, 64]:
            run_job("batch_size={}-{}".format(batch_size, run), batch_size=batch_size)
        for keep_prob in [0.1, 0.2, 0.5, 0.8, 1.0]:
            run_job("keep_prob={}-{}".format(keep_prob, run), keep_prob=keep_prob)
        for thermal_threshold in [-100, -50, -20, -10, 0, 10, 20, 50, 100]:
            run_job("Thermal Threshold={}-{}".format(thermal_threshold, run), thermal_threshold=thermal_threshold)
        for lstm_units in [4,8,16,32,64,128,256,512,1024,2048]:
            run_job("LSTM Units={}-{}".format(lstm_units, run), lstm_units=lstm_units)

    # we take 5000 segments and 500 tracks as full, and so 10 tracks / 100 segments represeents 2%

    segment_counts = [100, 250, 500, 1000, 2500, 5000]
    track_counts =   [10,  25,  50,  100,  250, 500]

    # investigate the effect track count
    for tracks_to_use in track_counts:
        # we want to train each model for the same equiv time.
        run_job("Tracks {}".format(tracks_to_use),   limit_training_tracks=tracks_to_use,
                resample_segments=30000, early_stop=True)

    # investigate the effect segment count
    for segments_to_use in segment_counts:
        # we want to train each model for the same equiv time.
        run_job("Segments {}".format(segments_to_use),   limit_training_segments=segments_to_use,
                resample_segments=30000, early_stop=True)

    # segment length
    for run in range(1,5+1):
        for segment_length in [1,2,4,9,18,27]:
            run_job("Segment Length {}-{}".format(segment_length, run),   segment_length=segment_length)

    # todo: random segment lengths, segment lengths longer than 27 frames...
    # todo: test set error for various lengths, including quite long ones...
    # todo: test where LSTM units become dense layer with averaging accross segment
    # compare avg, max, single frame, and lstm
    # train 10 segments at 1, then 1-2, 1-4, 1-9, 1-18, 1-27
    # single frame without flow
    # no threshold cut, just raw frames? also try standard normalisation
    # background subtracted frames

    # all 8 combinations (flow only etc...)

    # check augmentation vs no augmentation accuracy.

    #  in classify we do frame at a time with decay and ema, test this vs the 27 frames thing, also
    #  training on random lengths is probably important

    # really need to know if training longer segments helps or if it's just evaluating them.
    # also really want some kind of ciriculum with varying length segments (i.e 1-x where x progresses)
    # finally need to check if we should eval once at the end or every step (mayabe a paper on this)
    # I think probably evaulatuting every step gives us a lot more information, but we can't do an update
    # and keep the old state, so maybe just random length.

    # include training time in experiments.txt
    # write a nice notebook to evaulate the experiments

    # add easy support for multiple trials, where all experiments are run trial 1, then trial 2 etc..
    # default to 5 trials for each test.

    # ask richard for a GPU :)

    # thougher testing of effectivness of LSTM units (do we need 2 layers?)
    # thougher testing of effectivness of optical flow... this is problematic
    # thougher testing of normalisation
    # thougher testing of thermal cut

    # train on [
    #    lstm: 1,2,4,9,18,27,45,1-45,1-45 curriculum ]
    #    average: 1,2,4,9,18,27,45,1-45,1-45 curriculum ]
    #    stacked: 1,2,4,9,18,27,45,1-45,1-45 curriculum ]
    # then test on [1,2,4,9,2*9,3*9,5*9] (block and frame at a time (with decay / ema)

    # copy experiments when completed to a seperate folder
    # allow testing of previously trained experiments at certian epochs under given conditions.

    # looks like we get .73 R^2 with log2(examples)~error. with slope 0.022
    # so 2 % reduction in error per doubling of examples.

    # key to pause,
    # key to break

    # a good look into normalisation, maybe just use the simplest one?  i.e. find mean and std
    # a good look into augmentation


if __name__ == "__main__":
    # execute only if run as a script
    main()