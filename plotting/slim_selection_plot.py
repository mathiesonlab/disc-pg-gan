"""
Create seaborn plot for discriminator predictions on slim data
Author: Rebecca Riley
Date: 03/15/2023
"""

import sys
import tensorflow as tf

sys.path.insert(1, "../")
import distribution_plot
from parse import parse_output
import prediction_utils

ALT_BATCH_SIZE = 1000

SLIM_COLORS = ["grey", "pink", "salmon", "red", "darkred"]


def get_params_trial_data(trial_file):
    if trial_file[-4:] == ".txt": # list of TRIAL FILES containing the data we want
        files = open(trial_file, 'r').readlines()
        params, trial_data = parse_output(files[0][:-1])
    else:
        files = [trial_file+"\n"] # add newline to be discarded, matches list format
        params, trial_data = parse_output(trial_file)
    return params, trial_data, files

'''
Plots and saves discriminator predictions on SLiM data provided by arguments

ARGUMENTS: 5 files whose contents are lists of numpy arrays corresponding to
the selection strengths: neutral (0.0), 0.01, 0.025, 0.05, 0.10
'''
def plot_selection(in_trial_data, files, use_pdf):
    sel_paths = [sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]]
    regions = [SlimIterator(sel_path).real_batch(ALT_BATCH_SIZE) for sel_path in sel_paths]

    colors = SLIM_COLORS
    title_data = {"train": in_trial_data["pop"], "test": "SLiM"}

    for f in files:
        infile = f[:-1]
        params, trial_data = parse_output(infile)
        outfile = infile[:-4]+"_sel"

        title_data["seed"] = trial_data["seed"]
        trained_disc = tf.saved_model.load("saved_model/"+trial_data["disc"]+"/")

        labels = ["neutral", "s=0.01", "s=0.025", "s=0.05", "s=0.10"]
        data = [None for s in labels] # same length

        for i in range(len(regions)):
            data[i] = distribution_plot.process_regions(trained_disc, regions[i])

        assert len(colors) == len(data)

        distribution_plot.save_violin_plot(data, colors, labels, outfile, title_data, use_pdf)

if __name__ == "__main__":
    trial_files = sys.argv[2]
    options = sys.argv[-1]
    use_pdf = ("png" not in options)

    print("arguments: trial files, 5 files whose contents are lists of numpy arrays "+\
            "corresponding to the selection strengths: neutral (0.0), 0.01, "+\
            "0.025, 0.05, 0.10")
    params, trial_data, files = get_params_trial_data(trial_files)
    plot_selection(trial_data, files, use_pdf)