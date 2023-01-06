"""
Create seaborn plot for discriminator predictions on
(1) SLiM-generated under-selection data
(2) Same population (as training) predictions (msprime, random, balancing/HLA, pos sel.)
(3) Different population predictions ("")
Author: Rebecca Riley
Date: 01/03/2023
"""

import math
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import tensorflow as tf

import global_vars
import discriminator
from parse import parse_output
import pred
from slim_iterator import SlimIterator

SEED = global_vars.DEFAULT_SEED

# =============================================================================
# COLOR SETTINGS
# =============================================================================
REAL_DATA_COLORS = {"CEU": ["grey", "dodgerblue", "midnightblue", "slateblue"],
                    "GBR": ["grey", "dodgerblue", "midnightblue", "slateblue"],
                    "YRI": ["grey", "yellow", "sienna", "darkorange"],
                    "ESN": ["grey", "yellow", "sienna", "darkorange"],
                    "CHB": ["grey", "limegreen", "darkgreen", "olivedrab"],
                    "CHS": ["grey", "limegreen", "darkgreen", "olivedrab"]}

SLIM_COLORS = ["grey", "pink", "salmon", "red", "darkred"]

# =============================================================================
# PLOT UTILS
# =============================================================================
def plot_generic(ax, name, data, colors, labels): # copied from summary_stats.py
    num_datasets = len(data)
    for i in range(num_datasets):
        sns.distplot(data[i], ax=ax, color=colors[i], label=labels[i])
    ax.set(xlabel=name)
    ax.legend()

def save_plot(data, colors, labels, output, title_data):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
    plot_generic(axes, main_label, data, colors, labels)

    plt.title("train: "+title_data["train"]+", test: "+
              title_data["test"]+", seed: "+title_data["seed"])

    plt.tight_layout()
    plt.savefig(output+".pdf", format='pdf', dpi=350)
    #  plt.savefig(output+".png", dpi=300)

# =============================================================================
# DATA PROCESSING
# =============================================================================
def get_prob(x):
    return 1 / (1 + math.exp(-x))

def process_regions(disc, regions, positions=None):
    preds = disc(regions, training=False).numpy()
    probs = [get_prob(pred[0]) for pred in preds]
    return probs

def get_params_trial_data(trial_file):
    if trial_file[-4:] == ".txt": # list of TRIAL FILES containing the data we want
        files = open(trial_file, 'r').readlines()
        params, trial_data = parse_output(files[0][:-1])
    else:
        files = [trial_file+"\n"] # add newline to be discarded, matches list format
        params, trial_data = parse_output(trial_file)
    return params, trial_data, files

'''
Returns an array of regions of different types, on which the trained
discriminator will make predictions
'''
def get_real_data(DATA_RANGE, trial_data, pop_sel_list):
    regions = [None for l in DATA_RANGE]

    generator = pred.get_generator(trial_data,
                                   num_samples=iterator.num_samples,
                                   seed=SEED)
    regions[0] generator.simulate_batch(batch_size=ALT_BATCH_SIZE)

    iterator, pop_name = pred.get_iterator(trial_data)
    regions[1] = iterator.real_batch(batch_size=ALT_BATCH_SIZE)

    pop_indices = pred.POP_INDICES[pop_name]
    regions[2], _ =  pred.special_section(iterator, population_indices["HLA"])

    pos_indices = pred.load_indices(pos_sel_list)
    s_regions = []
    for index in pos_indices:
        s_regions, _ = pred.special_section(iterator, index)
        pos_sel_regions.extend(s_regions)
    regions[3] = s_regions

    return regions, pop_name

# =============================================================================
# LOAD FILES
# =============================================================================
'''
Plots and saves discriminator predictions on real data from the same
population that the discriminator was trained on

ARGUMENTS: pos. sel. list for given population
'''
def plot_real(params, trial_data, files):
    pos_sel_list = sys.argv[3]

    DATA_RANGE = range(4)

    regions, pop_name = get_real_data(DATA_RANGE, trial_data, pop_sel_list)

    labels = ["msprime", pop_name+" (random)", "HLA ("+pop_name+")", \
        "pos. sel. ("+pop_name+")"]

    title_data = {"train": trial_data["pop"], "test": trial_data["pop"]}

    predictions = [None for l in DATA_RANGE]
    COLORS = REAL_DATA_COLORS[pop_name]

    for tf in files:
        infile = tf[:-1] # discard newline char
        outfile = infile[:-4] + "_reals")
        title_data["seed"] = trial_data["seed"]

        trained_disc = tf.saved_model.load("saved_model/" + trial_data["disc"] + "/")

        for i in DATA_RANGE:
            predictions[i] = process_regions(trained_disc, regions[i])

        save_plot(predictions, COLORS, labels, outfile, title_data)

'''
Plots and saves discriminator predictions on SLiM data provided by arguments

ARGUMENTS: 5 files whose contents are lists of numpy arrays corresponding to
the selection strengths: neutral (0.0), 0.01, 0.025, 0.05, 0.10
'''
def plot_selection(trial_data0, files):
    sel_paths = [sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]]
    regions = [SlimIterator(sel_path).real_batch() for sel_path in sel_paths]

    title_data = {"train": trial_data0["pop"], "test": "SLiM"}

    for tf in files:
        params, trial_data = parse_output(tf)
        title_data["seed"] = trial_data["seed"]
        trained_disc = tf.saved_model.load("saved_model/"+trial_data["disc"]+"/")

        labels = ["neutral", "s=0.01", "s=0.025", "s=0.05", "s=0.10"]
        data = [None for s in labels] # same length

        assert len(sel_paths) == len(labels)

        for i in range(len(regions)):
            data[i] = process_regions(regions[i])

        colors = SLIM_COLORS
        assert len(colors) == len(data)

        save_plot(data, colors, labels, outfile, title_data)

'''
Accepts an h5 and file/list of files corresponding to discriminators to test on
the given h5 file

ARGUMENTS: path to xpop h5 file, and pos. sel. list for xpop
'''
def plot_cross_disc(in_trial_data, files):
    # setup ====================================================================
    h5 = sys.argv[3]
    pos_sel_list = sys.argv[4]

    test_pop_name = h5.split("/")[6][0:3]

    test_trial_data = in_trial_data.copy()
    test_trial_data["h5"] = h5

    title_data = {"train": in_trial_data["pop"], "test": test_pop_name}

    pos_indices = pred.load_indices(pos_sel_list)
    regions, pop_name = get_real_data(DATA_RANGE, test_trial_data, pop_sel_list)

    # iterate through discriminator list or load trial file ====================
    for tf in files:
        trial_file = tf[:-1] # get rid of newline char
        params, train_trial_data = parse_output(trial_file)
        disc_name = train_trial_data["disc"]

        title_data["seed"] = train_trial_data["seed"]
        outfile = "xpop_"+test_pop_name+"_"+disc_name
        generator.update_params(params)

        trained_disc = tf.saved_model.load("saved_model/" + disc_name + "/")

        for i in DATA_RANGE:
            predictions[i] = process_regions(trained_disc, regions[i])

        save_plot(data, colors, labels, outfile, title_data)

if __name__ == "__main__":
    type = sys.argv[1] # real,sel, or xpop,
    trial_files = sys.argv[2]
    params, trial_data, files = get_params_trial_data(trial_files)

    if "real" in type:
        plot_real(params, trial_data, files)
    elif "sel" in type:
        print("arguments: 5 files whose contents are lists of numpy arrays "+\
            +"corresponding to the selection strengths: neutral (0.0), 0.01, "+\
            +"0.025, 0.05, 0.10")
        plot_selection(files)
    elif "xpop" in type:
        print("arguments: trial (train) files, xpop h5, and xpop pos. sel. file")
        plot_cross_disc(trial_data, files)
    else:
        print("please provide \"real\", \"sel(ection)\", or \"xpop\" as the"+\
            +" first argument, and trial_file or .txt containing a list of "+\
            +"trial files as the second argument")
        exit()
