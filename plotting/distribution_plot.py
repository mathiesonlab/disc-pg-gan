"""
Create seaborn plot for discriminator predictions on real data
Author: Rebecca Riley
Date: 03/15/2023
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(1, "../")
from parse import parse_output
import prediction_utils
import real_data_random


# =============================================================================
# COLOR SETTINGS
# =============================================================================
REAL_DATA_COLORS = {"CEU": ["grey", "dodgerblue", "midnightblue", "slateblue"],
                    "GBR": ["grey", "dodgerblue", "midnightblue", "slateblue"],
                    "YRI": ["grey", "yellow", "sienna", "darkorange"],
                    "ESN": ["grey", "yellow", "sienna", "darkorange"],
                    "CHB": ["grey", "limegreen", "darkgreen", "olivedrab"],
                    "CHS": ["grey", "limegreen", "darkgreen", "olivedrab"]}

# =============================================================================
# PLOT UTILS
# =============================================================================
def get_title(title_data):
    return "train: {train_pop}, test: {test_pop}, seed: {seed}".format(\
        train_pop=title_data["train"], test_pop=title_data["test"],
        seed=title_data["seed"])

def save_violin_plot(data, colors, labels, output, title_data, use_pdf):
    RANGE = range(len(colors))

    quantiles = []
    quantile_colors = []
    for i in RANGE:
        quantiles.append([0.05, 0.95])
        quantile_colors.extend([colors[i], colors[i]])

    parts = plt.violinplot(data, RANGE, showmeans=True, showextrema=False, quantiles=quantiles)
        
    for i in RANGE:
        parts["bodies"][i].set_facecolor(colors[i])
    parts["cmeans"].set_color(colors)
    parts["cquantiles"].set_color(quantile_colors)

    # final plotting
    plt.xticks(RANGE, labels)
    plt.ylim([0,1])
    plt.ylabel("discriminator prediction")
    plt.title(get_title(title_data))
    plt.tight_layout()

    if use_pdf:
        plt.savefig(output+"_violin.pdf", format='pdf', dpi=350)
    else:
        plt.savefig(output+"_violin.png", format='png', dpi=300)

    plt.clf() # important

# =============================================================================
# DATA PROCESSING
# =============================================================================
def get_prob(x):
    return 1 / (1 + math.exp(-x))

def process_regions(disc, regions, positions=None):
    preds = disc(regions, training=False).numpy()
    probs = [get_prob(pred[0]) for pred in preds]
    return probs

# =============================================================================
# LOAD FILES
# =============================================================================
'''
Accepts a file containing a list of prepared prediction files 
(see analysis/genome_disc.py), a bed file corresponding to 
positive selection in the test population, and a signal to save the final
as a pdf or a png.
Saves violin representations of the predictions on neutral test data and
test data under selection.
'''
def plot_real(prediction_list, pos_sel_bed, use_pdf):
    pos_sel_mask = real_data_random.read_mask(pos_sel_bed)
    pred_files = open(prediction_list).readlines()

    tokens = pred_files[0].split('.')[-2].split("_")
    train_pop_name = tokens[-2]
    test_pop_name = tokens[-1]

    labels = ["neutral ("+test_pop_name+")", "pos. sel. ("+test_pop_name+")"]
    colors_all = REAL_DATA_COLORS[test_pop_name]
    colors = [colors_all[1], colors_all[3]]
    
    title_data = {"train": train_pop_name, "test": test_pop_name}

    # iterate through each discriminator's predictions
    for pred_file in pred_files:
        name = pred_file.split('.')[-2].split("_")[-4]
        if name[-2].isnumeric():
            seed = name[-2:]
        else:
            seed = name[-1:]

        outfile = train_pop_name+"_"+test_pop_name+"_"+seed
        title_data["seed"] = seed

        predictions = np.loadtxt(pred_file[:-1], delimiter="\t")

        neutrals = []
        sels = []

        for row in predictions:
            chrom, start, end, pred = row[0], row[1], row[2], row[3]
            
            region = real_data_random.Region(int(chrom), start, end)
            if region.inside_mask(pos_sel_mask):
                sels.append(pred)
            else:
                neutrals.append(pred)

        save_violin_plot([neutrals, sels], colors, labels, outfile, title_data, use_pdf)

if __name__ == "__main__":
    prediction_list = sys.argv[1]
    pos_sel_bed = sys.argv[2]

    options = sys.argv[-1]
    use_pdf = ("png" not in options)

    plot_real(prediction_list, pos_sel_bed, use_pdf)

