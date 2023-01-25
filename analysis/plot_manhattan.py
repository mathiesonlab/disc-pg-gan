"""
Create manhattan-plot-like visualizations of the discriminator predictions.
Authors: Sara Mathieson, Iain Mathieson
Date: 12/14/22
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import sys

################################################################################
# GLOBALS
################################################################################

ODD = "cornflowerblue"
EVEN = "navy"

################################################################################
# HELPERS
################################################################################

def logit(p):
    return math.log(p/(1-p))

def pos(x):
    return x > 0

def filter_one(x):
    return x[:,3] < 1

################################################################################
# MAIN
################################################################################

if __name__ == "__main__":

    # input and output files
    pred_file = sys.argv[1]
    fig_file = sys.argv[2]
    title = sys.argv[3]
    print("pred file", pred_file)
    print("fig file", fig_file)
    print("title", title)

    # chr, start, end, prediction (probabilitiy)
    data = np.loadtxt(pred_file)
    b = len(data)
    data = data[filter_one(data)]
    p = len(data)
    print("filtered", b-p, "rows w/ prob 1")

    # undo probability and convert to zscores
    logits = np.array([logit(x) for x in data[:,3]])

    if np.std(logits) != 0.0:
        zscores = (logits - np.mean(logits))/np.std(logits)
    else:
        zscores = logits - np.mean(logits)

    # switch colors every chrom
    chrom_lst = data[:,0]
    color_lst = np.array([EVEN if c % 2 == 0 else ODD for c in chrom_lst])

    # omit non-positive zscores
    '''mask = pos(zscores)
    zscores_pos = zscores[mask]
    colors_pos = color_lst[mask]
    #bases_pos = bases[mask]'''

    # rescale base positions
    cum_len = 0
    positions = np.zeros((p))
    for i in range(p):
        base = (data[i,1] + data[i,2])/2
        positions[i] = base + cum_len
        # chrom switch or last chrom
        if (i < p-1 and data[i,0] != data[i+1,0]) or i == p-1:
            cum_len += base # approximate length of chrom

    # chromosome ticks
    switches = []
    for i in range(1,int(chrom_lst[-1])+1):
        first_idx = np.where(chrom_lst == i)[0][0]
        switches.append(positions[first_idx])
    switches.append(positions[p-1])
    ticks = []
    for i in range(len(switches)-1):
        ticks.append((switches[i] + switches[i+1])/2)
    #print(ticks, len(ticks))

    # plotting
    plt.figure(figsize=(14, 4))
    plt.scatter(positions, zscores, s=5, c=color_lst)
    plt.xticks(ticks, [str(i) for i in range(1,23)])
    buffer = 1e7
    plt.xlim((min(positions)-buffer, max(positions)+buffer))
    plt.ylim((min(zscores), max(zscores)+1))

    plt.xlabel("chromosome")
    plt.ylabel("logit")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fig_file)
