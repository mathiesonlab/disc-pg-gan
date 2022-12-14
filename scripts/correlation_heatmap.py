"""
Compute correlation between summary statistics and last hidden layer.
Author: Sara Mathieson
Date: 12/14/22
"""

# python imports
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import numpy as np
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
import sys

# our imports
import global_vars
import ss_helpers

TICKS = [4.5, 26.5, 51.5, 59.5, 60.5]
LABELS = ['SFS', 'inter-SNP distances', 'LD', '$\pi$', '#haps']

# globals
ABS = False # absolute value
COLOR_MAP = {'YRI': 'PuOr', 'CEU': 'RdBu', 'CHB': 'PiYG', 'ESN': 'PuOr',
    'GBR': 'RdBu', 'CHS': 'PiYG'}

def corr_sum(matrix):
    num_hidden = matrix.shape[1]
    return np.array([sum(matrix[:,j]) for j in range(num_hidden)])

def order_from_children(idx, child_pairs):
    """ from an array of shape (n-1, 2), determine order of columns in
    clustering.... here n=128 """

    n = len(child_pairs)+1
    left_node = child_pairs[idx][0]
    right_node = child_pairs[idx][1]

    # 4 cases (both internal nodes, both leaves, or one of each)
    if left_node >= n and right_node >= n:
        l = order_from_children(left_node-n, child_pairs)
        r = order_from_children(right_node-n, child_pairs)
        return order_from_children(left_node-n, child_pairs) + \
            order_from_children(right_node-n, child_pairs)
    elif left_node < n and right_node >= n:
        return [left_node] + order_from_children(right_node-n, child_pairs)
    elif left_node >= n and right_node < n:
        return order_from_children(left_node-n, child_pairs) + [right_node]
    else:
        return list(child_pairs[idx])

def get_colormap(stats_file):
    pop = stats_file.split("/")[-1].split("_")[1].split(".")[0]
    return COLOR_MAP[pop]

def format_function(tick, tick_pos):
    idx = TICKS.index(tick)
    return LABELS[idx]

def main():
    # input and output files
    stats_file = sys.argv[1]
    hidden_file = sys.argv[2]
    output_file = sys.argv[3]
    title = sys.argv[4]
    print("stats file", stats_file)
    print("hidden file", hidden_file)
    print("output file", output_file)
    print("title", title)

    # colormap
    map = get_colormap(stats_file)

    stats = np.load(stats_file)
    stats = np.delete(stats, 0, axis=1) # remove non-seg sites since 1-pop
    stats = np.delete(stats, 9, axis=1) # remove first inter-SNP (all zeros)

    hidden = np.load(hidden_file)
    print(stats.shape, hidden.shape)
    assert stats.shape[0] == hidden.shape[0]

    num_stats = stats.shape[1]
    num_hidden = hidden.shape[1]

    all_correlations = np.zeros((num_stats, num_hidden))
    not_nan = 0
    #max_corr = 0

    for i in range(num_stats):
        for j in range(num_hidden):
            vec1 = stats[:,i]
            vec2 = hidden[:,j]
            merged = np.vstack((vec1, vec2))
            if ABS:
                corr = abs(np.corrcoef(merged)[0,1]) # doing absolute value
            else:
                corr = np.corrcoef(merged)[0,1]

            if not math.isnan(corr):
                all_correlations[i,j] = corr
                not_nan += 1

                if abs(corr) > 0.35:
                    print("corr", corr, "stat", i, "hidden", j)
                    #max_corr = corr

    print(all_correlations)
    print("frac not nan:", not_nan/(num_stats*num_hidden))

    # sort columns (hidden units) by sum of their correlations
    '''all_cor_sums = corr_sum(all_correlations)
    order = np.argsort(all_cor_sums)[::-1]'''

    # sort using clustering instead
    clustering = AgglomerativeClustering().fit(np.transpose(all_correlations))
    order = order_from_children(-1, clustering.children_) # last pair 2 clusters
    all_correlations_sorted = all_correlations[:, order]

    # plot heatmap
    if ABS:
        sns.heatmap(all_correlations_sorted, vmin=0, vmax=0.5, cmap="Blues")
    else:
        ax = sns.heatmap(all_correlations_sorted, vmin=-0.5, vmax=0.5, cmap=map)

    # tick locations
    ax.yaxis.set_minor_locator(ticker.FixedLocator(TICKS))
    ax.yaxis.set_major_locator(ticker.FixedLocator([0,9,44,59,60,61]))

    # tick labels
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(format_function)

    # Remove the tick lines
    ax.tick_params(axis='y', which='minor', tick1On=False, tick2On=False)

    # align the minor tick label
    for label in ax.get_yticklabels(minor=True):
        label.set_verticalalignment('center')

    # rotate long names and space out last few
    tick_objs = ax.get_yticklabels(minor=True)
    tick_objs[0].set_rotation(90)
    tick_objs[1].set_rotation(90)
    #tick_objs[-2].set_verticalalignment('bottom')
    tick_objs[-1].set_verticalalignment('top')

    plt.title(title)
    plt.tight_layout()
    #plt.show()
    plt.savefig(output_file)

def test_clustering():
    pairs = np.array([[0, 3], [1, 2], [4,5]])
    order = order_from_children(-1, pairs)
    print(order) # should be [0,3,1,2]

if __name__ == "__main__":
    #test_clustering()
    main()
