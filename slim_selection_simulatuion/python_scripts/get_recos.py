# Parses recombination rates and stores them in a newline-separated list
# Author: Rebecca Riley
# Date: 04/28/23

import numpy as np
from numpy.random import default_rng
import sys

import global_vars

COUNT=3000

def store_reco_values(reco_folder, outpath):
    # overwite as needed
    files = [reco_folder + "genetic_map_GRCh37_chr" + str(i) +
        ".txt" for i in global_vars.HUMAN_CHROM_RANGE]
    rates, weights = parse_hapmap_empirical_prior(files)

    rng = default_rng(global_vars.DEFAULT_SEED)
    
    writer = open(outpath, 'w')

    for i in range(COUNT):
        rate = rng.choice(rates, p=weights)
        writer.write(str(rate) + ",")
    writer.close()

# from util --------------------------------------------------------------------
def parse_hapmap_empirical_prior(files):
    """
    Parse recombination maps to create a distribution of recombintion rates to
    use for real data simulations. Based on defiNETti software package.
    """
    print("Parsing HapMap recombination rates...")

    # set up weights (probabilities) and reco rates
    weights_all = []
    prior_rates_all = []

    for f in files:
        mat = np.loadtxt(f, skiprows = 1, usecols=(1,2))
        #print(mat.shape)
        mat[:,1] = mat[:,1]*(1.e-8)
        mat = mat[mat[:,1] != 0.0, :] # remove 0s
        weights = mat[1:,0] - mat[:-1,0]
        prior_rates = mat[:-1,1]

        weights_all.extend(weights)
        prior_rates_all.extend(prior_rates)

    # normalize
    prob = weights_all / np.sum(weights_all)

    # make smaller by a factor of 50 (collapse)
    indexes = list(range(len(prior_rates_all)))
    indexes.sort(key=prior_rates_all.__getitem__)

    prior_rates_all = [prior_rates_all[i] for i in indexes]
    prob = [prob[i] for i in indexes]

    new_rates = []
    new_weights = []

    collapse = 50
    for i in range(0,len(prior_rates_all),collapse):
        end = collapse
        if len(prior_rates_all)-i < collapse:
            end = len(prior_rates_all)-i
        new_rates.append(sum(prior_rates_all[i:i+end])/end) # average
        new_weights.append(sum(prob[i:i+end])) # sum
        
    new_rates = np.array(new_rates)
    new_weights = np.array(new_weights)

    return new_rates, new_weights

if __name__ == "__main__":
    reco_path = sys.argv[1]
    outpath = sys.argv[2]
    store_reco_values(reco_path, outpath)