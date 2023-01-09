import msprime
import numpy as np
import pyslim
import subprocess
import tskit
import sys

from numpy.random import default_rng

import global_vars

# params = {"Ne": 10000., "reco": 1.25e-8, "mut": 1.25e-8}
# NEED TO USE EXP NE=N1
params = {"Ne": 22552., "mut": 1.25e-8,
          "reco_path": "/bigdata/smathieson/pg-gan/1000g/genetic_map/"}

SEED = global_vars.DEFAULT_SEED

num_haps = 198
num_inds = num_haps//2 # dihaploid

n_regions = 3

n_range = range(n_regions)

SLIM_FILE = sys.argv[1]
SUFFIX = sys.argv[2]
SEL = sys.argv[3]

def main():

    # reco setupt---------------------------------------------
    files = global_vars.get_reco_files(params["reco_path"])
    prior, weights = global_vars.parse_hapmap_empirical_prior(files)
    rng = default_rng(SEED)
    def get_reco():
        # draw_background_rate_from_prior
        return rng.choice(prior, p=weights)
    # --------------------------------------------------------
    
    # (n, 36, 198)
    matrices = np.zeros((n_regions, global_vars.NUM_SNPS, num_haps))
    distances = np.zeros((n_regions, global_vars.NUM_SNPS))
    
    matrices_list = [None for i in n_range]
    distances_list = [None for i in n_range]

    max_SNPs = -1

    use_selection = False
            
    # for file_name in files:
    for i in n_range:

        # make file in process -- will this fix our previous issues?
        # outpath = "exp_sel_"+sel_strength+"_"+str(i)+".trees"
        outpath = SUFFIX+"_"+str(i)+".trees"
        
        reco = str(get_reco())
        subprocess.check_output(["slim", "-d", "outpath=\""+outpath+"\"", "-d", "reco="+reco, "-d", "sel="+SEL, SLIM_FILE])

        ts = None
                
        while not ts:
            try:
                ts = pyslim.update(tskit.load(outpath))
            except:
                print("an error occured while loading the tree sequence. Be sure that conda is activated. Trying again...")
                subprocess.check_output(["slim", "-d", "outpath=\""+outpath+"\"", "-d", "reco="+reco, "-d", "sel="+SEL, SLIM_FILE])

        ts = clean_tree(ts, use_selection, reco)
        subprocess.check_output(["rm", outpath])
        
        gt_matrix = ts.genotype_matrix()
        num_snps_present = gt_matrix.shape[0]
        dist_vec = get_dist_vec(ts, num_snps_present)

        print(str(i) + " " + str(num_snps_present))
                
        # store for region_len first
        matrices_list[i] = gt_matrix
        distances_list[i] = dist_vec        

        if max_SNPs < num_snps_present:
            max_SNPs = num_snps_present
            
        # now trim/pad
        if num_snps_present < global_vars.NUM_SNPS:
            gt_matrix, dist_vec = center_pad(gt_matrix, dist_vec, num_snps_present, global_vars.NUM_SNPS)
        elif num_snps_present > global_vars.NUM_SNPS:
            gt_matrix, dist_vec = trim_matrix(gt_matrix, dist_vec, num_snps_present)
        
        matrices[i] = gt_matrix
        distances[i] = dist_vec

    # finish regions_len section ----------------------------
    assert len(matrices_list) == n_regions
    print("seting up matrices: max SNPs = " + str(max_SNPs) + ", num matrices = " + str(n_regions))
    
    matrices_regions = np.zeros((n_regions, max_SNPs, num_haps))
    distances_regions = np.zeros((n_regions, max_SNPs))

    for j in n_range:
        print(j)
        gt_matrix = matrices_list[j]
        dist_vec = distances_list[j]
        num_snps_present = gt_matrix.shape[0]

        assert num_snps_present <= max_SNPs

        if num_snps_present < max_SNPs:
            gt_matrix, dist_vec = center_pad(gt_matrix, dist_vec, num_snps_present, max_SNPs)

        matrices_regions[j] = gt_matrix
        distances_regions[j] = dist_vec

    np.save("matrices_"+SUFFIX, matrices)
    np.save("distances_"+SUFFIX, distances)
    
    np.save("matrices_regions_"+SUFFIX, matrices_regions)
    np.save("distances_regions_"+SUFFIX, distances_regions)    
    
'''
necessary step. Do not remove.
'''
def clean_tree(ts, selection, reco):
    ts_recap = pyslim.recapitate(ts, recombination_rate=reco,
                                 ancestral_Ne=params["Ne"])

    if selection:
        ts_simplified = simplify_selection_tree(ts_recap)
    else:
        ts_simplified = simplify_tree(ts_recap)
    ts_mutated = msprime.mutate(ts_simplified, rate=params["mut"], keep=True)
    return ts_mutated

'''
part of above necessary step
'''
def simplify_tree(ts_recap):   
    # alive_inds = ts_recap.individuals_alive_at(0)
    alive_inds = np.array([i for i in range(ts_recap.num_individuals)])
    keep_inds = np.random.choice(alive_inds, num_inds, replace=False)
    keep_nodes = []

    for i in keep_inds:
        keep_nodes.extend(ts_recap.individual(i).nodes)

    ts_simplified = ts_recap.simplify(keep_nodes)
    return ts_simplified

def simplify_selection_tree(ts_recap):
    tries = 0
    num_muts = 0
    
    while num_muts < 1:
        tries += 1
        ts_simplified = simplify_tree(ts_recap)
        num_muts = ts_simplified.genotype_matrix().shape[0]
        # print(tries)

    return ts_simplified

def trim_matrix(gt_matrix, dist_vec, num_snps_present):
    assert type(dist_vec) == type(np.array([]))

    half_snps_present = num_snps_present//2
    half_S = global_vars.NUM_SNPS//2

    new_matrix = gt_matrix[half_snps_present - half_S : half_snps_present + half_S]
    # note: we DON'T adjust the distance vector
    new_dist = dist_vec[half_snps_present - half_S : half_snps_present + half_S]
    
    return new_matrix, new_dist

def get_dist_vec(ts, snps_total):
    positions = [round(variant.site.position) for variant in ts.variants()]
    assert len(positions) == snps_total
    
    dist_vec = [0] + [(positions[j+1] - positions[j])/ \
              global_vars.L for j in range(snps_total-1)]
    return np.array(dist_vec)
    
def center_pad(gt_matrix, dist_vec, num_snps_present, S):
    inds = gt_matrix.shape[1]
    half_snps_present = num_snps_present//2
    if num_snps_present % 2 == 0:
        other_half_snps_present = half_snps_present
    else:
        other_half_snps_present = half_snps_present + 1

    half_S = S//2
    if S % 2 == 0 or (num_snps_present % 2 == 1 and S % 2 == 1) or num_snps_present % 2 == 0:
        other_half_S = half_S
    else:
        other_half_S = half_S+1 # even/odd
            

    new_matrix = np.zeros((S, inds))
    new_dist = np.zeros((S))

    new_matrix[half_S - half_snps_present : other_half_S + other_half_snps_present] = gt_matrix
    new_dist[half_S - half_snps_present : other_half_S + other_half_snps_present] = dist_vec

    return new_matrix, new_dist


# RECO ==========================================================================================
def draw_background_rate_from_prior(prior_rates, prob):
    return np.random.choice(prior_rates, p=prob)

if __name__ == "__main__":
    main()
