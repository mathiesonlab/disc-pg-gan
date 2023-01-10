"""
Utilities for measuring discriminator prediction to specific
types of real and simulated genomic regions.
Author: Rebecca Riley
Date: 01/03/2023
"""

# python imports
import math
import numpy as np
import random
import sys
import tensorflow as tf

sys.path.insert(1, "../")

# our imports
import global_vars
import discriminator
import real_data_random
import simulation
import util

from generator import Generator
from parse import parse_output
from real_data_random import Region

ALT_BATCH_SIZE = 3
SEED = global_vars.DEFAULT_SEED

NEG_1 = True
NUM_SNPS = global_vars.NUM_SNPS

# =============================================================================
# POPULATION DATA
# =============================================================================
CEU_indices = {"HLA": ("6", 4734043, 4786618), "lactase": ("2", 1570677, 1572759)}
CHB_indices = {"HLA": ("6", 4457139, 4512055), "lactase": ("2", 1476913, 1479760)}
YRI_indices = {"HLA": ("6", 7679269, 7735750), "lactase": ("2", 2543963, 2548385)}
GBR_indices = {"HLA": ("6", 4469304, 4522876), "lactase": ("2", 1479264, 1481169)}
CHS_indices = {"HLA": ("6", 4358573, 4410316), "lactase": ("2", 1444582, 1447416)}
ESN_indices = {"HLA": ("6", 7596731, 7653551), "lactase": ("2", 2514619, 2518919)}

POP_INDICES = {"CEU": CEU_indices, "CHB": CHB_indices, "YRI": YRI_indices,
               "GBR": GBR_indices, "CHS": CHS_indices, "ESN": ESN_indices}

# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================
'''Load disc and print its predictions on msprime, random real, HLA, & lactase data'''
def main(generator, iterator, trained_disc, population_indices):

    generated_regions = generator.simulate_batch(batch_size=ALT_BATCH_SIZE)
    probs_sim = process_regions(trained_disc, generated_regions)

    real_regions = iterator.real_batch(batch_size=ALT_BATCH_SIZE)
    probs_real = process_regions(trained_disc, real_regions)

    HLA_regions, HLA_pos = special_section(iterator, population_indices["HLA"])
    probs_HLA = process_regions(trained_disc, HLA_regions, HLA_pos)

    lactase_regions, lct_pos = special_section(iterator, population_indices["lactase"])
    probs_lactase = process_regions(trained_disc, lactase_regions, lct_pos)

    # print("sim\treal\tHLA\tlactase")
    for i in range(ALT_BATCH_SIZE):
        print(str(probs_sim[i])+"\t"+str(probs_real[i])+"\t"+str(probs_HLA[i])+"\t"+str(probs_lactase[i]))

'''
Accepts an iterator (for the selected population), and tuple containing the
chromosome number, and the hap data indices corresponding to the start and end
of the section.
Returns all of the regions in that section in linear order
'''
def special_section(iterator, section_data, max_size=None):
    chrom, section_start_idx, section_end_idx = section_data

    final_end = section_end_idx - NUM_SNPS

    regions_arr = []
    positions_arr = []
    
    i = 0

    for start_idx in range(section_start_idx, final_end, NUM_SNPS):
        result = special_region(chrom, start_idx, iterator)

        if result is None:
            continue

        region, start_pos, end_pos = result
        regions_arr.append(region)
        positions_arr.append([chrom, start_pos, end_pos])
        
        # for testing
        i += 1
        if max_size is not None and i >= max_size:
            break

    print(str(i) + " regions found")

    regions = np.array(regions_arr)
    positions = np.array(positions_arr)
    
    return regions, positions

'''
Copied from real_data_random
Returns the region that starts at the given index
'''
def special_region(chrom, start_idx, iterator):
    chrom = str(chrom) # mask expects a str
    end_idx = start_idx + global_vars.NUM_SNPS

    # get bases to set up region object
    start_base = iterator.pos_all[start_idx]
    end_base = iterator.pos_all[end_idx]
    region = Region(chrom, start_base, end_base)
    inside_mask = region.inside_mask(iterator.mask_dict)

    if not inside_mask:
        return None

    positions = iterator.pos_all[start_idx:end_idx]
    hap_data = iterator.haps_all[start_idx:end_idx, :]
    dist_vec = [0] + [(positions[j+1] - positions[j])/\
                     global_vars.L for j in range(len(positions)-1)]

    after = util.process_gt_dist(hap_data, dist_vec,
                                                  real=True, neg1=True,
                                                  region_len=False)

    return after, region.start_pos, region.end_pos

# =============================================================================
# PROBABILITY FUNCTIONS
# =============================================================================
# simoid
def get_prob(x):
    return 1 / (1 + math.exp(-x))

def process_regions(disc, regions, positions=None):
    preds = disc(regions, training=False).numpy()
    probs = [get_prob(pred[0]) for pred in preds]
    return probs

# =============================================================================
# UTILITES
# =============================================================================
def get_generator(trial_data, num_samples, seed=SEED, param_values=None):

    def get_sample_sizes(num_pops):
        size = int(num_samples/num_pops)
        return [size for i in range(num_pops)]

    # switch for model value
    model = trial_data['model']
    if model == 'const':
        simulator = simulation.simulate_const
        sample_sizes = get_sample_sizes(1)
    elif model == 'exp':
        simulator = simulation.simulate_exp
        sample_sizes = get_sample_sizes(1)
    elif model == 'im':
        simulator = simulation.simulate_im
        sample_sizes = get_sample_sizes(2)
    elif model == 'ooa2':
        simulator = simulation.simulate_ooa2
        sample_sizes = get_sample_sizes(2)
    elif model == 'ooa3':
        simulator = simulation.simulate_ooa3
        sample_sizes = get_sample_sizes(3)
    elif model == 'post_ooa':
        simulator = simulation.postOOA
        sample_sizes = get_sample_sizes(2)
    else:
        print("could not locate model. An error occured.")
        exit()

    generator = Generator(simulator, trial_data['params'].split(','), sample_sizes,\
                          seed=seed, mirror_real=True, reco_folder=trial_data['reco_folder'])

    if param_values is not None:
        generator.update_params(param_values)

    return generator

def get_iterator(trial_data):
    h5 = trial_data['data_h5']
    iterator = real_data_random.RealDataRandomIterator(h5, trial_data['bed_file'], seed=SEED)

    pop_name = None

    for pop in POP_INDICES.keys():
        if pop in h5:
            pop_name = pop
            break

    if pop_name is None:
        print("population not found")
        exit()

    return iterator, pop_name

# =============================================================================
# GET INDICES FUNCTIONS
# =============================================================================
def get_indices(iterator, chrom, start_pos, end_pos):

    start_chrom_idx = 0
    while iterator.chrom_all[start_chrom_idx] != chrom:
        start_chrom_idx += 1

    start_idx = start_chrom_idx
    while iterator.pos_all[start_idx] <= start_pos:
        start_idx += 1

    end_idx = start_idx
    while iterator.pos_all[end_idx] <= end_pos:
        end_idx += 1

    print(chrom, start_idx, end_idx)
    return start_idx, end_idx

'''
For use in loading the extended list for positive selection
Assumes format chrom,start,end
'''
def load_indices(filepath):
    lines = open(filepath).readlines()

    indices = []

    for l in lines:
        if l == "\n":
            continue

        chrom_str, start_str, end_str = l.split(',')
        section_data = (int(chrom_str), int(start_str), int(end_str))
        indices.append(section_data)

    return indices

if __name__ == "__main__":
    final_params, trial_data =  parse_output("../../local-pg-gan/1207/YRI/brooks9_exp_YRI.out")
    iterator, pop_name = get_iterator(trial_data)
    num_samples = iterator.num_samples
    generator = get_generator(trial_data, num_samples)
    generator.update_params(final_params)
    
    trained_disc = tf.saved_model.load("saved_model/" + trial_data["disc"] + "/")
    trained_disc = discriminator.OnePopModel(num_samples, saved_model=trained_disc)
    
    population_indices = POP_INDICES[pop_name]

    main(generator, iterator, trained_disc, population_indices)
