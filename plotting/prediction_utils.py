"""
Utilities for measuring discriminator prediction to specific
types of real and simulated genomic regions.
Author: Rebecca Riley
Date: 01/03/2023
"""

# python imports
import math
import numpy as np
import sys

sys.path.insert(1, "../")

# our imports
import global_vars
import discriminator
import real_data_random
import simulation
import util

from generator import Generator
from parse import parse_output
from slim_iterator import SlimIterator

ALT_BATCH_SIZE = 1000
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
    region = real_data_random.Region(chrom, start_base, end_base)
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

    # switch for model value
    model = trial_data['model']
    if model == 'const':
        simulator = simulation.simulate_const
        num_pops = 1
    elif model == 'exp':
        simulator = simulation.simulate_exp
        num_pops = 1
    elif model == 'im':
        simulator = simulation.simulate_im
        num_pops = 2
    elif model == 'ooa2':
        simulator = simulation.simulate_ooa2
        num_pops = 2
    elif model == 'ooa3':
        simulator = simulation.simulate_ooa3
        num_pops = 3
    elif model == 'post_ooa':
        simulator = simulation.postOOA
        num_pops = 2
    else:
        print("could not locate model. An error occured.")
        exit()

    # sample size calculation
    size = int(num_samples/num_pops)
    sample_sizes = [size for i in range(num_pops)]

    generator = Generator(simulator, trial_data['params'].split(','), sample_sizes,\
                          seed=seed, mirror_real=True, reco_folder=trial_data['reco_folder'])

    if param_values is not None:
        generator.update_params(param_values)

    return generator

def get_iterator(trial_data):
    h5 = trial_data['data_h5']
    iterator = real_data_random.RealDataRandomIterator(h5, SEED, trial_data['bed_file'])

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
def load_indices(filepath, pos_all, store=False, output=None):
    lines = open(filepath).readlines()

    if store:
        writer = open(output, 'w')
        bed_formatter = "{}\t{}\t{}\n"

    indices = []
    pos_sel_mask = {}
    for i in global_vars.HUMAN_CHROM_RANGE:
        pos_sel_mask[str(i)] = [] # set up mask dict

    for l in lines:
        if l == "\n":
            continue

        chrom_str, start_str, end_str = l.split(',')
        start_int, end_int = int(start_str), int(end_str)
        section_data = (int(chrom_str), start_int, end_int)
        indices.append(section_data)

        start_bp = pos_all[start_int]
        end_bp = pos_all[end_int]
        pos_sel_mask[chrom_str].append([start_bp, end_bp])

        if store:
            writer.write(bed_formatter.format("chr"+chrom_str, start_bp, end_bp))

    return indices, pos_sel_mask

if __name__ == "__main__":
    infiles = sys.argv[1]
    output = sys.argv[2]
    store_generator_predictions(infiles, output)
