"""
Computes the values of the last hidden layer (or the predictions) of the
discriminator for regions of real data along the genome.
Authors: Rebecca Riley, Sara Mathieson
Date: 12/14/22
"""

# python imports
import math
import numpy as np
import os
import random
import sys
import tensorflow as tf

# our imports
import discriminator
import global_vars
import real_data_random

# globals
#NUM_REGIONS = 1000
NUM_SNPS = global_vars.NUM_SNPS
HIDDEN = False # if True, compute last hidden layer, o.w. compute probability

def get_iterator(input_file, bed_file):
    iterator = real_data_random.RealDataRandomIterator(input_file, bed_file)
    return iterator

# simoid
def get_prob(x):
    return 1 / (1 + math.exp(-x))

def get_pop(h5_filename):
    return h5_filename.split("/")[-1].split(".")[0]

################################################################################
# LAST HIDDEN LAYER
################################################################################

def disc_along_genome(iterator, input_folder, output_file):

    disc = tf.saved_model.load(input_folder)
    disc_recon = discriminator.OnePopModel(iterator.num_samples,
        saved_model=disc)

    # options for discriminator (neg1 should be False for summary stats)
    neg1 = True
    region_len = False
    prev_chrom = None

    # setup output array
    all_regions = []

    # go through entire genome
    final_end = iterator.num_snps-NUM_SNPS
    num_total = 0
    #final_end = NUM_REGIONS*NUM_SNPS # fewer for testing
    for start_idx in range(0, final_end, NUM_SNPS):
        curr_chrom = iterator.chrom_all[start_idx]
        if curr_chrom != prev_chrom:
            print("chrom", curr_chrom, "idx", start_idx)
            prev_chrom = curr_chrom

        # get the region of real data
        region = iterator.real_region(neg1, region_len, start_idx=start_idx)

        # compute hidden layer or probability
        if region is not None:
            corrected = np.zeros((1, iterator.num_samples, NUM_SNPS, 2),
                dtype=np.float32)
            corrected[0] = region

            if HIDDEN:
                hidden_values = disc_recon.last_hidden_layer(corrected)
                all_regions.append(hidden_values.numpy()[0])

            else:
                #pred = disc(corrected, training=False).numpy()
                pred_recon = disc_recon(corrected, training=False).numpy()
                #prob = get_prob(pred)
                prob_recon = get_prob(pred_recon)

                start_base = iterator.pos_all[start_idx]
                end_idx = start_idx + global_vars.NUM_SNPS
                end_base = iterator.pos_all[end_idx]
                all_regions.append([curr_chrom,start_base,end_base,prob_recon])

        num_total += 1

    print("num good regions", len(all_regions), "/", num_total) #NUM_REGIONS)
    if HIDDEN:
        np.save(output_file + ".npy", np.array(all_regions))
    else:
        f = open(output_file + ".txt", 'w')
        for row in all_regions:
            f.write("\t".join([str(x) for x in row]) + "\n")
        f.close()

################################################################################
# MAIN
################################################################################

if __name__ == "__main__":

    h5_filename = sys.argv[1]   # h5 file (i.e. real genomic regions)
    bed_filename = sys.argv[2]  # accessibility mask
    input_folder = sys.argv[3]  # folder of discriminator folders
    output_folder = sys.argv[4] # folder for npy files of hidden values

    disc_folders = sorted(os.listdir(input_folder))

    # get iterator which will return real genomic regions
    iterator = get_iterator(h5_filename, bed_filename)

    # last hidden layer or prediction for all regions
    for saved_model in disc_folders:
        input_file = input_folder + saved_model
        print("input disc", input_file)
        if HIDDEN:
            kw = "hidden_"
        else:
            kw = "prob_"
        pop = get_pop(h5_filename)
        output_file = output_folder + kw + saved_model + "_" + pop
        print("output file", output_file)
        disc_along_genome(iterator, input_file, output_file)
