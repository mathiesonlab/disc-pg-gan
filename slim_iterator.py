"""
Iterates through a given collection of numpy arrays representing SLiM-produced
tree sequences, in a linear order.
Interfaces both the Generator class and the RealDataRandomIterator class.

The SlimIterator accepts a .txt file containing the paths to these numpy array files.
See process_slim.py for information on generating these arrays, and below for
additional usage notes.

Files MUST be named in the pattern:
*matrices*.npy, *matrices_regions*.npy, *distances*.npy, *distances_regions*.npy,
and npy files from the same set should be listed together in the file list
(A_matrices.npy, A_distances.npy, B_matrices.npy, B_distances.npy, etc.)

The "regions" files are optional and only used for summary stats.

Author: Rebecca Riley
Date: 01/05/2023
"""

# python imports
import numpy as np

# our imports
import global_vars
import util

class SlimIterator:

    def __init__(self, file_list):
        self.matrices, self.distances, \
            self.matrices_regions, self.distances_regions = [], [], [], []

        file_names = open(file_list, 'r').readlines()

        for file_name in file_names:
            file_name = file_name[:-1] # get rid of \n

            if "regions" in file_name:
                if "distances" in file_name:
                    self.distances_regions.append(np.load(file_name))
                elif "matrices" in file_name:
                    self.matrices_regions.append(np.load(file_name))
            elif "matrices" in file_name:
                self.matrices.append(np.load(file_name))
            elif "distances" in file_name:
                self.distances.append(np.load(file_name))
            else:
                print("warning: no match for "+file_name)

        num_options = len(self.matrices)
        opt_range = range(num_options)

        self.options = np.array([i for i in opt_range])
        self.num_samples = self.matrices[0].shape[2]

        self.curr_arr_idx, self.curr_idx = 0, 0

    def real_region(self, neg1, region_len=False):

        arr_idx = self.curr_arr_idx
        index = self.curr_idx

        self.increment_indices()

        if region_len:
            gt_matrix = self.matrices_regions[arr_idx][index]
            dist_vec = self.distances_regions[arr_idx][index]

            count=0
            for i in range(len(dist_vec)):
                if dist_vec[i] != 0.0:
                    count += 1

            # print(count)
            gt_matrix, dist_vec = trim_matrix(gt_matrix, dist_vec, count)
        else:
            gt_matrix = self.matrices[arr_idx][index]
            dist_vec = self.distances[arr_idx][index]

        after, success = util.process_gt_dist(gt_matrix, dist_vec, region_len=region_len, neg1=neg1, real=True)

        if not success:
            print("Not success!")
            return self.real_region(neg1, region_len)

        # maybe we could re integrate snp_counts later, but it's not a priority rn
        return after

    def real_batch(self, batch_size=global_vars.BATCH_SIZE, neg1=True, region_len=False):
        """Use region_len=True for fixed region length, not by SNPs"""

        if region_len:
            regions = [] # note: atow, all regions will be same shape, but don't actually have same # of snps due to padding!
            for i in range(batch_size):
                region = self.real_region(neg1=neg1, region_len=region_len)
                regions.append(region)

        else:
            regions = np.zeros((batch_size, self.num_samples, global_vars.NUM_SNPS, 2), dtype=np.float32)

            for i in range(batch_size):
                regions[i] = self.real_region(neg1=neg1, region_len=region_len)

        return regions

    # interface generator too ===================================================================================
    def simulate_batch(self, batch_size=global_vars.BATCH_SIZE, neg1=True, region_len=False):
        return self.real_batch(batch_size=batch_size, neg1=neg1, region_len=region_len)

    def update_params(self, new_params):
        pass

    def increment_indices(self):
        if self.curr_idx == len(self.matrices[self.curr_arr_idx])-1: # last index in arr
            if self.curr_arr_idx == self.options[-1]: # last array
                self.curr_arr_idx = self.options[0]
            else:
                self.curr_arr_idx += 1
            self.curr_idx = 0 # start at the beginning of the array
        else:
            self.curr_idx += 1

def trim_matrix(gt_matrix, dist_vec, goal_snps):
    excess_size = len(dist_vec)

    half_excess = excess_size//2
    half_goal = goal_snps//2
    other_half_excess = half_excess if excess_size%2==0 else half_excess+1 # even/odd
    if goal_snps % 2 == 0 or (excess_size % 2 == 1 and goal_snps % 2 == 1):
        other_half_goal = half_goal
    else:
        other_half_goal = half_goal + 1 # even/odd

    new_matrix = gt_matrix[half_excess - half_goal : other_half_excess + other_half_goal]
    new_dist = dist_vec[half_excess - half_goal : other_half_excess + other_half_goal]

    return new_matrix, new_dist

def trim_matrix2(gt_matrix, dist_vec, goal_SNPs):
    assert type(dist_vec) == type(np.array([]))

    new_matrix = np.zeros((goal_SNPs, global_vars.DEFAULT_SAMPLE_SIZE))
    new_dist = np.zeros((goal_SNPs))

    count = 0
    for i in range(len(dist_vec)):
        if dist_vec[i] != 0.0:
            new_matrix[count] = gt_matrix[i]
            new_dist[count] = dist_vec[i]
            count+=1

    return new_matrix, new_dist

if __name__ == "__main__":
    # testing

    iterator = SlimIterator("FILES.txt")

    iterator.real_batch(batch_size=3, neg1=False, region_len=False)
