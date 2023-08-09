import numpy as np
import sys

def main():
    assert len(sys.argv) >= 4
    assert sys.argv[-1][-4:] != ".npy"

    arrays = []
    total_regions = 0
    output = sys.argv[-2]
    region_len = "region" in sys.argv[-1] or \
        "len" in sys.argv[-1]

    max_snps = 0

    for i in range(1, len(sys.argv)-2):
        arr = np.load(sys.argv[i])
        arrays.append(arr)
        total_regions += arr.shape[0]
        
        if region_len and arr.shape[1] > max_snps:
            max_snps = arr.shape[1]

    if region_len:
        # N, max_snps or N,max_snps,
        new_shape = [total_regions] + list(arrays[0].shape[1:])
        new_shape[1] = max_snps
        new_shape = tuple(new_shape)

        combined = np.empty(new_shape)

        pointer = 0
        for arr in arrays:
            arr_size = arr.shape[0]
            next_pointer = pointer + arr_size
            combined[pointer:next_pointer] = center_pad(arr, max_snps)
            pointer = next_pointer
    else:
         # either N,36 or N,36,198
        new_shape = (total_regions,) + (arrays[0].shape[1:])
        combined = np.empty(new_shape)

        pointer = 0
        for arr in arrays:
            arr_size = arr.shape[0]
            next_pointer = pointer + arr_size
            combined[pointer:next_pointer] = arr
            pointer = next_pointer

    np.save(output, combined)

def center_pad(matrix, goal_SNPs):
    num_snps_present = matrix.shape[1]
    if num_snps_present == goal_SNPs:
        return matrix
    assert num_snps_present < goal_SNPs # should not be greater than

    half_snps_present = num_snps_present//2
    if num_snps_present % 2 == 0:
        other_half_snps_present = half_snps_present
    else:
        other_half_snps_present = half_snps_present + 1

    half_S = goal_SNPs//2
    if goal_SNPs % 2 == 0 or (num_snps_present % 2 == 1 and goal_SNPs % 2 == 1) or num_snps_present % 2 == 0:
        other_half_S = half_S
    else:
        other_half_S = half_S+1 # even/odd
            
    if len(matrix.shape) > 2: # includes inds
        new_matrix = np.zeros((matrix.shape[0], goal_SNPs, matrix.shape[2]))
    else:
        new_matrix = np.zeros((matrix.shape[0], goal_SNPs))

    new_matrix[:,half_S - half_snps_present : other_half_S + other_half_snps_present] = matrix

    return new_matrix

# -------------------------------------------------------------------------------
main()
