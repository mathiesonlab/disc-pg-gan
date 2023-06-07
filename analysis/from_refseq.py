'''
A data structure for easy inserts and lookups from a given refseq file
For use in quickly determining what genes intersect a large set of genomic regions

Author: Rebecca Riley
Date 6/7/23
'''

import numpy as np
import sys

PRESORTED = True # presorted uses linear search, so it's faster

get_chrom = lambda row: row[3]
get_start = lambda row: int(row[1])
get_end = lambda row: int(row[2])
get_name = lambda row: row[0]

'''
Data is only necessrially in genome order if SORTED = True
a data structure that allows for simple inserts and lookups of gene data
Given a genome region, can return a list of all the genes that intersect that region
'''
class ChromRefSeq:
    def __init__(self, row): # initialize with some data
        self.start_data = [get_start(row)]
        self.end_data = [get_end(row)]
        self.name_data = [get_name(row)]
        self.size = 1

    '''
    Maintains a concurrent lists of start position, end position, and gene name,
    sorted in ascending order of start position. Specific to one chromosome.
    O(1) if presorted and O(n) if not sorted
    '''
    def add(self, row):
        start_bp = get_start(row)

        if PRESORTED:
            idx = self.size
        else:
            idx = self.linear_search(start_bp)

        self.start_data.insert(idx, start_bp)
        self.end_data.insert(idx, get_end(row))
        self.name_data.insert(idx, get_name(row))
        self.size += 1

    '''
    Returns a list of gene names, for every gene that intersects the given
    region.
    Due to linear search, has a runtime of O(n)
    (technically could be more but we know the while loop will only iterate
    a small number of times)
    '''
    def get(self, start_bp, end_bp):
        furthest_idx = self.linear_search(target = end_bp)
        results = []

        if furthest_idx == self.size:
            return results # don't enter loop

        while furthest_idx >= 0 and self.end_data[furthest_idx] > start_bp:
            if self.start_data[furthest_idx] < end_bp:
                name = self.name_data[furthest_idx]
                if name not in results:
                    results.append(name)
            furthest_idx -= 1

        return results

    '''
    binary search just wasn't working
    finds the index of the first start position that is greater than the target
    '''
    def linear_search(self, target):
        size = len(self.start_data)

        for i in range(size):
            if target < self.start_data[i]:
                return i

        return size # last element

'''
given a refsequence file path, builds a chromosome dictionary whose values are
data structures with straightforward gene data inserts and gene data lookups
for a given region
'''
def load_refseq(refseq_path):
    refseq = np.loadtxt(refseq_path, delimiter="\t", skiprows=1, dtype=str)

    chrom_dict = {}

    for row in refseq:
        chrom = get_chrom(row)

        if chrom in chrom_dict:
            chrom_dict[chrom].add(row)
        else:
            chrom_dict[chrom] = ChromRefSeq(row)

    return chrom_dict

'''
does a comma-separated pretty print of a list
'''
def print_genes(gene_list):
    if gene_list == []:
        return ""

    result = gene_list.pop(0)
    for gene in gene_list:
        result = result + ", " + gene

    print(result)

if __name__ == "__main__":
    # helpers specific to infile format
    get_region_chrom = lambda row: row[0] # str
    get_region_start = lambda row: int(row[1])
    get_region_end = lambda row: int(row[2])

    # overlay refseq data onto given infile
    refseq_path = sys.argv[1]
    chrom_dict = load_refseq(refseq_path)

    # validation
    # sample = chrom_dict['1'].start_data
    # assert sample == sorted(sample)

    data_path = sys.argv[2]
    region_data = np.loadtxt(data_path, delimiter="\t", skiprows=1, dtype=str)

    for row in region_data:
        chrom = get_region_chrom(row)

        if chrom in chrom_dict:
            refseq = chrom_dict[chrom]
            genes = refseq.get(get_region_start(row), get_region_end(row))
            print_genes(genes)
