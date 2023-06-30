'''
For parsing files containing selection data
Formatted for a table from Grossman et al.'s paper,
"Identifying Recent Adaptations..."

Loads the data and allows for lookups.
Because the data is non-overlapping and in genome order,
loading and lookups are very fast

Author: Rebecca Riley
Date 06/12/2023
'''

import numpy as np
import sys

class Sel_Region:
    def __init__(self, name, chrom, start_pos, end_pos):
        self.name = name
        self.chrom = chrom
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.FORMAT = "{}: {}, {}, {}"

    def __str__(self):
        return self.FORMAT.format(self.name, self.chrom, self.start_pos,\
            self.end_pos)

def load_genes(path, pop_name):
    # set up--------------------------------------------------------------------
    gene_data = {}

    gene_table = np.loadtxt(path, delimiter='\t', skiprows=1, dtype=str)
    chrom = lambda row: int(row[1])
    name = lambda row: row[5]
    pop = lambda row: row[7]

    def start_end(row):
        chr_start_end = row[9]
        # get rid of chr section
        start_end_str = chr_start_end[chr_start_end.index(":")+1:]
        start_end_arr = start_end_str.split("-")
        start, end = int(start_end_arr[0]), int(start_end_arr[1])
        return start, end

    # process data!-------------------------------------------------------------
    for gene in gene_table:
        if pop(gene) != pop_name:
            continue

        gene_start, gene_end = start_end(gene)
        gene_chrom = chrom(gene)
        gene_name = name(gene)

        if gene_name == "":
            gene_name = "unknown"

        sel_region = Sel_Region(gene_name, gene_chrom, gene_start, gene_end)

        if gene_chrom in gene_data.keys():
            gene_data[gene_chrom].append(sel_region)
        else:
            gene_data[gene_chrom] = [sel_region]

    return gene_data

def get_sel_genes(region_chrom, region_start, region_end, sel_data_dict):
    gene_not_found = ""

    genes_on_chrom = sel_data_dict[region_chrom] # ordered list

    # linear search
    for gene in genes_on_chrom:
        if region_start > gene.end_pos: # too early
            continue
        if region_end < gene.start_pos:
            return gene_not_found
        # interior
        return gene.name
    return gene_not_found

if __name__ == "__main__":
    path = sys.argv[1]
    pop = sys.argv[2]
    sel_dict = load_genes(path, pop)
