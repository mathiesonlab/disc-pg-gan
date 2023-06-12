'''
Given either a single file of region data (match_genes) or a set of serveral
region data files produced by different discriminators, find and filter the
gene data corresponding to the given regions, using provided reference files

Author: Rebecca Riley
Date 6/12/23
'''
import json
import optparse

from from_refseq import load_refseq
import selection_data

RANGE = 100000 # None

# helpers ----------------------------------------------------------------------
get_chrom_start_end_pred = lambda row: (row[0], row[1], row[2], row[3])

OUTPUT_FORMAT ="{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n" # extra space for copyable
# COPYABLE = "{}:{}-{}" # the format for ucsc genome browser lookups

# gene maching functions -------------------------------------------------------
'''
Given a set of discriminator prediction files containing the regions whose
prediction score was above a certain threshold, and reference files:

For each region within a prediction file, lookup and record what genes if any
intersect the region (using the given json or refseq references.) Also record
the average prediction value and number of discriminators that found this region.

This data is written to the outpath as a TSV file, with columns for the
chromosome number, start position, end position, average prediction value,
genes found, and the number of discriminators that found the region.
A formatted description of the region is also provided for easy USCS genome
browser lookup.
'''
def consolidate_disc_gene_results(disc_files_path, outpath, optional_json=None,
    optional_refseq=None, optional_sel_data=None):

    disc_files_paths = open(disc_files_path).readlines()

    # Part 1: looking up the genes
    json_dict, refseq_dict, sel_data_dict = get_lookups(optional_json,
        optional_refseq, optional_sel_data, get_pop(disc_files_paths[0]))
    gene_data = {} # a two-dep dictionary with keys chrom, start bp, value is genes
    end_idx = 0
    pred_idx = 1
    gene_idx = 2
    range_gene_idx = 3
    known_sel_idx = 4
    num_disc_idx = 5

    # iterate through every file and check the givne regions
    for disc_path in disc_files_paths:
        disc_reader = open(disc_path[:-1])
        for row in disc_reader:
            # first check if we've already done this lookup:
            # lookups are expensive, only only do them when necessary
            chrom, start_pos, end_pos, pred_value = \
                get_chrom_start_end_pred(row[:-1].split("\t"))
            pred_value = float(pred_value)

            if chrom in gene_data and start_pos in gene_data[chrom]:
                # only update disc count and pred value
                prev_data = gene_data[chrom][start_pos]
                prev_num_discs = prev_data[num_disc_idx]
                num_discs_new = prev_num_discs + 1

                gene_data[chrom][start_pos][pred_idx] = \
                    ((prev_data[pred_idx] * prev_num_discs) + \
                        pred_value) / num_discs_new

                gene_data[chrom][start_pos][num_disc_idx] = num_discs_new
            else:
                # do lookup and store results even if they're blank, to save time on future lookups
                genes, range_genes, sel_genes = region_lookup(chrom, start_pos, end_pos,
                    json_dict, refseq_dict, sel_data_dict)
                data = [end_pos, pred_value, genes, range_genes, sel_genes, 1]
                if chrom in gene_data:
                    gene_data[chrom][start_pos] = data
                else:
                    gene_data[chrom] = {start_pos: data}

        disc_reader.close()

    # Part 2: writing out results
    writer = open(outpath, 'w')
    writer.write(OUTPUT_FORMAT.format("chrom", "start bp", "end pos",
        "avearge pred value", "genes", "range genes", "sel genes", "num discs")) # header

    for chrom in gene_data:
        for start_pos in gene_data[chrom]:
            data = gene_data[chrom][start_pos]

            # optional: ignore unknown or missing genes
            # if data[gene_idx] == "" or data[gene_idx] == "?":
            #     continue

            writer.write(OUTPUT_FORMAT.format(chrom, start_pos, data[end_idx],
                data[pred_idx], data[gene_idx], data[range_gene_idx],
                data[known_sel_idx], data[num_disc_idx]))
    writer.close()

'''
Given a path to a single file, containing data for regions that, with
prediction values averaged across each discriminator, met a certain overall
threshold, and reference files, look up and output the genes that intersect
the given region. Also has the option to output the genes that appear within a
given RANGE of base-pair values.
lookups
The optional reference files can be a json file produced from previous manual
lookups (generate with parse_gene_dict_from_tsv) and a refseq file.
'''
def match_genes(inpath, outpath, optional_json=None, optional_refseq=None,
    optional_sel_data=None):
    # set up optional overlays
    json_dict, refseq_dict, sel_data_dict = get_lookups(optional_json,
        optional_refseq, optional_sel_data, get_pop(inpath))

    # open file, whose columns are chrom, start bp, and end bp
    data_reader = open(inpath, 'r')
    writer = open(outpath, 'w')
    writer.write(OUTPUT_FORMAT.format("chrom", "start bp", "end pos",
        "pred value", "genes", "genes range", "sel genes", "")) # header

    for row in data_reader:
        chrom, start_pos, end_pos, pred_value = \
            get_chrom_start_end_pred(row[:-1].split('\t'))
        genes, genes_range, sel_genes = region_lookup(chrom, start_pos, end_pos,
            json_dict, refseq_dict, sel_data_dict)

        writer.write(OUTPUT_FORMAT.format(chrom, start_pos, end_pos, pred_value,
            genes, genes_range, sel_genes, ""))

    data_reader.close()
    writer.close()

# more complex helper functions-------------------------------------------------
'''
Loads and returns the given reference files, or exits the program if no
reference files are provided
Ideally, use a refseq file that is already sorted in genome order. Otherwise
toggle SORTED = False in from_refseq.py
'''
def get_lookups(optional_json, optional_refseq, optional_sel_data, optional_pop):
    lookup = False
    json_dict = None
    refseq_dict = None

    if optional_json:
        json_dict = json.load(open(optional_json, 'r'))
        lookup = True
    if optional_refseq:
        refseq_dict = load_refseq(optional_refseq)
        lookup = True
    if optional_sel_data and optional_pop:
        sel_data_dict = selection_data.load_genes(optional_sel_data, optional_pop)
        lookup = True

    if not lookup:
        print("no gene lookups were provided, exiting")
        exit()

    return json_dict, refseq_dict, sel_data_dict

'''
Given positional data for a region, finds the names of genes that intersect the
region from the given reference data
O(n) runtime per region
'''
def region_lookup(chrom, start_pos, end_pos, json_dict=None, refseq_dict=None,
    sel_data_dict=None):

    # format list to printable string
    def pretty_format_list_to_str(mylist, result = ""):
        if mylist == []:
            return result

        if result == "":
            result = mylist.pop(0) # start off

        for item in mylist:
            result = result + ", " + item

        return result
    #------------------------------------------------

    genes = ""
    range_genes = ""
    sel_genes = ""
    chrom_int = int(chrom)
    start_pos_int = int(start_pos)
    end_pos_int = int(end_pos)

    # json lookups are O(1)
    if json_dict and chrom in json_dict and start_pos in json_dict[chrom]:
            genes = json_dict[chrom][start_pos] # start off

    # refseq data structure lookups are O(n)
    if refseq_dict and chrom in refseq_dict:

        refseq_genes = refseq_dict[chrom].get(start_pos_int, end_pos_int)
        genes = pretty_format_list_to_str(refseq_genes, genes)

        if RANGE:
            start_pos_int = start_pos_int - RANGE
            end_pos_int = end_pos_int + RANGE
            refseq_genes_range = refseq_dict[chrom].get(start_pos_int, end_pos_int)

            if refseq_genes_range != []:
                range_genes = pretty_format_list_to_str(refseq_genes_range)

    if sel_data_dict and chrom_int in sel_data_dict:
        sel_genes = selection_data.get_sel_genes(chrom_int, start_pos_int,
            end_pos_int, sel_data_dict)

    return genes, range_genes, sel_genes

'''
given a tsv, parses gene data and stores it as a json with chrom and start
position keys and gene name values.
'''
def parse_gene_dict_from_tsv(tsv_path, outpath):
    CHROM_IDX = 0
    START_POS_IDX = 1
    GENE_IDX = 7

    data_reader = open(tsv_path, 'r')

    gene_dict = {}

    for row in data:
        row = row.split('\t')
        gene_name = row[GENE_IDX]
        if gene_name != '':
            chrom = int(row[CHROM_IDX])
            start_pos = int(row[START_POS_IDX])
            if chrom in gene_dict:
                gene_dict[chrom][start_pos] = gene_name
            else:
                gene_dict[chrom] = {start_pos: gene_name}

    writer = open(outpath, 'w')
    json.dump(gene_dict, writer)
    writer.close()

def get_pop(infile):
    for pop in ["YRI", "CHB", "CEU"]:
        if pop in infile:
            return pop

    print("WARNING: POP NOT FOUND")
    return "NONE"

def parse_args():
    parser = optparse.OptionParser(description='match_sig_genes entry point')

    parser.add_option('-i', '--infile', type='string',
        help='text file containing prediction data to process, '+\
        'or a list of data files to process')
    parser.add_option('-o', '--outfile', type='string',
        help='outfile path (tsv)', default=None)
    parser.add_option('-j', '--json', type='string',
        help='json reference file', default=None)
    parser.add_option('-r', '--refseq', type='string',
        help='refseq reference file', default=None)
    parser.add_option('-s', '--sel', type='string',
        help='selection data reference file', default=None)

    (opts, args) = parser.parse_args()

    return opts

if __name__ == "__main__":
    opts = parse_args()
    # match_genes(opts.infile, opts.outfile, opts.json, opts.refseq, opts.sel)
    consolidate_disc_gene_results(opts.infile, opts.outfile, opts.json,
        opts.refseq, opts.sel)
