# slim_selection_simulation

This repository is for generating synthetic genetic data, using the Messer Lab's SLiM application, under a flexible demography and with various strengths of selection. The end goal is to have four `.npy` arrays, named `matrices`, `matrices_regions`, `distances`, and `distances_regions` for use in the `slim_iterator` available in the Mathieson Lab's `disc-pg-gan` repository, and can be used in substitution of `msprime`-generated synthetic data or real genetic data. Unlike `msprime`, SLiM is able to simulate selection, and the nature of generating the data in advance and loading it into a data structure results in a constant runtime for lookups.

To generate synthetic data in this repository, please first visit the Messer Lab's website for instructions setting up `SLiM` and `pyslim`. Once setup is complete, follow the following directions:

1. Run `get_recos.py` with arguments of the path to your recombination rate data files, and the path to save a comma-separated list of recombination rates. Adjust the variable COUNT as needed. This should not take more than a few seconds.
For example:
```
python get_recos.py /data/genetic_map/ reco_rates.txt
```

2. Run `bash make_exp_trees.sh` with arguments in the following order: the recombination rates text file, the number of regions to simulate, the parameters `N1,N2,growth,T1,T2` for your SLiM simulation (comma-separated, no spaces), the path to the output folder, the output prefix, the path to the SLiM script, and the strength of selection (unused if no selection). This may take several (or dozens) of hours, so it is recommended that these scripts are run in the background.
For example:
```
bash make_exp_trees.sh 3000 ../reco_rates.txt 23231,29962,0.00531,4870,581 ../output/ YRI_neutral ../slim_scripts/exp_neutral.slim 0.0

bash make_exp_trees.sh 600 ../reco_rates.txt 22552,3313,0.00535,3589,1050 ../output/ CEU_sel01 ../slim_scripts/exp_selection.slim 0.01

bash make_exp_trees.sh 600 ../reco_rates.txt 24609,3481,0.00403,4417,1024 ../output/ CHB_sel05 ../slim_scripts/exp_selection.slim 0.05
```

3. The next step is to run `process_trees.py`, but before that, you must generate a newline-separated list of paths to the trees you want to process. This can be obtained by using `ls` and a redirect: see below for an example. Once your list is generated, call `process_trees.py` with the mandatory arguments: `-i` or `--infile`, is the list of trees you just produced, `-o` (for out) or `--suffix` is the suffix to attach to the aforementioned prefixes before the file type ("YRI_neutral" will produce "matrices_YRI_neutral.npy", etc.), `-r` or `--reco_path` is the list of recombination rates generated in part 1, and `--Ne` (or `-a`) is the ancestral size for recapitation, which should be the same value that you used for `N1` in part 2.

The arguments `--seed` and `--mut` (mutation rate for neutral mutations) are optional.
This process may take up to an hour, depending on the number of regions and complexity of the model (recapitation is slow,) so it is recommended that you run this process in the background.
```
ls output/YRI_neutral*.trees > all_yri_neutral_trees.txt

python process_trees.py -i all_yri_neutral_trees.txt -o YRI_neutral_3000 -r ../reco_rates.txt --Ne 23231

python process_trees.py -i all_CEU_sel01_trees.txt -o CEU_sel01_600 -r ../reco_rates.txt --Ne 22552

python process_trees.py -i all_CHB_sel05_trees.txt -o CHB_sel05_600 -r ../reco_rates.txt --Ne 24600
```

This will output four numpy files: `matrices_SUFFIX.npy`, `matrices_regions_SUFFIX.npy`, `distances_SUFFIX.npy`, and `distances_regions_SUFFIX.npy`, for use in `disc-pg-gan`'s `slim_iterator.py`.
