# disc-pg-gan

This software can be used to train a discriminator to detect selection in real population genetic data, assess the strength of this ability, and make predictions on the possibility of selection in new data.

Python 3.6 (or later) is required, along with the following libraries (these exact versions are likely not necessary, but should be similar):

~~~
msprime==0.7.4
numpy==1.17.2
tensorflow==2.2.0
SLiM==3.6
~~~

See the [msprime documentation](https://msprime.readthedocs.io/en/stable/index.html),
[tensorflow pip guide](https://www.tensorflow.org/install/pip), and
[SLiM installation guide](https://messerlab.org/slim/)
for installation instructions.

Dependencies for creating summary statistic plots and generating `SLiM` data:

~~~
allel==1.2.1
~~~

Link: [scikit-allel](https://scikit-allel.readthedocs.io/en/stable/)

## Training the discriminator

The discriminator is trained using the Mathieson lab's `pg-gan` application: please see the [pg-gan README](https://github.com/mathiesonlab/pg-gan) to understand the use of pg-gan.

By default, `pg-gan` does not save the trained discriminator: the version in this repository enables this option with the `--disc` flag. After this flag, list the name for this run's discrminator. Inside your working directory, a new directory `saved_model/` will be created if it does not already exist, and your discriminator will be saved inside this directory (as a directory itself.) For example, if your original `pg-gan` command would be:

~~~
python3 pg_gan.py -m exp -p N1,N2,growth,T1,T2 -d CHB.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5 -b 20120824_strict_mask.bed
~~~

To save the discriminator, it now becomes:
~~~
python3 pg_gan.py -m exp -p N1,N2,growth,T1,T2 -d CHB.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5 -b 20120824_strict_mask.bed --disc CHB_disc1
~~~

Saving the discrminator does not add any runtime to `pg-gan`, and the training will take several hours to run (likely 5-6 with a GPU, more without one).
Be sure to save the output of this file, for example:
~~~
python3 pg_gan.py ... > CHB_exp_trial1.out
~~~

## Generating SLiM files
One way we validate the ability of the trained discriminator to detect selection is to have it predict on SLiM-generated regions that have been simulated under selection.

A warning, the parameters necessary to simulate most human populations tend to reach the upper limit (or surpass) the computation ability of SLiM, so it is common for such `SLiM` trials to crash.

To generate regions in `SLiM`, be sure that your `SLiM` model matches the model you are simulating with `msprime` in `pg-gan`: matching const and exp models are provided as `const_neutral.slim` and `exp_neutral.slim`. See the aside "Verifying your..." below for more information.

Follow the `pg-gan` instructions to get a set of parameters that match your population. Confirm that these are a good match using `summary_stats.py`, as described in the `pg-gan` README. Edit your SLiM file to use your parameters. For the provided `SLiM` models, these parameters are declared at the beginning of the file.
As mentioned before, it is very possible that `SLiM` will crash under your given parameters. Run `SLiM` once to confirm that it will not crash: for example:
~~~
slim -d outpath=\"sample.trees\" exp_neutral.slim
~~~
See the aside "Reducing SLiM crashing" to resolve this. Likewise, check your selection `.slim` files, under selection strengths of s=0.01, 0.025, 0.05, and 0.10.
The given `exp` model `SLiM` file runs with the following code:
~~~
slim -d outpath=\"sample.trees\" -d sel=0.01 exp_selection.slim
~~~
Where the `sel` argument accepts the selection strength.
Remove the `.trees` files when done.

The python file `make_bases.py` will generate the sets of 100 regions. Run this file with the argument of the SLiM file that you want to run, followed by the prefix with which to save your regions, for example:
~~~
python3 make_bases.py exp_neutral.slim exp_neutrals 0.0
~~~
will produce the output
~~~
exp_neutrals_matrices_100.npy
exp_neutrals_matrices_regions_100.npy
exp_neutrals_distances_100.npy
exp_neutrals_distances_regions_100.npy
~~~

It is important that the suffixes of the files are not changed.
In addition to a set of neutral regions, you want to generate regions under selection strengths of s=0.01, 0.025, 0.05, and 0.10. The third argument of `makes_bases.py` is the chosen selection strength, so run, for example s=0.01:
~~~
python3 make_bases.py exp_selection.slim exp_sel_01 0.01
~~~

It is sufficient to have 100 regions of each selection strength for the seaborn plots.

### Aside: Verifying your SLiM model's accuracy
You can confirm that your SLiM model matches your `msprime` model by generating a small number (say, 100) of regions (see below) with a specific set of params via SLiM, using `summary_stats.py`. Check the main `pg-gan.py` README (linked above) to confirm that you have the right dependencies.
Then, open `global_vars.py` change the variable `OVERWRITE_TRIAL_DATA` to `True`, and change the contents of the `TRIAL_DATA` dictionary to match your model and parameters. Leave `data_h5`, `bed_file`, and `reco_folder` as `None`. Open `summary_stats.py` and change `NUM_TRIALS` to the number of SLiM regions you have (probably 100.)
Run `summary_stats.py` with an default value as the first argument (it will not be used) and the outfile path as the second argument. For example:
~~~
python3 summary_stats.py sample.out test_msprime_slim.png
~~~
Confirm that the summary stats are sufficiently similar. Undo the above changes when done.

### Aside: Reducing SLiM crashing
`SLiM` usually crashes due to the population size becoming too large. Test this resolution by changing your parameter values in SLiM. For the `exp` model, this issue is often the `growth` parameter. In your `pg-gan` file, open `util.py` and lower the selected parameter's upper limit accordingly (under the exp model, try decreasing the upper limit of `growth` from 0.05 to 0.01.) Re-run (several trials of) `pg-gan` to obtain new parameters. We have also decreased the upper limit of `T2`, under the `exp` model, in order to run `SLiM`, for some populations. It is not usually difficult to get paramaters that are a good fit for your population, despite these changes.
It is not necessary to use the `SLiM`-compatible parameters in your discriminator analyses.

## Generating distribution plots
Use `distribution_plot.py` to generate seaborn plots of the predictions of your discriminator on data. The first argument for the program describes what data to the discriminator should predict upon, and the second argument is the path to an outfile from a `pg-gan` trial (`sample.out`), or a `.txt` file containing a newline-separated list of paths of outfiles (as generated by `ls *.out`, etc.) Further information than the discriminator name is necessary for some of the plots, so it is important to provide the entire outfile(s), which will be parse by the program. Under the following first arguments, the program can perform different plots:
- `--real`: Plots the discriminator's predictions on the same data (same file path) it was trained on. Only recommended if there is insufficient test data. For a third argument, please provide the path to a "positive selection list" (see "Aside: The Positive Selection list") corresponding to the train population.
  - usage: `python3 disctribution_plot.py --real pop_A_trial1.out pop_A_pos_sel.txt`
- `--sel`: Plots the discriminator's predictions on `SLiM` data (produced in section "Generating SLiM data") produced under selection strengths of 0.00, 0.001, 0.025, 0.05, and 0.01. After providing the trial outfile as an argument, please provide a `.txt` file listing the paths (as produced by `ls *neutral*.npy`, `ls *sel_01*.npy`, etc.) to the data for a particular selection strength, for each selection strength, in ascending order. The filenames should be as described in "Generating SLiM data".
  - usage: `python3 distribution_plot.py --sel pop_A_trial1.out pop_A_neutrals.txt pop_A_sel_01.txt pop_A_sel_025.txt pop_A_sel_05.txt pop_A_sel_10.txt`
- `--xpop`: Plots the discriminator's predictions on data taken from the given `h5` file. This option is very useful for testing the discriminator's ability to predict on populations other than the training data. As arguments, please provide test population's `h5` file, followed by the test population's positive selection list (see aside.)
  - usage: `python3 distribution_plot.py --xpop pop_A_trial1.out pop_B.h5 pop_B_pos_sel.txt`

### Aside: The positive selection list
The positive selection list should be a text file whose contents is the chromosome number, the start index, and the end index of a section of the genome that is known to be under positive selection for your population, for the given h5 file, comma-separated, with different sections separated by newline characters.
To generate this data, supposed we know that sections M, N, and P are under selection in population A. Open `prediction_utils.py` in your editor and the following to the file's main method:
~~~
iterator = get_iterator("pop_A_trial1.out")

iterator.get_indices(iterator, chrom_M, start_bp_M, end_bp_M)
iterator.get_indices(iterator, chrom_N, start_bp_N, end_bp_N)
iterator.get_indices(iterator, chrom_P, start_bp_P, end_bp_P)
~~~
etc. The `get_indices` function will print the chromosome, start index, and end index for the section in the given h5 file, in the necessary format for `distribution_plot.py` to parse. Now, run this program and save the output to a `.txt` file:
~~~
python3 prediction_utils.py > pop_A_pos_sel_list.txt
~~~

If positive selection data is not available, the indices for the 1000g hg19 h5 files are available for the following populations: YRI, ESN, CHB, CHS, CEU, GBR. Replace the lines in `distribution_plot.py`,
~~~
pos_indices = prediction_utils.load_indices(pos_sel_list)
~~~
with
~~~
pos_indices = pop_indices["lactase"]
~~~
The lactase area is relatively not very large and is not recommended if additional information is available.

## Generating a Manhattan plot
To build the Manhattan plot, we first need to calculate the discriminator's predictions over the genome:

The `genome_disc.py` program accepts the path to a test population's h5 file and bed file, the path to a directory containing discrminators trained in `pg-gan.py`, and the path to an output directory. For each discriminator, regions of the h5 file are iteratively selected, and the discriminator makes a prediction, that is converted from logits to probability. This value is saved to a numpy file inside the output directory.
For purposes of making a Manhattan plot, find the global variable `HIDDEN` and set it to `False`.

Suppose we have discriminators trained on population A, and we want to test on population B, which uses the mask `strict_mask.bed`. We would use the example command:
~~~
python3 genome_disc.py pop_B.h5 strict_mask.bed saved_model/pop_A_discs/ output/correlation/
~~~

After calculating a discriminator's predictions over the test genome, we can generate the Manhattan plot. The program takes arguments of the discriminator's prediction (generated by `genome_disc.py`), the outpath for the manhattan plot, and the title for the plot.
An example command:
~~~
python3 plot_manhattan.py output/correlation/prob_train_pop_A_test_pop_B_disc1.txt manhattan_plots/pop_A/mh_train_pop_A_test_pop_B_disc1.pdf "sample_title"
~~~

## Generating a heatmap
To build the heatmap, we need to calculate some summary statistics over the test genome, in addition to the discriminator's predictions over the genome.

Run `genome_disc.py` as described above, but this time be sure that the global variable `HIDDEN` is set to `True`.

To calculate population summary statistics, the `genome_stats.py` program takes the path to your test population's h5 file, the path to your mask (bed) file, and the output path as arguments --  be sure to name the output file `stats_POPNAME.npy`. The summary statistics are calculated for each region found, and saved to a numpy array. This program uses the import `libsequence`, so conda must be activated for this step (`conda activate`.)

Suppose population B is our test population, and that population B uses the mask `strict_mask.bed`. We would use the command:
~~~
python3 genome_stats.py pop_B.h5 strict_mask.bed output/prediction/stats_B.npy
~~~

Once we have all of the necessary data, we can generate a heatmap. The program `correlation_heatmap.py` accepts the test population summary statistics (produced by `genome_stats.py`), the discrminator's genome predictions (produced by `genome_disc.py`), the output path for the heatmap, and the title for the resulting plot as arguments.

To build a heatmap for the predictions of discriminator "disc1", trained on population A, testing on population B, we would use the command:
~~~
python3 correlation_heatmap.py output/correlation/pop_B_summary_stats.npy output/correlation/hidden_train_pop_A_test_pop_B_disc1.npy figures/discriminator/heatmaps/train_pop_A_test_pop_B_disc1.pdf "sample_title"
~~~
