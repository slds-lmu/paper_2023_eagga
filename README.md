In this repository we release all code to replicate all results, tables and figures presented in the paper: Multi-Objective Optimization of Performance and Interpretability of Tabular Supervised Machine Learning Models

**Note that we also offer a standalone R package for the EAGGA implementation, which can be found at https://github.com/sumny/eagga.
While this repository here is primarily intended for replicating experiments, generating figures, and reproducing tables as reported in the paper, we strongly encourage utilizing https://github.com/sumny/eagga for practical use and research purposes.
This alternate repository is actively maintained, provides enhanced usability and features, and undergoes continuous testing.**

The repository is structured as follows:

  * The root directory resembles an R package containing the implementation of EAGGA at the time point of writing the paper and running benchmark experiments
    * The most important directory is `R/` containing the source code
    * Content of files within `R/` should be self-explanatory due to their naming.
      The EAGGA implementation can be found in `TunerEAGGA.R`
  * `attic/benchmarks` contains all code needed to rerun benchmarks and generate figures and reproduce tables as reported in the paper
    * `plots/` contains all figures included in the extended version of the paper
    * `results/` contains aggregated results of the benchmark experiments and ablation studies used to generate figures and tables.
       If you are interested in the raw results in the form of `batchtools` registries, please open an issue
    * `run_ours_so.R` and `run_ablation.R` contain code to run the actual benchmark experiments and ablation studies via `batchtools`
    * `analyze_ours_so.R` and `analyze_ablation.R` contain all code to analyze the aggregated `results/` and generate figures and tables
    * `helpers.R` contains helper functions and code used within the main routines
    * `majority_vote.R` contains code to generate the majority vote baseline on each task
    * `openml_tasks.R` contains code to generate an overview of the `OpenML` tasks used within benchmark experiments and ablation studies
    * `search_spaces.R` contains helper code describing the search space of each learner used within benchmark experiments and ablation studies

In the benchmark experiments and ablation studies, R 4.2.2 was used.
`batchtools` was used to run experiments on an internal HPC cluster equipped with Intel Xeon E5-2670 instances.

`renv.lock` provides an `renv` lock file that lists all packages and their versions used throughout all benchmark experiments and ablation studies.
The EBM was used as wrapped via https://github.com/sumny/EBmlr3 relying on python 3.8 and `interpret` 0.2.7.

To replicate our results, install R 4.2.2 and python 3.8 and proceed to install all required packages.

```r
install.packages("renv")  # 0.16.0
install.packages("devtools")  # 2.4.2
renv::restore(lockfile = "renv.lock")
remotes::install_local("", upgrade = "never")  # install the eagga package
```

Setting up `EBmlr3` is described here: https://github.com/sumny/EBmlr3

Then run the benchmarks and ablation studies as described in `attic/benchmarks/run_ours_so.R` and `attic/benchmarks/run_ablation.R`
Make sure to adapt some paths when constructing `batchtools` registries or saving results, i.e., in `run_ours_so.R` and `run_ablation.R`.
We indicate this via `"FIXME_path"` as a placeholder.

Note that `eagga.pdf` is an extended version of the paper containing supplementary material and additional analyses.

