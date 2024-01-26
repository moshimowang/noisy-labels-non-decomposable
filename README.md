# noisy-labels-non-decomposable

Code files for Multiclass Learning from Noisy Labels for Non-decomposable Performance Measures (AISTATS 2024).

Our implementations of NCFW and NCBS are based on implementations of FW and BS ([link to code](https://github.com/shivtavker/constrained-classification/tree/master/new_experiments/Unconstraint)).
Different variants of NCLR were implemented based on Patrini et al. (2017) ([link to code](https://github.com/giorgiop/loss-correction)).
See Section 6 of our paper for details.

Due to the size limit for supplementary materials, we cannot include all data.

**Command lines for real data experiments**

- To run BS with Micro F1 loss, use
`python3 bisection-expt-noisy.py 0`

- To run NCBS with Micro F1 loss, use
`python3 bisection-expt-noisy.py 1`

- To run FW with H-mean loss, use
`python3 fwunc-expt-noisy.py 0 h`

- To run NCFW with H-mean loss, use
`python3 fwunc-expt-noisy.py 1 h`

- To run FW with G-mean loss, use
`python3 fwunc-expt-noisy.py 0 g`

- To run NCFW with G-mean loss, use
`python3 fwunc-expt-noisy.py 1 g`

- To run FW with Q-mean loss, use
`python3 fwunc-expt-noisy.py 0 q`

- To run NCFW with Q-mean loss, use
`python3 fwunc-expt-noisy.py 1 q`

- To run different variants of NCLR, use
`python3 noise-corrected-expt.py`


**Instructions for synthetic data experiments**

`calc_Bayes_simu.ipynb` calculates Bayes optimal error.

`run_clean_simu.py` runs algorithms on clean data sets.

`run_noisy_simu.py` runs algorithms on noisy data sets (for five noise matrices).

`show_simu.ipynb` plots the results.