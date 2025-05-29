# Source Code for the paper "Instance-Optimal Pure Exploration for Linear Bandits on Continuous Arms, ICML 2025"

## Setup
This project is managed by using [rye](https://rye.astral.sh/guide/installation/#updating-rye). First, please install rye and run `rye sync`.

## Modules 
The BAI algorithms (including baselines) are implemented in subgrad_subcicle.py and an optimization method for the fractional quadratic programing is implemented in qfp.py.

## Experiments
To run experiments in the experiments section of the paper, run the following command.

```
    rye run pytest tests/test_lin.py -sv
```

Then results will be stored in the `results` directory as pickle files.
Then you can generate figures in Appendix (that includes experimental results in the experiments section) by running the following command. Then, pdf files will be generated in `results/images` directory.

```
     rye run pytest tests/test_dump_image.py -sv
```

## License
See LICENSE.txt