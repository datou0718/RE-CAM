# RE-CAM

Implementation of **RE-CAM: Readout-Enhanced Content-Addressable Memory for Quantitative Similarity Search**.

Includes hardware-aware readout training and function-approximation experiments,
with SPICE matchline waveforms in `recam/data/10T_06V_CAM.csv`.

## Installation

Requires Python 3.12 or later (tested with 3.12.11).

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Quick start

```sh
recam train --function linear --output results/linear
```

Trains and evaluates RE-CAM for the linear function using the default settings.

## Usage

### Paper experiments

Each command evaluates all four target functions.

> **Runtime warning:** The comparison-budget sweep took about 45 minutes per
> dataset with four functions trained in parallel. Runtime varies by hardware.

```sh
# Table II: readout comparison.
recam benchmark --dataset monte-carlo --output results/monte_carlo
recam benchmark --dataset distribution --output results/distribution

# Table III: comparison budgets 3, 7, and 15.
recam benchmark --experiment comparison-budget --dataset monte-carlo --output results/budget_mc
recam benchmark --experiment comparison-budget --dataset distribution --output results/budget_distribution
```

Table II compares the readout implementations below.
Results are saved in `metrics.csv`; Table III averages are in `overall.csv`.

Verify the paper's NRMSE values and relative changes with
`python tests/check_paper_results.py results`.

Comparison budget 15 and Finer Sensing (budget 7) use staged schedule search.
Use `--search exhaustive` to optimize all feasible schedules (slower).

### System Constraints

| Constraint | Default | Command-line option |
|---|---|---|
| CAM search frequency | 500 MHz | `--search-mhz` |
| System clock rate | 4 GHz | `--clock-ghz` |
| Comparator latency | 70 ps | `--latency-ps` |
| Comparison budget | 7 | `--budget` |

The CAM search period is `1 / search frequency` (2 ns by default). Comparisons
occur at clock ticks before the search ends, separated by at least the comparator
latency.

To change the constraints for an individual experiment:

```sh
recam train --method recam --function linear --dataset monte-carlo \
  --clock-ghz 8 --search-mhz 500 --latency-ps 70 \
  --budget 15 --output results/custom
```

Table II uses a comparison budget of 7, as required by the ADC methods.
Finer Sensing doubles the clock rate with the same budget.
The Table III sweep fixes budgets to 3, 7, and 15 and their clock rates to
2, 4, and 8 GHz, respectively.

### Hardware Nonidealities

| Nonideality | Default | Command-line option |
|---|---|---|
| ML discharge speed variation | 10% | `--discharge-std 0.1` |
| Comparator residual offset STD | 1 mV | `--offset-mv 1` |
| Comparator sensing noise STD | 1 mV | `--noise-mv 1` |

Append these options to `train` or `benchmark`. Discharge variation and residual
offset are fixed within each matchline search; sensing noise varies per comparison.

### Target Functions

Similarity is `x = 32 - Hamming distance`, from 0 to 32. Training minimizes
expected MSE of noisy outputs using FP16 decoding parameters.

| Function | `f(x)` |
|---|---|
| Linear | `x` |
| Piecewise-Linear | `0.5*(x-16)` for `x < 16`; `x-16` otherwise |
| Sigmoid | `1 / (1 + exp(-(x-16)/2))` |
| Exponential | `exp(x/8)` |

### Datasets and Metric

- **Monte Carlo:** 5,000 noisy searches per similarity level; uniform training.
- **Distribution:** Separate sets of 10,000 training and 10,000 validation queries,
  sharing 32 candidates. All vectors are random and 32-bit. Only RE-CAM methods
  use the training similarity distribution.

NRMSE is `sqrt(MSE) / (f(32)-f(0))`, reported in percent. Relative changes compare
against U-ADC in Table II and comparison budget 3 in Table III.

## Readout implementations

| Implementation | `--method` | Description |
|---|---|---|
| U-ADC | `u-adc` | Uniform voltage references with nominal-waveform decoding. |
| RE-CAM(U) | `recam-u` | Uniform voltage references with optimized decoding. |
| NU-ADC | `nu-adc` | Fixed output levels with nominal-waveform reference voltages. |
| RE-CAM(N) | `recam-n` | Optimized references and decoding at one timestamp. |
| RE-CAM | `recam` | Optimized comparison schedule, references, and decoding. |
| Finer Sensing | `finer-sensing` | RE-CAM with twice the clock rate and the same comparison budget. |

## Citation

```bibtex
@misc{RE-CAM,
      title={RE-CAM: Readout-Enhanced Content-Addressable Memory for Quantitative Similarity Search}, 
      author={Yi-Chun Liao and Dongyoung Lee and Jiazheng Chen and Tergel Molom-Ochir and Tomohisa Kawakami and Tania Roy and Ik-Joon Chang and X. Sharon Hu and Hai "Helen" Li and Yiran Chen},
      year={2026},
      eprint={2611.25326},
      archivePrefix={arXiv},
      primaryClass={cs.AR},
      url={https://arxiv.org/abs/2611.25326}, 
}
```

Licensed under the [MIT License](LICENSE).
