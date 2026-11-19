"""Check the printed accuracy values in Tables II and III after a full run."""
import csv
from pathlib import Path
import sys


FUNCTIONS = ('linear', 'piecewise-linear', 'sigmoid', 'exponential')
METHODS = ('u-adc', 'recam-u', 'nu-adc', 'recam-n', 'recam', 'finer-sensing')
# Entries are NRMSE percent and relative change percent, in paper column order.
TABLE_II = {
    'monte_carlo': (
        ((8.20, 0), (7.92, -3.38), (7.52, -8.28), (6.67, -18.66), (6.90, -15.82), (6.43, -21.50)),
        ((7.16, 0), (6.96, -2.67), (6.54, -8.64), (5.79, -19.11), (5.75, -19.63), (5.59, -21.91)),
        ((12.64, 0), (12.49, -1.14), (9.44, -25.31), (9.20, -27.23), (9.56, -24.34), (8.80, -30.33)),
        ((4.62, 0), (4.57, -.96), (4.79, 3.66), (4.20, -9.04), (4.11, -10.89), (4.01, -13.20)),
    ),
    'distribution': (
        ((8.52, 0), (5.93, -30.45), (7.19, -15.63), (4.97, -41.74), (5.03, -41.03), (4.74, -44.35)),
        ((8.06, 0), (5.62, -30.35), (6.69, -17.07), (4.88, -39.45), (4.91, -39.11), (4.66, -42.25)),
        ((21.28, 0), (17.52, -17.63), (16.52, -22.35), (14.69, -30.94), (15.01, -29.43), (14.22, -33.18)),
        ((5.37, 0), (3.27, -39.08), (5.16, -3.76), (2.90, -45.93), (2.86, -46.73), (2.72, -49.39)),
    ),
}
TABLE_III = {
    'budget_mc': ((9.08, 0), (6.58, -27.48), (5.60, -38.25)),
    'budget_distribution': ((7.90, 0), (6.95, -12.05), (6.54, -17.22)),
}


def check(root):
    comparisons = []
    for dataset, expected in TABLE_II.items():
        with (root/dataset/'metrics.csv').open() as stream:
            rows = {(r['function'], r['method']): r for r in csv.DictReader(stream)}
        for function, values in zip(FUNCTIONS, expected):
            for method, (nrmse, change) in zip(METHODS, values):
                row = rows[function, method]
                comparisons.extend([
                    (f'{dataset}/{function}/{method}: NRMSE', row['nrmse_percent'], nrmse),
                    (f'{dataset}/{function}/{method}: change', row['relative_change_percent'], change),
                ])
    for dataset, expected in TABLE_III.items():
        with (root/dataset/'overall.csv').open() as stream:
            rows = {int(r['comparison_budget']): r for r in csv.DictReader(stream)}
        for budget, (nrmse, change) in zip((3, 7, 15), expected):
            row = rows[budget]
            comparisons.extend([
                (f'{dataset}/{budget}: NRMSE', row['mean_nrmse_percent'], nrmse),
                (f'{dataset}/{budget}: change', row['relative_change_percent'], change),
            ])
    failures = []
    for label, actual, expected in comparisons:
        if f'{float(actual):.2f}' != f'{expected:.2f}':
            failures.append(f'{label}: computed {float(actual):.6f}%, paper {expected:.2f}%')
    for failure in failures:
        print(failure)
    print(f'{len(comparisons)-len(failures)}/{len(comparisons)} accuracy values match the paper to two decimals.')
    return not failures


if __name__ == '__main__':
    try:
        passed = check(Path(sys.argv[1] if len(sys.argv) > 1 else 'results'))
    except (OSError, KeyError, ValueError) as error:
        raise SystemExit(f'Complete all four README experiments before checking results: {error}')
    raise SystemExit(0 if passed else 1)
