#!/usr/bin/env python3
"""Compare recur87 vs recur88 to find which points changed after inflation.

Usage (from repo root or this folder):
    python mesh/exp05_ds24/compare_recur87_88.py \
        --file_a ruyi_recur87_n_deformed.xyz \
        --file_b ruyi_recur88_n_deformed.xyz \
        --tol 1e-6

Outputs:
    - Prints how many rows differ (expected ~124 with tol=1e-6).
    - Saves differing rows to:
        diff_from_87.txt  (rows from file_a)
        diff_from_88.txt  (rows from file_b)
        diff_pairs.txt    (concat of A row | B row | max_abs_diff)
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_a', type=str, default='ruyi_recur87_n_deformed.xyz', help='baseline file (before inflation)')
    parser.add_argument('--file_b', type=str, default='ruyi_recur88_n_deformed.xyz', help='inflated file (after inflation)')
    parser.add_argument('--tol', type=float, default=1e-6, help='tolerance for considering two rows identical')
    args = parser.parse_args()

    path_a = Path(args.file_a)
    path_b = Path(args.file_b)
    if not path_a.exists() or not path_b.exists():
        raise FileNotFoundError(f"Missing file: {path_a if not path_a.exists() else path_b}")

    a = np.loadtxt(path_a)
    b = np.loadtxt(path_b)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    # max absolute diff per row
    row_max_diff = np.max(np.abs(a - b), axis=1)
    diff_mask = row_max_diff > args.tol
    n_diff = int(diff_mask.sum())
    n_total = a.shape[0]

    print(f"Total rows: {n_total}")
    print(f"Rows differing (> {args.tol}): {n_diff}")

    if n_diff > 0:
        a_diff = a[diff_mask]
        b_diff = b[diff_mask]
        row_max_diff = row_max_diff[diff_mask]

        np.savetxt('diff_from_87.txt', a_diff, fmt='%.8f')
        np.savetxt('diff_from_88.txt', b_diff, fmt='%.8f')
        paired = np.concatenate([a_diff, b_diff, row_max_diff[:, None]], axis=1)
        np.savetxt('diff_pairs.txt', paired, fmt='%.8f')
        print("Saved differing rows to diff_from_87.txt, diff_from_88.txt, diff_pairs.txt")


if __name__ == '__main__':
    main()
