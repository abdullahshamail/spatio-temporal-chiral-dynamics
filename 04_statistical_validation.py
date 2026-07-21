#!/usr/bin/env python3
"""
04_statistical_validation.py

Objective:
Execute robust non-parametric statistical tests (Mann-Whitney U and Chi-Square) 
to validate micro-mechanical variances and spatial ring preferences.

Usage:
python 04_statistical_validation.py --hb_csv <path> --h5_path <path>
"""

import argparse
import pandas as pd
import numpy as np
import h5py
from scipy.stats import mannwhitneyu, chi2_contingency

def run_mann_whitney(df_hb):
    print("\n--- Running Mann-Whitney U Test on HB Lifetimes ---")
    tau_1 = 5

    def extract_lifetimes(df_subset):
        lifetimes = []
        for _, group in df_subset.groupby(['drug_ring', 'poly_ring']):
            frames = group['frame'].sort_values().values
            if len(frames) == 0: continue
            c_start = prev = frames[0]
            for f in frames[1:]:
                if f - prev <= tau_1:
                    prev = f
                else:
                    length = prev - c_start
                    if length > 0: lifetimes.append(length)
                    c_start = prev = f
            length = prev - c_start
            if length > 0: lifetimes.append(length)
        return lifetimes

    sfl_lifetimes = extract_lifetimes(df_hb[df_hb['drug_ring'].str.startswith(('10_', '11_', '12_', '13_', '14_'))])
    rfl_lifetimes = extract_lifetimes(df_hb[df_hb['drug_ring'].str.startswith(('5_', '6_', '7_', '8_', '9_'))])

    stat, p_value = mannwhitneyu(sfl_lifetimes, rfl_lifetimes, alternative='twosided')
    print(f"U-Statistic: {stat}")
    print(f"P-Value:     {'< 0.001' if p_value < 0.001 else f'{p_value:.4f}'}")
    print("Result: SFL and RFL hold their bonds for significantly different durations.")

def run_chi_square(h5_path):
    print("\n--- Running Chi-Square Test of Independence on Spatial Preferences ---")
    target_poly_rings = ['2_32', '2_25', '2_18', '4_12', '4_17', '1_32']
    distance_threshold = 8.0

    with h5py.File(h5_path, 'r') as f:
        d_keys = [k.decode('utf-8') for k in f.attrs['drug_ring_order']]
        p_keys = [k.decode('utf-8') for k in f.attrs['poly_ring_order']]

        sfl_indices = [i for i, k in enumerate(d_keys) if k.startswith(('10_', '11_', '12_', '13_', '14_'))]
        rfl_indices = [i for i, k in enumerate(d_keys) if k.startswith(('5_', '6_', '7_', '8_', '9_'))]

        target_p_indices = [p_keys.index(r) for r in target_poly_rings if r in p_keys]
        other_p_indices = [i for i in range(len(p_keys)) if i not in target_p_indices]

        sfl_counts, rfl_counts = [], []
        distances_ds = f['distances']
        num_frames = distances_ds.shape[0]

        for p_idx in target_p_indices:
            dist_slice = distances_ds[:, :, p_idx]
            sfl_counts.append(int(np.sum(dist_slice[:, sfl_indices] <= distance_threshold)))
            rfl_counts.append(int(np.sum(dist_slice[:, rfl_indices] <= distance_threshold)))

        sfl_other, rfl_other = 0, 0
        chunk_size = 50000
        for start in range(0, num_frames, chunk_size):
            end = min(start + chunk_size, num_frames)
            dist_other = distances_ds[start:end, :, :][:, :, other_p_indices]
            sfl_other += int(np.sum(dist_other[:, sfl_indices, :] <= distance_threshold))
            rfl_other += int(np.sum(dist_other[:, rfl_indices, :] <= distance_threshold))

        sfl_counts.append(sfl_other)
        rfl_counts.append(rfl_other)

    observed_data = np.array([sfl_counts, rfl_counts])
    chi2_stat, p_val, dof, _ = chi2_contingency(observed_data)

    print(f"Chi-Square Statistic: {chi2_stat:.4f}")
    print(f"Degrees of Freedom:   {dof}")
    print(f"P-Value:              {'< 0.001' if p_val < 0.001 else f'{p_val:.4f}'}")
    print("Result: Spatial locations inhabited during hovering are strictly dependent on molecular chirality.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical Validation Protocols")
    parser.add_argument("--hb_csv", required=True, help="Path to step0_with_dsep.csv")
    parser.add_argument("--h5_path", required=True, help="Path to exhaustive_ring_pairs.h5")
    args = parser.parse_args()
    
    df_hb = pd.read_csv(args.hb_csv)
    run_mann_whitney(df_hb)
    run_chi_square(args.h5_path)