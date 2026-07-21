#!/usr/bin/env python3
"""
03_hb_lifetime_kinetics.py

Objective:
Calculate continuous hydrogen bond chain lifetimes (L) from raw instance data 
using a permissible state gap tolerance (tau_1 = 5 frames) to evaluate bond persistence.

Usage:
python 03_hb_lifetime_kinetics.py --hb_csv <path> --out_dir <path>
"""

import os
import argparse
import pandas as pd
import numpy as np

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Loading HB data from CSV...")
    df_hb = pd.read_csv(args.hb_csv)
    df_hb.columns = df_hb.columns.str.strip()

    tau_1 = args.tau_1

    def extract_all_lifetimes(df_subset):
        lifetimes = []
        for _, group in df_subset.groupby(['drug_ring', 'poly_ring']):
            frames = group['frame'].sort_values().values
            if len(frames) == 0: 
                continue
            
            c_start = prev = frames[0]
            for f in frames[1:]:
                if f - prev <= tau_1:
                    prev = f
                else:
                    length = prev - c_start
                    if length > 0:
                        lifetimes.append(length)
                    c_start = prev = f
            
            length = prev - c_start
            if length > 0:
                lifetimes.append(length)
                
        return lifetimes

    print("Extracting SFL chain lifetimes...")
    df_sfl_raw = df_hb[df_hb['drug_ring'].str.startswith(('10_', '11_', '12_', '13_', '14_'))]
    sfl_lifetimes = extract_all_lifetimes(df_sfl_raw)

    print("Extracting RFL chain lifetimes...")
    df_rfl_raw = df_hb[df_hb['drug_ring'].str.startswith(('5_', '6_', '7_', '8_', '9_'))]
    rfl_lifetimes = extract_all_lifetimes(df_rfl_raw)

    print(f"\nTotal SFL chains found: {len(sfl_lifetimes)}")
    print(f"Total RFL chains found: {len(rfl_lifetimes)}")

    # Save outputs for reference
    out_csv = os.path.join(args.out_dir, "hydrogen_bond_lifetimes_summary.csv")
    summary_df = pd.DataFrame({
        'Enantiomer': ['SFL']*len(sfl_lifetimes) + ['RFL']*len(rfl_lifetimes),
        'Lifetime_Frames': sfl_lifetimes + rfl_lifetimes
    })
    summary_df.to_csv(out_csv, index=False)
    print(f"Lifetime distributions saved to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Hydrogen Bond Chain Lifetimes")
    parser.add_argument("--hb_csv", required=True, help="Path to step0_with_dsep.csv")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--tau_1", type=int, default=5, help="Continuous chain gap tolerance")
    
    args = parser.parse_args()
    main(args)