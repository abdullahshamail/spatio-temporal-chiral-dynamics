#!/usr/bin/env python3
"""
Usage:
python 02_spatiotemporal_sliding_window.py --h5_path <path> --hb_success <path> --out_dir <path>
"""

import os
import argparse
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

def smooth_states(states, tau):
    """Bridges short state reversals based on state gap tolerance tau (Tau_3)."""
    s = list(states)
    for i in range(1, len(s) - tau):
        for gap in range(1, tau + 1):
            if s[i-1] == s[i+gap] and s[i] != s[i-1]:
                for j in range(i, i+gap):
                    s[j] = s[i-1]
    return s

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    output_file = os.path.join(args.out_dir, "master_predictive_log.csv")

    # Molecule Mapping
    SFL_MOLS = range(10, 15)
    RFL_MOLS = range(5, 10)
    criteria = {
        'SFL': {'dist': (4.0, 6.0), 'theta': (140, 170)},
        'RFL': {'dist': (6.0, 9.0), 'theta': (30, 70)}
    }

    print("Loading HB Success Data...")
    df_hb = pd.read_csv(args.hb_success)
    df_hb.columns = df_hb.columns.str.strip()

    hb_lookup = set(zip(
        df_hb['drug_ring'].astype(str).str.strip(),
        df_hb['poly_ring'].astype(str).str.strip(),
        df_hb['frame'].astype(int)
    ))

    print("Opening Exhaustive HDF5 Database for Spatio-temporal Processing...")
    with h5py.File(args.h5_path, 'r') as f:
        drug_keys = [k.decode('utf-8') for k in f.attrs['drug_ring_order']]
        poly_attr = 'polymer_ring_order' if 'polymer_ring_order' in f.attrs else 'poly_ring_order'
        poly_keys = [k.decode('utf-8') for k in f.attrs[poly_attr]]
        
        event_log = []

        for d_idx, d_key in enumerate(tqdm(drug_keys, desc="Processing Drug Rings")):
            full_drug_id = d_key 
            mol_idx = int(d_key.split('_')[0])
            ring_type = "Ring 1 (A-Ring)" if d_key.endswith('_1') else "Ring 0 (B-Ring)"
            enant = 'SFL' if mol_idx in SFL_MOLS else 'RFL'
            
            d_min, d_max = criteria[enant]['dist']
            a_min, a_max = criteria[enant]['theta']

            dists = f['distances'][:, d_idx, :]
            angles = f['angles'][:, d_idx, :]

            for p_idx, p_key in enumerate(poly_keys):
                full_poly_id = p_key
                
                mask = (dists[:, p_idx] >= d_min) & (dists[:, p_idx] <= d_max) & \
                       (angles[:, p_idx] >= a_min) & (angles[:, p_idx] <= a_max)
                indices = np.where(mask)[0]
                if len(indices) < args.window_size: 
                    continue

                diffs = np.diff(indices)
                breaks = np.where(diffs > args.session_gap)[0] + 1
                sessions = np.split(indices, breaks)

                for frames in sessions:
                    if len(frames) < args.window_size: 
                        continue
                    
                    has_bond = any((full_drug_id, full_poly_id, t) in hb_lookup for t in frames)
                    
                    raw = []
                    for i in range(args.window_size, len(frames)):
                        delta = dists[frames[i], p_idx] - dists[frames[i - args.window_size], p_idx]
                        if delta < -args.epsilon: 
                            raw.append('s_towards')
                        elif delta > args.epsilon: 
                            raw.append('s_away')
                        else: 
                            raw.append('s_along')
                    
                    smoothed = smooth_states(raw, args.state_gap)
                    if not smoothed: 
                        continue

                    curr = smoothed[0]
                    start_f = frames[args.window_size]
                    for i in range(1, len(smoothed)):
                        if smoothed[i] != curr:
                            event_log.append([full_drug_id, ring_type, full_poly_id, enant, curr, start_f, frames[i + args.window_size - 1], has_bond])
                            start_f = frames[args.window_size + i]
                            curr = smoothed[i]
                    event_log.append([full_drug_id, ring_type, full_poly_id, enant, curr, start_f, frames[-1], has_bond])

            # Intermediate checkpoint write
            if d_idx % 2 == 0:
                pd.DataFrame(event_log, columns=['drug_id', 'ring_type', 'poly_id', 'enant', 'state', 'start', 'end', 'success']).to_csv(output_file, index=False)

    df_final = pd.DataFrame(event_log, columns=['drug_id', 'ring_type', 'poly_id', 'enant', 'state', 'start', 'end', 'success'])
    df_final.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print("KINETIC STATE PREDICTION SUMMARY")
    print("="*80)
    
    summary = df_final.groupby(['enant', 'ring_type', 'state'])['success'].agg(
        Total_Events='count', 
        Successful_Bonds='sum', 
        Success_Rate='mean'
    )
    summary['Success_Rate'] = (summary['Success_Rate'] * 100).round(4).astype(str) + ' %'
    print(summary.to_string())
    print("="*80)
    print(f"Master predictive log successfully compiled and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Spatio-temporal Kinetic State Predicates")
    parser.add_argument("--h5_path", required=True, help="Path to exhaustive_ring_pairs.h5")
    parser.add_argument("--hb_success", required=True, help="Path to step0_with_dsep.csv")
    parser.add_argument("--out_dir", required=True, help="Output directory for generated logs")
    parser.add_argument("--epsilon", type=float, default=0.10, help="Movement threshold in Angstroms")
    parser.add_argument("--window_size", type=int, default=10, help="Sliding window size")
    parser.add_argument("--state_gap", type=int, default=3, help="State gap tolerance (Tau_3)")
    parser.add_argument("--session_gap", type=int, default=5, help="Stacking zone session gap tolerance")
    
    args = parser.parse_args()
    main(args)