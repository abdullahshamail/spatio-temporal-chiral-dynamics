#!/usr/bin/env python3
"""
Usage:
python 05_micromechanics_visualization.py --h5_path <path> --hb_csv <path> --out_dir <path>
"""

import matplotlib
matplotlib.use('Agg')
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import gc
import argparse

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    
    h5_path = args.h5_path
    csv_path = args.hb_csv
    out_dir = args.out_dir

    tau_1 = 5
    max_allowed_distance = 5.5
    dsep_col_name = 'd_sep'

    df_hb = pd.read_csv(csv_path)

    def get_chains(df_subset):
        records = []
        for (drug_ring, poly_ring), group in df_subset.groupby(['drug_ring', 'poly_ring']):
            frames = group['frame'].sort_values().values
            if len(frames) == 0: continue
            
            actual_drug = group['drug_id'].iloc[0] if 'drug_id' in group.columns else drug_ring.split('_')[0]
            actual_poly = group['poly_id'].iloc[0] if 'poly_id' in group.columns else poly_ring.split('_')[0]
            
            c_start = prev = frames[0]
            for f in frames[1:]:
                if f - prev <= tau_1:
                    prev = f
                else:
                    if (prev - c_start) > 100: 
                        records.append({'drug_ring': drug_ring, 'poly_ring': poly_ring, 'hb_drug_id': actual_drug, 'hb_poly_id': actual_poly, 'start': c_start, 'end': prev, 'length': prev - c_start})
                    c_start = prev = f
            if (prev - c_start) > 100:
                records.append({'drug_ring': drug_ring, 'poly_ring': poly_ring, 'hb_drug_id': actual_drug, 'hb_poly_id': actual_poly, 'start': c_start, 'end': prev, 'length': prev - c_start})
        return pd.DataFrame(records)

    # Process SFL Chains
    df_sfl_raw = df_hb[df_hb['drug_ring'].str.startswith(tuple(['10_', '11_', '12_', '13_', '14_']))]
    chains_sfl = get_chains(df_sfl_raw).sort_values(by='length', ascending=False).head(100).reset_index(drop=True)

    valid_sfl = []
    with h5py.File(h5_path, 'r') as f:
        d_keys, p_keys = [k.decode('utf-8') for k in f.attrs['drug_ring_order']], [k.decode('utf-8') for k in f.attrs['poly_ring_order']]
        for _, row in chains_sfl.iterrows():
            c_s, c_e = int(row['start']), int(row['end'])
            d_idx, p_idx = d_keys.index(row['drug_ring']), p_keys.index(row['poly_ring'])
            dist_slice = f['distances'][c_s:c_e, d_idx, p_idx] if len(f['distances'].shape) == 3 else f['distances'][c_s:c_e, d_idx * len(p_keys) + p_idx]
            mean_d = np.mean(dist_slice)
            if mean_d <= max_allowed_distance:
                row_dict = row.to_dict()
                row_dict['mean_distance'] = mean_d
                valid_sfl.append(row_dict)
    final_sfl = pd.DataFrame(valid_sfl).sort_values(by='length', ascending=False).head(1).reset_index(drop=True)

    with h5py.File(h5_path, 'r') as f:
        n_frames_total = f['distances'].shape[0]
        for rank, row in final_sfl.iterrows():
            c_start, c_end = int(row['start']), int(row['end'])
            fixed_x_width = int(row['length'] + 3000)
            center = (c_start + c_end) // 2
            plot_start = max(0, int(center - (fixed_x_width / 2)))
            plot_end = min(n_frames_total, int(center + (fixed_x_width / 2)))

            d_idx, p_idx = d_keys.index(row['drug_ring']), p_keys.index(row['poly_ring'])
            dist_top = f['distances'][plot_start:plot_end, d_idx, p_idx] if len(f['distances'].shape) == 3 else f['distances'][plot_start:plot_end, d_idx * len(p_keys) + p_idx]
            frame_range = np.arange(plot_start, plot_end)

            active_hb = df_hb[(df_hb['drug_ring'] == row['drug_ring']) & 
                              (df_hb['poly_ring'] == row['poly_ring']) & 
                              (df_hb['frame'] >= c_start) & (df_hb['frame'] <= c_end)]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1.2]})
            
            # Top panel
            ax1.plot(frame_range, dist_top, color='#1e40af', linewidth=1.5, alpha=0.9, label=r'Ring-Ring Proximity ($d_{ring-ring}$)')
            ax1.axvspan(c_start, c_end, color='#86efac', alpha=0.3)
            ax1.axvline(c_start, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7)
            ax1.axvline(c_end, color='darkred', linestyle='--', linewidth=2, alpha=0.7)
            ax1.set_title(f"SFL Deep Nesting - Ring Pair: {row['drug_ring']} to {row['poly_ring']}", fontsize=14)
            ax1.set_ylabel(r"Ring-Ring Distance ($\AA$)", fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper right')

            # Bottom panel
            if not active_hb.empty and dsep_col_name in active_hb.columns:
                dsep_vals = active_hb[dsep_col_name]
                ax2.plot(active_hb['frame'], dsep_vals, color='#7e22ce', marker='.', linestyle='None', alpha=0.7, label=r'Distance $d_{sep}$')
            ax2.axvspan(c_start, c_end, color='#86efac', alpha=0.3)
            ax2.axvline(c_start, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7)
            ax2.axvline(c_end, color='darkred', linestyle='--', linewidth=2, alpha=0.7)
            ax2.set_ylabel(r"$d_{sep}$ Distance ($\AA$)", fontsize=12)
            ax2.set_xlabel("Simulation Frame Number", fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right')

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"SFL_Micromechanics_Rank_{rank+1}.png"), dpi=300, bbox_inches='tight')
            plt.close('all')
            gc.collect()

    print(f"Successfully generated micro-mechanics visualization in {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Micromechanics Visualization Script")
    parser.add_argument("--h5_path", required=True, help="Path to exhaustive_ring_pairs.h5")
    parser.add_argument("--hb_csv", required=True, help="Path to step0_with_dsep.csv")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()
    main(args)