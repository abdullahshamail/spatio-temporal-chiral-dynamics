#!/usr/bin/env python3
"""
Usage:
python 01_trajectory_extraction.py --prmtop <path> --dcd <paths> --ring_ids <path> ...
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import MDAnalysis as mda

# ==========================================
# 1. MATH & GEOMETRY UTILITIES
# ==========================================
def min_image_dist(p1, p2, box_x, box_y):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    dx = dx - box_x * np.round(dx / box_x)
    dy = dy - box_y * np.round(dy / box_y)
    return np.sqrt(dx**2 + dy**2 + dz**2)

def min_image_vector(p1, p2, box_x, box_y):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    dx = dx - box_x * np.round(dx / box_x)
    dy = dy - box_y * np.round(dy / box_y)
    return np.array([dx, dy, dz])

def get_ring_normal(coords):
    centered = coords - np.mean(coords, axis=0)
    cov = np.dot(centered.T, centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    normal = eigenvectors[:, 0]
    return normal / np.linalg.norm(normal)

def angle_between_normals(n1, n2):
    dot = np.clip(np.abs(np.dot(n1, n2)), 0.0, 1.0)
    return np.degrees(np.arccos(dot))

def angle_between_vectors(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

# ==========================================
# 2. MAIN EXTRACTION PIPELINE
# ==========================================
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading data mappings...")
    df_ring_ids = pd.read_csv(args.ring_ids)
    df_ring_ids.columns = df_ring_ids.columns.str.strip()
    atom_col = 'atom_id' if 'atom_id' in df_ring_ids.columns else 'ActualIndex'
    offset = 0 if atom_col == 'atom_id' else 1

    df_atom_info = pd.read_csv(args.atom_info)
    df_atom_info.columns = df_atom_info.columns.str.strip()
    df_atom_info = df_atom_info.set_index("atom_id")

    with open(args.geometries, 'r') as f: 
        geometries = json.load(f)

    drug_visits = {}
    for chain in geometries:
        if 'pairs_to_track' in chain and len(chain['pairs_to_track']) > 0:
            drug_ring = chain['pairs_to_track'][0]['drug_ring']
            if drug_ring not in drug_visits:
                drug_visits[drug_ring] = []
            drug_visits[drug_ring].append(chain)

    print(f"Detected {len(drug_visits)} distinct Drug Molecule visits.")
    print("Initializing MDAnalysis trajectory stream (Lazy Loading)...")
    u = mda.Universe(args.prmtop, args.dcd)
    total_frames = len(u.trajectory)

    for drug_ring, chains in drug_visits.items():
        print(f"\n--- Processing Molecule [Primary Ring: {drug_ring}] ({len(chains)} HB events) ---")
        
        # 1. Setup Primary Ring
        d_atoms_df = df_ring_ids[df_ring_ids['unique_ring_id'] == int(drug_ring)]
        if len(d_atoms_df) == 0: 
            continue
        
        primary_idx = d_atoms_df[atom_col].astype(int).values[0] + offset - 1
        d_atoms_0based = d_atoms_df[atom_col].astype(int).values + offset - 1
        d_group = u.atoms[d_atoms_0based]
        subst_name = df_atom_info.loc[d_atoms_0based[0] + 1, 'subst_name']
        
        # 2. Automatically Detect Sister Ring
        molecule_atom_indices = set(u.atoms[primary_idx].residue.atoms.indices)
        sister_ring_id = None
        for ring_id, group in df_ring_ids.groupby('unique_ring_id'):
            if ring_id == int(drug_ring): continue
            ring_indices = set(group[atom_col].astype(int).values + offset - 1)
            if ring_indices.issubset(molecule_atom_indices):
                sister_ring_id = ring_id
                break
                
        if sister_ring_id is None:
            print(f"  -> Warning: Could not find sister ring for {drug_ring}. Skipping.")
            continue
            
        print(f"  -> Found Sister Ring: [{sister_ring_id}]")
        s_atoms_0based = df_ring_ids[df_ring_ids['unique_ring_id'] == sister_ring_id][atom_col].astype(int).values + offset - 1
        s_group = u.atoms[s_atoms_0based]

        # 3. Setup Polymer Rings
        poly_rings = set()
        all_hb_triplets = []
        for c in chains:
            all_hb_triplets.append([int(x) - 1 for x in c['hb_triplet']])
            for pair in c['pairs_to_track']:
                poly_rings.add(pair['poly_ring'])
                
        p_groups = {}
        for pr in poly_rings:
            pr_df = df_ring_ids[df_ring_ids['unique_ring_id'] == int(pr)]
            pr_0based = pr_df[atom_col].astype(int).values + offset - 1
            p_groups[pr] = u.atoms[pr_0based]

        # 4. Define Temporal Window
        start_frames = [c['start_frame'] for c in chains]
        end_frames = [c['end_frame'] for c in chains]
        visit_start = max(0, min(start_frames) - args.buffer_frames)
        visit_end = min(total_frames - 1, max(end_frames) + args.buffer_frames)

        # 5. Output Configuration
        csv_name = f"{subst_name}_Molecule_{drug_ring}_6Pair_Master.csv"
        csv_path = os.path.join(args.out_dir, csv_name)
        headers = [
            "Frame", "Stereo", "Drug_Ring", "Poly_Ring", "HB_Active", 
            "dp_dist", "hp_dist", "theta_p", "alpha_face",
            "Sister_Ring", "Sister_dp_dist", "Sister_hp_dist", "Sister_theta_p", "Sister_alpha_face"
        ]
        pd.DataFrame(columns=headers).to_csv(csv_path, index=False)
        
        records = []
        print(f"  -> Streaming frames {visit_start} to {visit_end} (Writing every {args.chunk_size} frames)...")
        
        # 6. Optimized Single-Pass Trajectory Stream
        for ts in u.trajectory[visit_start : visit_end + 1]:
            abs_frame = ts.frame
            
            # Primary coordinates
            d_coords = d_group.positions
            d_com = d_group.center_of_mass()
            d_normal = get_ring_normal(d_coords)
            
            # Sister coordinates
            s_coords = s_group.positions
            s_com = s_group.center_of_mass()
            s_normal = get_ring_normal(s_coords)
            
            # Resolve Hydrogen Bond Anchor Status
            current_hb = None
            for c in chains:
                if c['start_frame'] <= abs_frame <= c['end_frame']:
                    current_hb = [int(x) - 1 for x in c['hb_triplet']]
                    break
                    
            if current_hb is None:
                hb_center = np.mean([u.atoms[triplet].positions for triplet in all_hb_triplets], axis=(0, 1))
                drug_reactive_atom = u.atoms[all_hb_triplets[0][0]].position
            else:
                h_pos, n_pos, o_pos = u.atoms[current_hb].positions
                hb_center = np.mean([h_pos, n_pos, o_pos], axis=0)
                dists = [np.linalg.norm(pos - d_com) for pos in [h_pos, n_pos, o_pos]]
                drug_reactive_atom = [h_pos, n_pos, o_pos][np.argmin(dists)]

            # Iterate against surrounding polymer pockets
            for p_ring, p_group in p_groups.items():
                p_coords = p_group.positions
                p_com = p_group.center_of_mass()
                p_normal = get_ring_normal(p_coords)
                
                # Primary Ring Metrics
                dp = min_image_dist(d_com, p_com, args.box_x, args.box_y)
                anchor_center_d = (d_com + p_com) / 2.0
                hp_d = min_image_dist(anchor_center_d, hb_center, args.box_x, args.box_y)
                theta_p_d = angle_between_normals(d_normal, p_normal)
                v_internal_d = min_image_vector(d_com, drug_reactive_atom, args.box_x, args.box_y)
                v_scaffold_d = min_image_vector(d_com, p_com, args.box_x, args.box_y)
                alpha_face_d = angle_between_vectors(v_internal_d, v_scaffold_d)
                
                # Sister Ring Metrics
                dp_s = min_image_dist(s_com, p_com, args.box_x, args.box_y)
                anchor_center_s = (s_com + p_com) / 2.0
                hp_s = min_image_dist(anchor_center_s, hb_center, args.box_x, args.box_y)
                theta_p_s = angle_between_normals(s_normal, p_normal)
                v_internal_s = min_image_vector(s_com, drug_reactive_atom, args.box_x, args.box_y)
                v_scaffold_s = min_image_vector(s_com, p_com, args.box_x, args.box_y)
                alpha_face_s = angle_between_vectors(v_internal_s, v_scaffold_s)
                
                records.append({
                    "Frame": abs_frame,
                    "Stereo": subst_name,
                    "Drug_Ring": drug_ring,
                    "Poly_Ring": p_ring,
                    "HB_Active": 1 if current_hb else 0,
                    "dp_dist": round(dp, 3),
                    "hp_dist": round(hp_d, 3),
                    "theta_p": round(theta_p_d, 2),
                    "alpha_face": round(alpha_face_d, 2),
                    "Sister_Ring": sister_ring_id,
                    "Sister_dp_dist": round(dp_s, 3),
                    "Sister_hp_dist": round(hp_s, 3),
                    "Sister_theta_p": round(theta_p_s, 2),
                    "Sister_alpha_face": round(alpha_face_s, 2)
                })
                
            # Memory-safe chunk dumping
            if len(records) >= args.chunk_size:
                pd.DataFrame(records).to_csv(csv_path, mode='a', header=False, index=False)
                records.clear()

        # Dump remaining
        if len(records) > 0:
            pd.DataFrame(records).to_csv(csv_path, mode='a', header=False, index=False)
            records.clear()
            
        print(f"  -> Saved consolidated metrics to {csv_name}")

    print(f"\nExtraction Complete! Unified O(N) Master files saved to {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MD Spatio-temporal Trajectory Data")
    parser.add_argument("--prmtop", required=True, help="Path to the topology (.prmtop) file")
    parser.add_argument("--dcd", required=True, nargs='+', help="Path to the trajectory (.dcd) file(s)")
    parser.add_argument("--ring_ids", required=True, help="CSV mapping unique ring IDs to atom indices")
    parser.add_argument("--atom_info", required=True, help="CSV containing atom info and subst_names")
    parser.add_argument("--geometries", required=True, help="JSON containing tracked HB chain geometries")
    parser.add_argument("--out_dir", required=True, help="Output directory for the consolidated CSVs")
    parser.add_argument("--box_x", type=float, default=72.475, help="PBC Box X dimension (Default: 72.475)")
    parser.add_argument("--box_y", type=float, default=72.475, help="PBC Box Y dimension (Default: 72.475)")
    parser.add_argument("--chunk_size", type=int, default=5000, help="Frames per chunk for safe CSV writing")
    parser.add_argument("--buffer_frames", type=int, default=1000, help="Buffer frames around active HB events")
    
    args = parser.parse_args()
    main(args)