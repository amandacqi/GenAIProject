#!/usr/bin/env python
import os
import argparse

import torch
import pandas as pd
from transformers import AutoTokenizer, EsmForProteinFolding


# 1. Load candidates from CSV
def load_candidates_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV '{path}' not found.")

    df = pd.read_csv(path)

    if "seq" not in df.columns:
        raise ValueError("CSV must contain a 'seq' column.")

    df["seq"] = df["seq"].astype(str).str.strip().str.upper()

    if "id" not in df.columns:
        df["id"] = [f"cand_{i+1}" for i in range(len(df))]

    return df[["id", "seq"]]

# 2. Scoring stub
def score_with_ai4amp(df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for functional scoring."""
    df = df.copy()
    df["ai4amp_score"] = float("nan")
    return df

# 3. Simple PDB writer (CA-only)
def write_ca_pdb(seq_id, seq, positions, plddt, out_path):
    """
    Minimal CA-only PDB writer for HuggingFace EsmForProteinFolding.
    positions: tensor of shape (L, ..., 3) after taking batch dim 0
    plddt: tensor of shape (L, ...) with per-residue scores
    """
    import numpy as np
    import torch

    # Make sure we have numpy arrays
    if isinstance(positions, torch.Tensor):
        pos = positions.detach().cpu().numpy()
    else:
        pos = np.asarray(positions)

    if plddt is not None:
        if isinstance(plddt, torch.Tensor):
            conf = plddt.detach().cpu().numpy()
        else:
            conf = np.asarray(plddt)
    else:
        conf = None

    L = len(seq)

    with open(out_path, "w") as f:
        atom_index = 1
        for i, aa in enumerate(seq):
            if i >= pos.shape[0]:
                break

            # Flatten coordinates for residue i, take first 3 as x,y,z
            coord = np.asarray(pos[i]).reshape(-1)
            if coord.size < 3:
                continue
            x, y, z = coord[:3]

            # Per-residue B-factor: mean of conf[i], whatever its shape
            if conf is not None and i < conf.shape[0]:
                conf_i = np.asarray(conf[i]).reshape(-1)
                b = float(conf_i.mean())
            else:
                b = 0.0

            f.write(
                "ATOM  {:5d}  CA  ALA A{:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00{:6.2f}           C\n".format(
                    atom_index, i + 1, x, y, z, b
                )
            )
            atom_index += 1

        f.write("END\n")

# 4. Folding with HF ESMFold
def fold_with_hf_esmfold(df, out_dir="esmfold_candidates_hf"):
    os.makedirs(out_dir, exist_ok=True)

    print("Loading HuggingFace ESMFold (EsmForProteinFolding)...")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print("Using device:", device)

    results = []

    for _, row in df.iterrows():
        seq_id = row["id"]
        seq = row["seq"]
        print(f"Folding {seq_id} (length {len(seq)})...")

        inputs = tok(seq, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # positions: [batch, L, ... , 3], plddt: [batch, L]
        positions = outputs.positions[0]        # take batch 0
        plddt = outputs.plddt[0]                # per-residue confidence

        pdb_path = os.path.join(out_dir, f"{seq_id}.pdb")
        write_ca_pdb(seq_id, seq, positions, plddt, pdb_path)

        results.append({
            "id": seq_id,
            "seq": seq,
            "pLDDT_mean": float(plddt.mean().item()),
            "pdb_path": pdb_path,
        })

    return pd.DataFrame(results)

# 5. Main
def main():
    parser = argparse.ArgumentParser(description="Score + Fold peptide candidates using HF ESMFold.")
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Input CSV with at least a 'seq' column, optional 'id'."
    )
    parser.add_argument(
        "--out_csv",
        default="candidates_scored_and_folded.csv",
        help="Output CSV for scores + folding results."
    )
    parser.add_argument(
        "--out_dir",
        default="esmfold_candidates",
        help="Directory to save PDB files."
    )
    args = parser.parse_args()

    df = load_candidates_csv(args.input_csv)
    print("Loaded candidates:")
    print(df)

    df_scored = score_with_ai4amp(df)

    summary = fold_with_hf_esmfold(df_scored, out_dir=args.out_dir)

    merged = df_scored.merge(summary, on=["id", "seq"], how="left")
    merged.to_csv(args.out_csv, index=False)

    print("\nDone. Summary:")
    print(merged[["id", "seq", "ai4amp_score", "pLDDT_mean", "pdb_path"]])
    print(f"\nSaved combined results to: {args.out_csv}")
    print(f"PDBs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()