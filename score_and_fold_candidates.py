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
def write_ca_pdb(seq_id: str, seq: str, positions, plddt, pdb_path: str):
    """
    Write a very simple CA-only PDB.
    positions: [L, n_atoms, 3] tensor/ndarray
    plddt: [L] tensor/ndarray
    We assume atom index 1 is CA (standard for OpenFold/ESMFold ordering).
    """
    import numpy as np

    pos = positions.cpu().numpy() if torch.is_tensor(positions) else positions
    conf = plddt.cpu().numpy() if torch.is_tensor(plddt) else plddt

    L = pos.shape[0]
    # CA index in OpenFold atom order is 1 (N, CA, C, O, ...)
    ca_idx = 1

    with open(pdb_path, "w") as f:
        atom_serial = 1
        for i in range(L):
            x, y, z = pos[i, ca_idx]
            b = conf[i]
            res_idx = i + 1
            res_name = "ALA"  # generic residue name (sequence-level identity not needed for visualization)
            atom_name = "CA"
            # Standard PDB ATOM line (limited info, but usable in PyMOL/Chimera)
            f.write(
                "ATOM  {atom:5d} {name:^4s} {res:>3s} A{resid:4d}    "
                "{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}           C\n".format(
                    atom=atom_serial,
                    name=atom_name,
                    res=res_name,
                    resid=res_idx,
                    x=x,
                    y=y,
                    z=z,
                    b=b,
                )
            )
            atom_serial += 1

        # END record
        f.write("END\n")

# 4. Folding with HF ESMFold
def fold_with_hf_esmfold(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)

    print("Loading HuggingFace ESMFold (EsmForProteinFolding)...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1",
        low_cpu_mem_usage=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")

    results = []

    for _, row in df.iterrows():
        seq_id = row["id"]
        seq = row["seq"]
        print(f"Folding {seq_id} (length {len(seq)})...")

        # Tokenize
        tokens = tokenizer(
            [seq],
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(device)

        with torch.no_grad():
            outputs = model(tokens)

        # outputs.positions: [B, L, atoms, 3]
        # outputs.plddt: [B, L]
        positions = outputs.positions[0]  # [L, atoms, 3]
        plddt = outputs.plddt[0]         # [L]

        plddt_mean = float(plddt.mean().item())

        pdb_path = os.path.join(out_dir, f"{seq_id}.pdb")
        write_ca_pdb(seq_id, seq, positions, plddt, pdb_path)

        results.append({
            "id": seq_id,
            "seq": seq,
            "pLDDT_mean": plddt_mean,
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