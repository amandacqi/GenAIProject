#!/usr/bin/env python
import os
import argparse

import torch
import pandas as pd
from esm import pretrained


# 1. Load candidates from CSV
def load_candidates_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV '{path}' not found.")

    df = pd.read_csv(path)

    # Must contain sequences
    if "seq" not in df.columns:
        raise ValueError("CSV must contain a 'seq' column.")

    # Clean and uppercase sequences
    df["seq"] = df["seq"].astype(str).str.strip().str.upper()

    # Assign ids if not provided
    if "id" not in df.columns:
        df["id"] = [f"cand_{i+1}" for i in range(len(df))]

    return df[["id", "seq"]]

# 2. Scoring stub
def score_with_ai4amp(df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder scoring function."""
    df = df.copy()
    df["ai4amp_score"] = float("nan")
    return df


# 3. Folding with ESMFold
def fold_with_esmfold(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)

    print("Loading ESMFold model...")
    model = pretrained.esmfold_v1()
    model = model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")

    results = []

    for _, row in df.iterrows():
        seq_id = row["id"]
        seq = row["seq"]

        print(f"Folding {seq_id} (length {len(seq)})...")

        # High-level fold (string PDB)
        with torch.no_grad():
            pdb_str = model.infer_pdb(seq)

        # Second pass to extract pLDDT
        with torch.no_grad():
            toks = model.tokenize([seq])
            if device == "cuda":
                toks = {k: v.to(device) for k, v in toks.items()}
            out = model(**toks)
            plddt = out["plddt"].mean().item()

        # Save PDB
        pdb_path = os.path.join(out_dir, f"{seq_id}.pdb")
        with open(pdb_path, "w") as f:
            f.write(pdb_str)

        results.append({
            "id": seq_id,
            "seq": seq,
            "pLDDT_mean": plddt,
            "pdb_path": pdb_path,
        })

    return pd.DataFrame(results)


# 4. Main: load → score → fold → save
def main():
    parser = argparse.ArgumentParser(description="Score + Fold peptide candidates using ESMFold.")
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Input CSV containing at least a 'seq' column, optional 'id'."
    )
    parser.add_argument(
        "--out_csv",
        default="candidates_scored_and_folded.csv",
        help="Output CSV to write results."
    )
    parser.add_argument(
        "--out_dir",
        default="esmfold_candidates",
        help="Directory to save folded PDB files."
    )

    args = parser.parse_args()

    # Step 1: Load sequences
    df = load_candidates_csv(args.input_csv)
    print("Loaded candidates:")
    print(df)

    # Step 2: Score
    df_scored = score_with_ai4amp(df)

    # Step 3: Fold with ESMFold
    summary = fold_with_esmfold(df_scored, out_dir=args.out_dir)

    # Step 4: Save combined CSV
    merged = df_scored.merge(summary, on=["id", "seq"], how="left")
    merged.to_csv(args.out_csv, index=False)

    print("\nDone. Summary:")
    print(merged[["id", "seq", "ai4amp_score", "pLDDT_mean", "pdb_path"]])
    print(f"\nSaved combined results to: {args.out_csv}")
    print(f"PDBs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()