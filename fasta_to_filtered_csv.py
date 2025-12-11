#!/usr/bin/env python
import argparse
import math
import pandas as pd

# Kyte–Doolittle hydrophobicity scale
KD_SCALE = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
    "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}

# Simple charge weights at ~physiological pH
CHARGE_WEIGHTS = {
    "K": +1.0,
    "R": +1.0,
    "H": +0.1,  # partially protonated
    "D": -1.0,
    "E": -1.0,
}

# Hydrophobic moment for alpha-helix
def hydrophobic_moment_alpha(seq: str, angle_deg: float = 100.0) -> float:
    theta = math.radians(angle_deg)
    vx = 0.0
    vy = 0.0
    L = len(seq)
    if L == 0:
        return 0.0
    for i, aa in enumerate(seq):
        h = KD_SCALE.get(aa, 0.0)
        angle = i * theta
        vx += h * math.cos(angle)
        vy += h * math.sin(angle)
    return (vx**2 + vy**2) ** 0.5 / L

# Compute properties for a single sequence
def seq_properties(seq: str):
    seq = seq.strip().upper()
    L = len(seq)
    if L == 0:
        return None

    # mean Kyte–Doolittle hydrophobicity
    kd_sum = 0.0
    counted = 0
    for aa in seq:
        if aa in KD_SCALE:
            kd_sum += KD_SCALE[aa]
            counted += 1
    if counted == 0:
        return None
    kd_mean = kd_sum / counted

    # simple net charge estimate
    q = 0.0
    for aa in seq:
        q += CHARGE_WEIGHTS.get(aa, 0.0)

    # Hydrophobic moment
    hm = hydrophobic_moment_alpha(seq)

    return {
        "seq": seq,
        "length": L,
        "kd": kd_mean,
        "hm": hm,
        "charge": q,
    }

# Filter function on length, charge, hydrophobic moment
def filter_sequences(
    seqs,
    length_range=(10, 40),
    charge_range=(+2.0, +10.0),
    hm_range=(0.25, 3.0),
):
    kept = []
    for s in seqs:
        props = seq_properties(s)
        if props is None:
            continue

        L = props["length"]
        hm = props["hm"]
        q = props["charge"]

        if not (length_range[0] <= L <= length_range[1]):
            continue
        if not (charge_range[0] <= q <= charge_range[1]):
            continue
        if not (hm_range[0] <= hm <= hm_range[1]):
            continue

        kept.append(props)

    return kept

def read_fasta(path: str):
    """Return a list of sequences from a FASTA file (headers ignored)."""
    seqs = []
    current = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # header line
                if current:
                    seqs.append("".join(current))
                    current = []
            else:
                current.append(line)

    if current:
        seqs.append("".join(current))

    return seqs

def main():
    parser = argparse.ArgumentParser(
        description="Convert FASTA sequences into filtered CSV with length, kd, hm, charge."
    )
    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    parser.add_argument(
        "--no_filter",
        action="store_true",
        help="If set, do NOT apply length/charge/hm filters"
    )
    args = parser.parse_args()

    raw_seqs = read_fasta(args.fasta)
    print(f"Read {len(raw_seqs)} sequences from {args.fasta}")

    if args.no_filter:
        records = []
        for s in raw_seqs:
            props = seq_properties(s)
            if props is not None:
                records.append(props)
        print(f"{len(records)} sequences had valid properties (no filtering)")
    else:
        records = filter_sequences(raw_seqs)
        print(f"{len(records)} sequences passed filters")

    pd.DataFrame(records).to_csv(args.out_csv, index=False)
    print(f"Saved {len(records)} sequences to {args.out_csv}")

if __name__ == "__main__":
    main()