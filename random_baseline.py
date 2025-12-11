#!/usr/bin/env python3
"""
Make a Random(p) mutation baseline from a test-set FASTA.

- Reads an input FASTA of test AMPs
- For each sequence, independently mutates each residue with probability p
- Writes a mutated FASTA with modified headers

Usage:
    python make_random_mutants.py \
        --input_fasta test_set.fasta \
        --out_fasta test_set_mut_p0.1.fasta \
        --p 0.1
"""

import argparse
import random
from typing import List, Tuple

# Canonical amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def parse_fasta(path: str) -> List[Tuple[str, str]]:
    """Parse a FASTA file into a list of (header, sequence)."""
    records = []
    header = None
    seq_chunks = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # save previous record
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:]  # drop ">"
                seq_chunks = []
            else:
                seq_chunks.append(line)

    # last record
    if header is not None:
        records.append((header, "".join(seq_chunks)))

    return records


def write_fasta(records: List[Tuple[str, str]], path: str) -> None:
    """Write a list of (header, sequence) to FASTA."""
    with open(path, "w") as f:
        for header, seq in records:
            f.write(f">{header}\n")
            # wrap lines at 60–80 chars if you want; here we keep it simple
            f.write(seq + "\n")


def mutate_sequence(seq: str, p: float) -> str:
    """
    Randomly mutate each residue with probability p.

    - Only mutate canonical amino acids in AMINO_ACIDS.
    - When mutating, choose a different amino acid uniformly at random.
    """
    seq = seq.strip().upper()
    out = []

    for aa in seq:
        if aa not in AMINO_ACIDS:
            # e.g., X or other characters: leave as-is
            out.append(aa)
            continue

        if random.random() < p:
            choices = [x for x in AMINO_ACIDS if x != aa]
            out.append(random.choice(choices))
        else:
            out.append(aa)

    return "".join(out)


def make_mutants(
    records: List[Tuple[str, str]],
    p: float,
    seed: int = 0,
) -> List[Tuple[str, str]]:
    """Create one mutant for each (header, sequence) pair."""
    random.seed(seed)

    mutated_records = []
    for header, seq in records:
        mut_seq = mutate_sequence(seq, p)
        new_header = f"{header}_mut_p{p}"
        mutated_records.append((new_header, mut_seq))

    return mutated_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fasta", required=True, help="Test set FASTA file")
    parser.add_argument("--out_fasta", required=True, help="Output FASTA for mutants")
    parser.add_argument(
        "--p",
        type=float,
        default=0.1,
        help="Per-residue mutation probability (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )
    args = parser.parse_args()

    records = parse_fasta(args.input_fasta)
    print(f"Loaded {len(records)} test sequences from {args.input_fasta}")

    mutated_records = make_mutants(records, p=args.p, seed=args.seed)
    write_fasta(mutated_records, args.out_fasta)

    print(f"Wrote {len(mutated_records)} mutated sequences to {args.out_fasta}")
    print(f"Per-residue mutation probability p = {args.p}")


if __name__ == "__main__":
    main()