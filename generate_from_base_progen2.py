#!/usr/bin/env python
import re
import math
import torch
import pandas as pd
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer

# 1. Config
MODEL_NAME = "hugohrban/progen2-medium"
device = "cuda" if torch.cuda.is_available() else "cpu"

OUT_RAW = "/jet/home/aqi/generated_raw_base.csv"
OUT_FILTERED = "/jet/home/aqi/generated_filtered_base.csv"

print("Using device:", device)

# 2. Load base ProGen2 model (no PEFT)
print("Loading base ProGen2 model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
).to(device)
base_model.eval()

# Patch config so .generate works nicely
cfg = base_model.config
if not hasattr(cfg, "num_hidden_layers") and hasattr(cfg, "n_layer"):
    cfg.num_hidden_layers = cfg.n_layer
if not hasattr(cfg, "hidden_size") and hasattr(cfg, "embed_dim"):
    cfg.hidden_size = cfg.embed_dim
if not hasattr(cfg, "num_attention_heads") and hasattr(cfg, "n_head"):
    cfg.num_attention_heads = cfg.n_head
if not hasattr(cfg, "max_position_embeddings") and hasattr(cfg, "n_positions"):
    cfg.max_position_embeddings = cfg.n_positions

# 3. Load tokenizer from the base model
#    (you could also load tokenizer.json from your tuned dir if you prefer)
print("Loading tokenizer...")
tok = Tokenizer.from_pretrained(MODEL_NAME)
tok.no_padding()

# Add or get pad token
pad_id = tok.token_to_id("<pad>")
if pad_id is None:
    pad_id = tok.get_vocab_size()
    tok.add_special_tokens(["<pad>"])
print("Pad token id:", pad_id)

# Make sure model uses the same pad id
if getattr(cfg, "pad_token_id", None) is None:
    cfg.pad_token_id = pad_id

# 4. Scales and property helpers
KD_SCALE = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
    "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}

CHARGE_WEIGHTS = {
    "K": +1.0,
    "R": +1.0,
    "H": +0.1,
    "D": -1.0,
    "E": -1.0,
}


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


def seq_properties(seq: str):
    seq = seq.strip()
    L = len(seq)
    if L == 0:
        return None

    kd_sum = 0.0
    counted = 0
    for aa in seq:
        if aa in KD_SCALE:
            kd_sum += KD_SCALE[aa]
            counted += 1
    if counted == 0:
        return None
    kd_mean = kd_sum / counted

    q = 0.0
    for aa in seq:
        q += CHARGE_WEIGHTS.get(aa, 0.0)

    hm = hydrophobic_moment_alpha(seq)

    return {
        "seq": seq,
        "length": L,
        "kd": kd_mean,
        "hm": hm,
        "charge": q,
    }


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


def generate_peptides(
    cond_tag="helix_high anti-MRSA peptide",
    num_seqs=50,
    max_new_tokens=64,
):
    prompt = f"{cond_tag}: "
    enc = tok.encode(prompt)
    input_ids = torch.tensor([enc.ids], device=device)

    with torch.no_grad():
        out = base_model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=pad_id,
            num_return_sequences=num_seqs,
        )

    all_seqs = []
    for i in range(out.size(0)):
        ids = out[i].tolist()
        text = tok.decode(ids)

        if ":" in text:
            seq_part = text.split(":", 1)[1]
        else:
            seq_part = text
        seq_part = seq_part.strip()

        # keep only amino acids
        seq_clean = re.sub("[^ARNDCQEGHILKMFPSTWYV]", "", seq_part)
        all_seqs.append(seq_clean)

    return all_seqs


if __name__ == "__main__":
    total_to_generate = 2000
    batch_size = 200

    raw_seqs = []

    n_batches = total_to_generate // batch_size
    if total_to_generate % batch_size != 0:
        n_batches += 1

    for i in range(n_batches):
        this_batch = min(batch_size, total_to_generate - len(raw_seqs))
        if this_batch <= 0:
            break
        print(f"Batch {i+1} generating {this_batch} sequences...")
        seqs = generate_peptides(
            cond_tag="helix_high anti-MRSA peptide",
            num_seqs=this_batch,
            max_new_tokens=64,
        )
        raw_seqs.extend(seqs)

    print(f"Generated {len(raw_seqs)} raw sequences")

    # Save raw sequences
    pd.DataFrame({"seq": raw_seqs}).to_csv(OUT_RAW, index=False)
    print(f"Saved raw sequences to {OUT_RAW}")

    # Filter sequences
    filtered = filter_sequences(raw_seqs)
    print(f"{len(filtered)} sequences passed filters")

    pd.DataFrame(filtered).to_csv(OUT_FILTERED, index=False)
    print(f"Saved filtered sequences to {OUT_FILTERED}")

    # Quick print of first 20
    for props in filtered[:20]:
        print(
            f"{props['seq']}\t"
            f"L={props['length']}\t"
            f"HM={props['hm']:.2f}\t"
            f"Q={props['charge']:.1f}"
        )