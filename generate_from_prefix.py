import re
import torch
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer
from peft import PeftConfig, PeftModel
import csv
import pandas as pd

MODEL_DIR = "progen2_prefix_tuned"  # adapter dir
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tuned model...")

# Load PEFT config from your adapter folder
peft_config = PeftConfig.from_pretrained(MODEL_DIR)

# Load the base ProGen2 model that you fine-tuned
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    trust_remote_code=True
)

# *** IMPORTANT: patch ProGen config to match what transformers.generate expects ***
cfg = base_model.config
if not hasattr(cfg, "num_hidden_layers") and hasattr(cfg, "n_layer"):
    cfg.num_hidden_layers = cfg.n_layer
if not hasattr(cfg, "hidden_size") and hasattr(cfg, "embed_dim"):
    cfg.hidden_size = cfg.embed_dim
if not hasattr(cfg, "num_attention_heads") and hasattr(cfg, "n_head"):
    cfg.num_attention_heads = cfg.n_head
if not hasattr(cfg, "max_position_embeddings") and hasattr(cfg, "n_positions"):
    cfg.max_position_embeddings = cfg.n_positions

# Attach the prefix-tuned adapter
model = PeftModel.from_pretrained(base_model, MODEL_DIR).to(device)
model.eval()

# Load tokenizer from your saved tokenizer.json
tok = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")

pad_id = tok.token_to_id("<pad>")
if pad_id is None:
    raise ValueError("Pad token <pad> not found in tokenizer")

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

# Compute properties for a single sequence
def seq_properties(seq: str):
    seq = seq.strip()
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

    return {
        "seq": seq,
        "length": L,
        "kd": kd_mean,
        "charge": q,
    }

# Filter function on length, charge, hydrophobicity
def filter_sequences(
    seqs,
    length_range=(10, 40),
    charge_range=(+2.0, +10.0),
    kd_range=(-0.5, 1.5),
):
    kept = []
    for s in seqs:
        props = seq_properties(s)
        if props is None:
            continue

        L = props["length"]
        kd = props["kd"]
        q = props["charge"]

        if not (length_range[0] <= L <= length_range[1]):
            continue
        if not (charge_range[0] <= q <= charge_range[1]):
            continue
        if not (kd_range[0] <= kd <= kd_range[1]):
            continue

        kept.append(props)

    return kept

def generate_peptides(
    cond_tag="helix_high anti-MRSA peptide",
    num_seqs=50,
    max_new_tokens=64,
):
    # Batch generation in a single call
    prompt = f"{cond_tag}: "
    enc = tok.encode(prompt)
    input_ids = torch.tensor([enc.ids], device=device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=pad_id,
            num_return_sequences=num_seqs,   # Batch generate
        )

    all_seqs = []
    for i in range(out.size(0)):
        ids = out[i].tolist()
        text = tok.decode(ids)

        # Take everything after the first colon
        if ":" in text:
            seq_part = text.split(":", 1)[1]
        else:
            seq_part = text
        seq_part = seq_part.strip()

        # Clean to amino acids only
        seq_clean = re.sub("[^ARNDCQEGHILKMFPSTWYV]", "", seq_part)
        all_seqs.append(seq_clean)

    return all_seqs

if __name__ == "__main__":
    # Generate thousands
    raw_seqs = []
    batch = 200 

    for i in range(2000 // batch):
        print(f"Batch {i+1}")
        seqs = generate_peptides(
            cond_tag="helix_high anti-MRSA peptide",
            num_seqs=batch,
            max_new_tokens=64,
        )
        raw_seqs.extend(seqs)

    print(f"Generated {len(raw_seqs)} raw sequences")

    # Save ALL raw sequences to CSV  --------------------- NEW
    pd.DataFrame({"seq": raw_seqs}).to_csv(
        "generated_raw.csv",
        index=False
    )
    print("Saved raw sequences to candidates_raw.csv")

    # Filter by length, charge, hydrophobicity
    filtered = filter_sequences(
        raw_seqs,
        length_range=(10, 40),
        charge_range=(+2.0, +10.0),
        kd_range=(-0.5, 1.5),
    )

    print(f"{len(filtered)} sequences passed filters")

    # Save filtered sequences + properties to CSV -------- NEW
    pd.DataFrame(filtered).to_csv(
        "generated_filtered.csv",
        index=False
    )
    print("Saved filtered sequences to candidates_filtered.csv")

    # Print the first 20 filtered sequences for a quick check
    for props in filtered[:20]:
        print(
            f"{props['seq']}\t"
            f"L={props['length']}\t"
            f"KD={props['kd']:.2f}\t"
            f"Q={props['charge']:.1f}"
        )