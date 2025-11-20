import re
import torch
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer

MODEL_DIR = "progen2_prefix_tuned"  # same as OUTPUT_DIR above
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tuned model and tokenizer
print("Loading tuned model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True
).to(device)

tok = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")

# Get pad token id
pad_id = tok.token_to_id("<pad>")
if pad_id is None:
    raise ValueError("Pad token <pad> not found in tokenizer")

def generate_peptides(cond_tag="helix_high anti-MRSA peptide", num_seqs=50, max_new_tokens=64):
    prompts = [f"{cond_tag}: " for _ in range(num_seqs)]

    all_seqs = []
    for prompt in prompts:
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
            )

        # Decode to text
        ids = out[0].tolist()
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
    cand_seqs = generate_peptides(
        cond_tag="helix_high anti-MRSA peptide",
        num_seqs=100
    )
    for s in cand_seqs[:10]:
        print(s)