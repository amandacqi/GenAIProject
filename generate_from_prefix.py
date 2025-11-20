import re
import torch
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer
from peft import PeftConfig, PeftModel

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