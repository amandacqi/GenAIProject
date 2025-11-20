import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from peft import PrefixTuningConfig, get_peft_model
from tokenizers import Tokenizer
import wandb

# 1. Paths and config
MODEL_NAME = "hugohrban/progen2-medium"  # or progen2-small
TRAIN_FILE = "progen2_prefix_train.txt"
OUTPUT_DIR = "progen2_prefix_tuned"

MAX_LENGTH = 256
BATCH_SIZE = 4
EPOCHS = 3
LR = 5e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load tokenizer (ProGen2 uses tokenizers.Tokenizer, not AutoTokenizer)
print("Loading tokenizer...")
tok = Tokenizer.from_pretrained(MODEL_NAME)
tok.no_padding()  # we will handle padding manually below

# Get a pad token id; if none exists, add one
if tok.token_to_id("<pad>") is None:
    # add a pad token at the end of vocab
    pad_id = tok.get_vocab_size()
    tok.add_special_tokens(["<pad>"])
else:
    pad_id = tok.token_to_id("<pad>")

print("Pad token id:", pad_id)

# 3. Load base ProGen2 model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
).to(device)

# 4. Freeze base model weights
for param in model.parameters():
    param.requires_grad = False

# 5. Attach prefix-tuning adapter
peft_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=10  # you can try 10–20
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()   # should show only prefix params are trainable

# 6. Load text dataset from your file
print("Loading dataset...")
dataset = load_dataset("text", data_files={"train": TRAIN_FILE})["train"]

# 7. Tokenization function
def tokenize_batch(batch):
    texts = batch["text"]
    encs = [tok.encode(t) for t in texts]

    # Truncate to MAX_LENGTH
    for e in encs:
        if len(e.ids) > MAX_LENGTH:
            e.ids = e.ids[:MAX_LENGTH]

    max_len = max(len(e.ids) for e in encs)

    input_ids = []
    attention_mask = []

    for e in encs:
        ids = e.ids
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)

    # For language modeling, labels are usually the same as input_ids
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids,
    }

print("Tokenizing dataset...")
tokenized = dataset.map(
    tokenize_batch,
    batched=True,
    remove_columns=["text"]
)

# 8. Init wandb run
wandb_config = {
    "model_name": MODEL_NAME,
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LR,
    "num_virtual_tokens": peft_config.num_virtual_tokens,
}
wandb_run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "progen2_prefix_tuning"),
    name="progen2_prefix_medium_prefix_tuning",
    config=wandb_config
)

# 9. TrainingArguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    logging_steps=50,
    save_strategy="epoch",
    report_to=["wandb"],      # send logs to wandb
    run_name=wandb_run.name,  # shows up as the run name in wandb
)

# 10. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)

# 11. Train
print("Starting training...")
trainer.train()

# 12. Save tuned model + tokenizer
print("Saving model and tokenizer...")
trainer.save_model(OUTPUT_DIR)
tok.save(os.path.join(OUTPUT_DIR, "tokenizer.json"))

# 13. Finish wandb run
wandb.finish()

print("Done. Prefix-tuned model saved to", OUTPUT_DIR)