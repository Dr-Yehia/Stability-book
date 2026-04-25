# =============================================================
# train_stability_model.py
# Fine-tune LLM on W.F. Chen - Structural Stability (2 vols)
# Google Colab  |  GPU: T4 or A100
# Run each CELL in order. At the end a ZIP file will download.
# =============================================================

import urllib.parse
import os


# ─────────────────────────────────────────────
# CELL 1 ─ Install libraries  (run once per session)
# ─────────────────────────────────────────────

# !pip install -q unsloth pymupdf trl peft accelerate bitsandbytes datasets transformers sentencepiece


# ─────────────────────────────────────────────
# CELL 2 ─ Download the two PDF books from GitHub
# ─────────────────────────────────────────────

import urllib.request

REPO_RAW = "https://raw.githubusercontent.com/Dr-Yehia/stability-book/claude/create-book-file-pzuwA"
BOOKS    = [
    "1structural stability w.f.chen.pdf",
    "2structural stability w.f.chen.pdf",
]
BOOK_DIR = "/content/books"
os.makedirs(BOOK_DIR, exist_ok=True)

for book in BOOKS:
    dest = f"{BOOK_DIR}/{book}"
    if not os.path.exists(dest):
        url = REPO_RAW + "/" + urllib.parse.quote(book)
        print(f"Downloading: {book} ...")
        urllib.request.urlretrieve(url, dest)
    print(f"  OK: {dest}")


# ─────────────────────────────────────────────
# CELL 3 ─ Extract text from PDFs
# ─────────────────────────────────────────────

import fitz  # PyMuPDF

def extract_pdf_text(pdf_path):
    doc  = fitz.open(pdf_path)
    text = "\n".join(p.get_text("text") for p in doc if p.get_text("text").strip())
    doc.close()
    return text

full_text = ""
for book in BOOKS:
    t = extract_pdf_text(f"{BOOK_DIR}/{book}")
    full_text += f"\n\n{'='*60}\n# {book}\n{'='*60}\n\n" + t
    print(f"{book}: {len(t):,} chars")

print(f"\nTotal: {len(full_text):,} chars")


# ─────────────────────────────────────────────
# CELL 4 ─ Chunk & format training examples
# ─────────────────────────────────────────────

CHUNK_SIZE = 1024
OVERLAP    = 128

SYSTEM_PROMPT = (
    "You are an expert structural engineer trained exclusively on "
    "'Structural Stability' by W.F. Chen. Answer every question with "
    "complete accuracy, matching the book's notation, equations, and steps."
)

def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        c = text[start:start + size].strip()
        if len(c) > 100:
            chunks.append(c)
        start += size - overlap
    return chunks

def format_example(chunk):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Explain the following passage from the structural stability textbook:\n\n{chunk}<|im_end|>\n"
        f"<|im_start|>assistant\n{chunk}<|im_end|>"
    )

chunks        = chunk_text(full_text)
training_data = [format_example(c) for c in chunks]
print(f"Training examples: {len(training_data)}")


# ─────────────────────────────────────────────
# CELL 5 ─ Load base model (Mistral-7B, 4-bit)
# ─────────────────────────────────────────────

from unsloth import FastLanguageModel

MAX_SEQ_LEN = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    max_seq_length = MAX_SEQ_LEN,
    dtype          = None,
    load_in_4bit   = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r                          = 16,
    target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
    lora_alpha                 = 32,
    lora_dropout               = 0.05,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",
    random_state               = 42,
)
print(model.print_trainable_parameters())


# ─────────────────────────────────────────────
# CELL 6 ─ Train
# ─────────────────────────────────────────────

from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

dataset = Dataset.from_dict({"text": training_data}).train_test_split(test_size=0.05, seed=42)
print(f"Train: {len(dataset['train'])}  |  Eval: {len(dataset['test'])}")

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = dataset["train"],
    eval_dataset       = dataset["test"],
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LEN,
    packing            = True,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        num_train_epochs            = 3,
        warmup_steps                = 20,
        learning_rate               = 2e-4,
        fp16                        = not is_bfloat16_supported(),
        bf16                        = is_bfloat16_supported(),
        logging_steps               = 10,
        evaluation_strategy         = "steps",
        eval_steps                  = 50,
        output_dir                  = "/content/checkpoints",
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = "cosine",
        seed                        = 42,
        report_to                   = "none",
    ),
)

stats = trainer.train()
print(f"Done. Loss: {stats.training_loss:.4f}")


# ─────────────────────────────────────────────
# CELL 7 ─ Save weights + package into ONE ZIP file
#          then auto-download it (click will appear in output)
# ─────────────────────────────────────────────

import shutil
from google.colab import files

WEIGHTS_DIR = "/content/stability_weights"
ZIP_PATH    = "/content/stability_model_weights"   # .zip appended automatically

os.makedirs(WEIGHTS_DIR, exist_ok=True)

# 1. Save LoRA adapter (~100 MB)
print("Saving LoRA adapter weights...")
model.save_pretrained(WEIGHTS_DIR)
tokenizer.save_pretrained(WEIGHTS_DIR)

# 2. Save a ready-to-use inference script inside the folder
inference_script = '''
import os, torch
from unsloth import FastLanguageModel

SYSTEM_PROMPT = (
    "You are an expert structural engineer trained exclusively on "
    "'Structural Stability' by W.F. Chen. Answer every question with "
    "complete accuracy, matching the book\'s notation, equations, and steps."
)

MODEL_DIR   = os.path.dirname(os.path.abspath(__file__))
MAX_SEQ_LEN = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_DIR,
    max_seq_length = MAX_SEQ_LEN,
    dtype          = None,
    load_in_4bit   = True,
)
FastLanguageModel.for_inference(model)

def ask(question, max_new_tokens=512):
    prompt = (
        f"<|im_start|>system\\n{SYSTEM_PROMPT}<|im_end|>\\n"
        f"<|im_start|>user\\n{question}<|im_end|>\\n"
        f"<|im_start|>assistant\\n"
    )
    inputs  = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens     = max_new_tokens,
        temperature        = 0.1,
        do_sample          = True,
        repetition_penalty = 1.1,
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("<|im_start|>assistant")[-1].strip()

if __name__ == "__main__":
    print("Structural Stability Expert Model (W.F. Chen)")
    print("Type your question or \'quit\' to exit.\\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in ("quit", "exit", ""):
            break
        print("\\nAnswer:", ask(q), "\\n")
'''

with open(f"{WEIGHTS_DIR}/inference.py", "w") as f:
    f.write(inference_script)

# 3. Write a README inside the ZIP
readme = """# Structural Stability Model (W.F. Chen)

## Contents
- adapter_model.safetensors  ← LoRA weights (the trained knowledge)
- adapter_config.json
- tokenizer files
- inference.py               ← run this to chat with the model

## Requirements
    pip install unsloth torch transformers

## Usage
    python inference.py

## Base model
    unsloth/mistral-7b-instruct-v0.2-bnb-4bit
"""
with open(f"{WEIGHTS_DIR}/README.txt", "w") as f:
    f.write(readme)

# 4. Compress everything into a single ZIP
print("Creating ZIP file...")
shutil.make_archive(ZIP_PATH, "zip", WEIGHTS_DIR)
zip_size_mb = os.path.getsize(ZIP_PATH + ".zip") / 1e6
print(f"ZIP created: {ZIP_PATH}.zip  ({zip_size_mb:.1f} MB)")

# 5. Trigger download — a download link appears in the cell output
print("Starting download...")
files.download(ZIP_PATH + ".zip")
print("Done! Check your browser downloads.")


# ─────────────────────────────────────────────
# CELL 8 ─ Quick test before downloading
# ─────────────────────────────────────────────

from unsloth import FastLanguageModel
FastLanguageModel.for_inference(model)

def ask(question, max_new_tokens=512):
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs  = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens     = max_new_tokens,
        temperature        = 0.1,
        do_sample          = True,
        repetition_penalty = 1.1,
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("<|im_start|>assistant")[-1].strip()

# Test with a sample question
print(ask("What is the Euler buckling load for a pinned-pinned column?"))
