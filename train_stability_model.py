# =============================================================
# train_stability_model.py
# Fine-tune an LLM on W.F. Chen - Structural Stability (2 vols)
# Run cell by cell in Google Colab (GPU: T4 or A100)
# =============================================================


# ─────────────────────────────────────────────
# CELL 1 ─ Install dependencies
# ─────────────────────────────────────────────

# Run this once per Colab session
# !pip install -q unsloth pymupdf trl peft accelerate bitsandbytes datasets transformers sentencepiece


# ─────────────────────────────────────────────
# CELL 2 ─ Mount Google Drive (to save weights)
# ─────────────────────────────────────────────

from google.colab import drive
drive.mount('/content/drive')

SAVE_DIR = "/content/drive/MyDrive/stability_model_weights"  # change if needed


# ─────────────────────────────────────────────
# CELL 3 ─ Download the two PDF books from GitHub
# ─────────────────────────────────────────────

import os
import urllib.request

REPO_RAW = "https://raw.githubusercontent.com/Dr-Yehia/stability-book/claude/create-book-file-pzuwA"

BOOKS = [
    "1structural stability w.f.chen.pdf",
    "2structural stability w.f.chen.pdf",
]

os.makedirs("/content/books", exist_ok=True)

for book in BOOKS:
    dest = f"/content/books/{book}"
    if not os.path.exists(dest):
        url = REPO_RAW + "/" + urllib.parse.quote(book)
        print(f"Downloading: {book}")
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved to {dest}")
    else:
        print(f"  Already exists: {dest}")

import urllib.parse  # needed above — move import to top in Colab


# ─────────────────────────────────────────────
# CELL 4 ─ Extract text from both PDFs
# ─────────────────────────────────────────────

import fitz  # PyMuPDF

def extract_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n".join(pages)

print("Extracting text from books...")
full_text = ""
for book in BOOKS:
    path = f"/content/books/{book}"
    text = extract_pdf_text(path)
    full_text += f"\n\n{'='*60}\n# {book}\n{'='*60}\n\n" + text
    print(f"  {book}: {len(text):,} characters")

print(f"\nTotal text: {len(full_text):,} characters")

# Save raw text for inspection
with open("/content/books/full_text.txt", "w", encoding="utf-8") as f:
    f.write(full_text)
print("Raw text saved to /content/books/full_text.txt")


# ─────────────────────────────────────────────
# CELL 5 ─ Chunk text into training examples
# ─────────────────────────────────────────────

from typing import List

CHUNK_SIZE   = 1024   # characters per chunk
OVERLAP      = 128    # overlap between chunks

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if len(chunk) > 100:          # skip tiny leftovers
            chunks.append(chunk)
        start += size - overlap
    return chunks

chunks = chunk_text(full_text)
print(f"Total training chunks: {len(chunks)}")


# Format each chunk as an instruction-following example
SYSTEM_PROMPT = (
    "You are an expert structural engineer trained exclusively on "
    "'Structural Stability' by W.F. Chen. Answer every question with "
    "complete accuracy, matching the book's notation, equations, and steps."
)

def format_example(chunk: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Explain the following passage from the structural stability textbook:\n\n"
        f"{chunk}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{chunk}<|im_end|>"
    )

training_texts = [format_example(c) for c in chunks]
print(f"Training examples ready: {len(training_texts)}")


# ─────────────────────────────────────────────
# CELL 6 ─ Load base model with Unsloth (QLoRA)
# ─────────────────────────────────────────────

from unsloth import FastLanguageModel
import torch

MODEL_NAME   = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"  # 4-bit: fits T4 GPU
MAX_SEQ_LEN  = 2048
DTYPE        = None   # auto-detect (float16 for T4, bfloat16 for A100)
LOAD_IN_4BIT = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LEN,
    dtype           = DTYPE,
    load_in_4bit    = LOAD_IN_4BIT,
)

# Apply LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r               = 16,       # LoRA rank (16 = good balance)
    target_modules  = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    lora_alpha      = 32,
    lora_dropout    = 0.05,
    bias            = "none",
    use_gradient_checkpointing = "unsloth",  # saves VRAM
    random_state    = 42,
)

print(model.print_trainable_parameters())


# ─────────────────────────────────────────────
# CELL 7 ─ Build HuggingFace Dataset
# ─────────────────────────────────────────────

from datasets import Dataset

dataset = Dataset.from_dict({"text": training_texts})
dataset = dataset.train_test_split(test_size=0.05, seed=42)

print(f"Train: {len(dataset['train'])} | Eval: {len(dataset['test'])}")


# ─────────────────────────────────────────────
# CELL 8 ─ Fine-tune with SFTTrainer
# ─────────────────────────────────────────────

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model            = model,
    tokenizer        = tokenizer,
    train_dataset    = dataset["train"],
    eval_dataset     = dataset["test"],
    dataset_text_field = "text",
    max_seq_length   = MAX_SEQ_LEN,
    dataset_num_proc = 2,
    packing          = True,   # packs short sequences → faster training
    args = TrainingArguments(
        per_device_train_batch_size  = 2,
        gradient_accumulation_steps  = 4,
        num_train_epochs             = 3,      # increase for deeper learning
        warmup_steps                 = 20,
        learning_rate                = 2e-4,
        fp16                         = not is_bfloat16_supported(),
        bf16                         = is_bfloat16_supported(),
        logging_steps                = 10,
        evaluation_strategy          = "steps",
        eval_steps                   = 50,
        save_steps                   = 100,
        output_dir                   = "/content/checkpoints",
        optim                        = "adamw_8bit",
        weight_decay                 = 0.01,
        lr_scheduler_type            = "cosine",
        seed                         = 42,
        report_to                    = "none",
    ),
)

print("Starting training...")
trainer_stats = trainer.train()
print("Training complete!")
print(f"  Loss: {trainer_stats.training_loss:.4f}")


# ─────────────────────────────────────────────
# CELL 9 ─ Save LoRA weights to Google Drive
# ─────────────────────────────────────────────

import os
os.makedirs(SAVE_DIR, exist_ok=True)

# Save LoRA adapter only (~50-150 MB)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"LoRA weights saved to: {SAVE_DIR}")

# Optional: save merged full model (large, ~14 GB) — uncomment if needed
# model.save_pretrained_merged(
#     SAVE_DIR + "_merged_16bit",
#     tokenizer,
#     save_method="merged_16bit",
# )

# Optional: save as GGUF (for llama.cpp / Ollama) — uncomment if needed
# model.save_pretrained_gguf(
#     SAVE_DIR + "_gguf",
#     tokenizer,
#     quantization_method="q4_k_m",
# )


# ─────────────────────────────────────────────
# CELL 10 ─ Load saved weights & chat / inference
# ─────────────────────────────────────────────
# Run this cell independently after training to use the model

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name   = SAVE_DIR,     # path to your saved LoRA weights
    max_seq_length = MAX_SEQ_LEN,
    dtype        = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)  # enable native 2x faster inference

def ask(question: str, max_new_tokens: int = 512) -> str:
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens    = max_new_tokens,
        temperature       = 0.1,   # low = more deterministic / accurate
        do_sample         = True,
        repetition_penalty = 1.1,
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # return only the assistant reply
    return result.split("<|im_start|>assistant")[-1].strip()


# ── Example questions ──
print(ask("What is the Euler buckling load formula for a pinned-pinned column?"))
print(ask("Explain the moment-curvature relationship in beam-column analysis."))
print(ask("What are the assumptions in the classical stability theory of plates?"))
