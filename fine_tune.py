
import math
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate
from tqdm import tqdm

# Config 
BASE_MODEL = "gpt2-medium"
DATA_PATH = Path("cleaned_fin_q&a_enhanced.json")
OUT_DIR = "model_checkpoints2/"
SEED = 42

MAX_LEN = 256
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 8
EPOCHS = 2
LR = 2e-5
FP16 = torch.cuda.is_available()

# LoRA params
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["c_attn", "c_fc", "c_proj"]

GEN_MAX_NEW_TOKENS = 120


random.seed(SEED)
torch.manual_seed(SEED)


if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

ds = load_dataset("json", data_files=str(DATA_PATH))["train"]
ds = ds.shuffle(seed=SEED)
split = ds.train_test_split(test_size=0.1, seed=SEED)
train_raw, eval_raw = split["train"], split["test"]

print(f"Train size: {len(train_raw)}  Eval size: {len(eval_raw)}")


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def preprocess(example):
    prompt = (example.get("prompt") or "").strip()
    completion = (example.get("completion") or "").strip()
    full = prompt + "\n" + completion + (tokenizer.eos_token or "")
    tok = tokenizer(full, truncation=True, padding="max_length", max_length=MAX_LEN)

    input_ids = tok["input_ids"]
    attention_mask = tok["attention_mask"]

    prompt_ids = tokenizer(prompt + "\n", add_special_tokens=False)["input_ids"]
    prompt_len = min(len(prompt_ids), MAX_LEN)

    labels = list(input_ids)  
    for i in range(prompt_len):
        labels[i] = -100
    for i, m in enumerate(attention_mask):
        if m == 0:
            labels[i] = -100
    tok["labels"] = labels
    return tok

print("Tokenizing train dataset...")
train_tok = train_raw.map(preprocess, remove_columns=train_raw.column_names)
print("Tokenizing eval dataset...")
eval_tok = eval_raw.map(preprocess, remove_columns=eval_raw.column_names)


print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if FP16 else torch.float32,
    low_cpu_mem_usage=True,
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
)
model = get_peft_model(model, lora_config)


found = False
for n, p in model.named_parameters():
    if "lora" in n.lower():
        p.requires_grad = True
        found = True
if not found:
    print("[WARN] No LoRA params found — enabling grad for all params.")
    for p in model.parameters():
        p.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"[INFO] Trainable params: {trainable_params} / {all_params} "
      f"({100 * trainable_params/all_params:.2f}%)")

if trainable_params == 0:
    raise RuntimeError("❌ No parameters require grad — aborting training!")


training_args = TrainingArguments(
    output_dir=str(OUT_DIR),
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=FP16,
    save_strategy="steps",
    save_steps=200, 
    logging_strategy="steps",
    logging_steps=200,
    save_total_limit=3,
    remove_unused_columns=False,
    seed=SEED,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    data_collator=default_data_collator,
    tokenizer=tokenizer,
)


print("Starting training...")
checkpoint_path = "model_checkpoints\checkpoint-6592"
#resume_from_checkpoint=checkpoint_path
train_result = trainer.train()
trainer.save_model(str(OUT_DIR))
tokenizer.save_pretrained(str(OUT_DIR))
print("✅ Training complete. Saved to", OUT_DIR)

print("Running generation on eval set for BLEU/ROUGE/BERTScore...")
bleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

preds, refs, records = [], [], []

for item in tqdm(eval_raw):
    prompt = (item.get("prompt") or "").strip()
    reference = (item.get("completion") or "").strip()
    full_prompt = prompt + "\n"
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=False)
    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    if "\n" in raw:
        answer = raw.split("\n", 1)[-1].strip()
    else:
        if raw.startswith(full_prompt):
            answer = raw[len(full_prompt):].strip()
        else:
            answer = raw.strip()
    preds.append(answer)
    refs.append(reference)
    records.append({"prompt": prompt, "ref": reference, "pred": answer})

bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs])
rouge_res = rouge.compute(predictions=preds, references=refs)
bert_res = bertscore.compute(predictions=preds, references=refs, lang="en")

print("\n=== Evaluation results ===")
print("BLEU:", bleu_res)
print("ROUGE:", rouge_res)
print("BERTScore (avg):", {
    "precision": sum(bert_res["precision"]) / len(bert_res["precision"]),
    "recall": sum(bert_res["recall"]) / len(bert_res["recall"]),
    "f1": sum(bert_res["f1"]) / len(bert_res["f1"]),
})

with open("eval_predictions1.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
print("Saved eval predictions to eval_predictions.json")
