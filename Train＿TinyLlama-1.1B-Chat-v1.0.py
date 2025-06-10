#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 21:46:25 2025

@author: mh
"""

# 🧠 モジュールのインポート
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import torch

# ✅ ステップ 1: データ読み込み
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train[:100]") # small test


# ✅ ステップ 2: Chat形式に変換（必要に応じて）
def format_chat(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    }
def format_chat(example):
    return {
        "text": f"### Instruction:\n{example['Question']}\n\n### Chain-of-Thought:\n{example['Complex_CoT']}\n\n### Response:\n{example['Response']}"
    }
#n### Chain-of-Thought:\n{example['Complex_CoT']}\n\
    

dataset = dataset.map(format_chat)


# ✅ ステップ 3: モデル＆トークナイザー読み込み
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# ✅ ステップ 4: LoRA設定
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)


# ✅ ステップ 5: トークナイズ
def tokenize(example):
    # トークン化関数：1つ（または複数）の "text" をトークンに変換
    return tokenizer(
        example["text"],              # → "text" フィールドの中身を対象に
        truncation=True,              # → 長さが長すぎる場合は切り捨てる
        padding="max_length",         # → 長さが足りない場合は最大長までパディング
        max_length=512                # → トークン長の上限を512に固定
    )

# データセット全体にトークン化関数を適用
tokenized_dataset = dataset.map(tokenize, batched=True)


# ✅ ステップ 6: 学習設定
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=False,
    report_to="none"
)


# ✅ ステップ 7: Trainer定義
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ✅ ステップ 8: 学習＆保存
trainer.train()

# ✅ ステップ 9: 保存（モデル＋トークナイザー）
model.save_pretrained("tinyllama-medqa-lora")
tokenizer.save_pretrained("tinyllama-medqa-lora") 



# ✅ ステップ 10: 任意評価（Trainer.evaluate）
input_text = "### Instruction:\nExplain symptoms of anemia\n\n### Response:\n"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))




