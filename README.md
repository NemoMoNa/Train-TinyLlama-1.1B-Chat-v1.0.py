# Train-TinyLlama-1.1B-Chat-v1.0.py
Detest: FreedomIntelligence/medical-o1-reasoning-SFT

# TinyLlama Medical Q&A Fine-Tuning (LoRA)

This project fine-tunes `TinyLlama-1.1B-Chat-v1.0` using a medical dataset (`FreedomIntelligence/medical-o1-reasoning-SFT`) with LoRA on a Mac M4 Pro.

## 🔧 Model

- Model: TinyLlama-1.1B-Chat
- Fine-tuning: LoRA (r=8, α=32)
- Training set: 100 samples (for testing purposes)
- Output: adapter weights, tokenizer, and evaluation

## 📂 Structure

- `Train_TinyLlama-1.1B-Chat-v1.0.py`: Full training script
- `tinyllama-medqa-lora/`: LoRA model output (adapter, tokenizer)
- `results/`: Training logs (optional)

## 🚀 Run

```bash
python Train_TinyLlama-1.1B-Chat-v1.0.py

##📄 TEST
input_text = "### Instruction:\nExplain symptoms of anemia\n\n### Response:\n"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))

---
### Instruction:
Explain symptoms of anemia

### Response:
Anemia is a condition where the body does not have enough red blood cells to carry oxygen to the body's tissues. This can happen due to a variety of reasons, including a lack of iron in the diet, a deficiency in red blood cells, or a decrease in the number of red blood cells.

Symptoms of anemia can vary depending on the cause, but they typically include fatigue, weakness, shortness of breath, and a decre

## License

This project is licensed under the Apache License 2.0.  
The original model [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)  
and dataset [medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)  
are also licensed under Apache-2.0.
