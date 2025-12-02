import torch
import os
import sys
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
from pathlib import Path





JSONL_FILE = Path("output/Emotion_EEG/Jsonl_For_Llama3/Train_Data.jsonl")


OUTPUT_DIR = Path("output/Emotion_EEG/Llama3_Result")






MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"출력 폴더를 생성했습니다: {OUTPUT_DIR}")


OPTIMAL_EPOCHS = 7
OPTIMAL_BATCH_SIZE = 1
LOGGING_FREQUENCY = 5
ACCUMULATION_STEPS = 8
GROUP_BY_LENGTH_FLAG = False






bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id





peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
print("Model successfully converted to PEFT (LoRA) model.")







print(f"Loading dataset from {JSONL_FILE}...")
try:
    
    dataset = load_dataset("json", data_files=str(JSONL_FILE), split="train")
except Exception as e:
    print(
        f"데이터셋 로드 오류: {e}. 'datasets' 라이브러리가 설치되어 있는지 확인하세요."
    )
    exit()



def apply_chat_template_to_text(example):
           
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}



dataset = dataset.map(
    apply_chat_template_to_text,
    remove_columns=dataset.column_names,
    desc="Applying chat template and creating 'text' column",
)



split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")






training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=OPTIMAL_EPOCHS,
    per_device_train_batch_size=OPTIMAL_BATCH_SIZE,
    gradient_accumulation_steps=ACCUMULATION_STEPS,
    optim="paged_adamw_8bit",
    save_strategy="no",
    eval_strategy="epoch",
    logging_steps=LOGGING_FREQUENCY,
    save_total_limit=0,
    learning_rate=3e-4,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    group_by_length=GROUP_BY_LENGTH_FLAG,
    report_to="none",
)


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  
    processing_class=tokenizer,
)




print("Starting training...")
trainer.train()


print("Moving adapter to CPU and saving (adapter-only, safetensors)...")

model.eval()
model.to("cpu")  
import torch

torch.cuda.empty_cache()  


model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Adapter & tokenizer saved to {OUTPUT_DIR}")






try:
    
    import gc

    gc.collect()

    
    torch.cuda.empty_cache()

    print("CUDA context cleanup initiated. Script will now terminate.")

except Exception as e:
    print(f"Final cleanup error: {e}")


import sys

sys.exit(0)  
