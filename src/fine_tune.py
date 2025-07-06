from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

model_name = "meta-llama/Llama-2-7b-hf"

dataset = load_dataset("json", data_files="data/training_dataset.jsonl")["train"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

dataset = dataset.map(tokenize)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    load_in_4bit=True
)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, peft_config)

args = TrainingArguments(
    output_dir="./llama-finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=2,
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

trainer.train()
