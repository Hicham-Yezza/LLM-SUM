import wandb
from datasets import load_dataset, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer

# 4 Sep

# Initialize Weights & Biases
wandb.init(project="GPT-CNN-1")

print("Loading CNN/Daily Mail dataset...")
cnn_dailymail_dataset = load_dataset("cnn_dailymail", "3.0.0")
print("Dataset loaded successfully.")

print("Initializing the GPT-2 tokenizer and setting its padding token...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer initialized successfully.")

print("Tokenizing the dataset...")
def tokenize_function(examples):
    return {
        "input_ids": tokenizer(examples["article"], padding="max_length", truncation=True, max_length=512).input_ids,
        "labels": tokenizer(examples["highlights"], padding="max_length", truncation=True, max_length=150).input_ids
    }

# Apply batched tokenization
tokenized_datasets = cnn_dailymail_dataset.map(tokenize_function, batched=True, batch_size=1000)
print("Dataset tokenized successfully.")

print("Initializing the GPT-2 model...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
print("Model initialized successfully.")

print("Defining training arguments...")
training_args = TrainingArguments(
    output_dir="C://GPT2",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=500,
    save_steps=5000,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=5000,
    push_to_hub=False,
    logging_first_step=True,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
)
print("Training arguments defined successfully.")

print("Initializing the Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)
print("Trainer initialized successfully.")

print("Starting fine-tuning...")
trainer.train()
print("Training completed.")
