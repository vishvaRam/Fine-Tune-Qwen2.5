import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,  # Import BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import matplotlib.pyplot as plt

# Check for CUDA availability
if not torch.cuda.is_available():
    print("CUDA not available, using CPU. This will be very slow.")
    device_map = "cpu"
else:
    device_map = "auto"

# 1. Define the model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token if it's missing
tokenizer.padding_side = "left" # IMPORTANT: Set padding_side to 'left' BEFORE tokenizing

# 2. Load the dataset
dataset_name = "databricks/databricks-dolly-15k"  # Replace with your desired dataset
dataset = load_dataset(dataset_name, split="train[:6000]")

# 1. Format the dataset first
def format_dolly(sample):
    instruction = sample["instruction"]
    context = sample["context"]
    response = sample["response"]
    prompt = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}"
    return {"text": prompt}

# Apply formatting
dataset = dataset.map(format_dolly).filter(lambda x: x is not None and x["text"] is not None)

# 2. Now tokenize the formatted data
def tokenize_function(examples):
    # Tokenize the texts with padding and truncation
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Apply tokenization to create input_ids, attention_mask, etc.
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["instruction", "context", "response", "category", "text"],
)

# Split the tokenized dataset
train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# 3. Configure QLoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# 4. Load the base model in 4-bit quantization with BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)

# Prepare the model for k-bit training
model = prepare_model_for_kbit_training(model)

# Add LoRA adapters to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 5. Set up training arguments
output_dir = "./qwen2_5_dolly_qlora"  # Directory to save fine-tuned model
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_train_epochs=3,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    push_to_hub=False,
    remove_unused_columns=False,
    logging_dir="./logs",  # Add logging directory
    logging_steps=10,  # Log every 10 steps
    report_to="none",  # remove default reports, if you want to use wandb, or tensorboard, keep it, and install those libraries.
)

# 6. Set up data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# 7. Initialize the Trainer
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    data_collator=data_collator,
)

# 8. Start training
train_result = trainer.train()

# 9. Save the fine-tuned LoRA adapters
model.save_pretrained(output_dir)

# Store training and evaluation metrics
train_history = train_result.metrics
eval_history = trainer.evaluate()

# Extract loss values
train_loss = [train_history[f"train_loss_epoch"]]
eval_loss = [eval_history["eval_loss"]]

# Extract epochs
epochs = range(1, len(train_loss) + 1)

# Plotting the training and evaluation loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label="Training Loss")
plt.plot(epochs, eval_loss, label="Evaluation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Evaluation Loss")
plt.legend()
plt.savefig(f"{output_dir}/loss_plot.png")  # Save the plot
plt.show()

# To later load the LoRA adapters for inference:
# from peft import PeftModel, PeftConfig
#
# peft_config = PeftConfig.from_pretrained(output_dir)
# model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, torch_dtype=torch.float16, device_map=device_map, trust_remote_code=True)
# model = PeftModel.from_pretrained(model, output_dir)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
