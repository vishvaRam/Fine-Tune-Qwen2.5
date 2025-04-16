import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# 1. Set paths
base_model_name = "Qwen/Qwen2.5-0.5B"  # Base model name
lora_adapter_path = "./qwen2_5_dolly_qlora"  # Path to your fine-tuned LoRA adapters
merged_model_save_path = "./qwen2_5_dolly_merged_fp16"  # Where to save merged model

# 2. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# 3. Load base model in fp16 (not quantized)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)

# 4. Load LoRA adapters
model = PeftModel.from_pretrained(base_model, lora_adapter_path)

# 5. Merge LoRA into base model (this makes the adapters permanent)
model = model.merge_and_unload()

# 6. Save merged model and tokenizer
model.save_pretrained(merged_model_save_path)
tokenizer.save_pretrained(merged_model_save_path)

print(f"✅ Merged model saved at: {merged_model_save_path}")
