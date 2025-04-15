import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import time

# Path to the fine-tuned LoRA model adapters
lora_model_path = "./qwen2_5_dolly_qlora"

# Load the PEFT configuration
peft_config = PeftConfig.from_pretrained(lora_model_path)

# Get the original model name
base_model_name = peft_config.base_model_name if hasattr(peft_config, "base_model_name") else "Qwen/Qwen2.5-0.5B"

# Set device configuration
if not torch.cuda.is_available():
    print("CUDA not available, using CPU. This will be slower.")
    device = torch.device("cpu")
else:
    device = torch.device("cuda")

# Load the base model WITHOUT quantization
print(f"Loading base model {base_model_name} without quantization...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map={"": 0} if torch.cuda.is_available() else "cpu",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16, # Use float16 for potentially better GPU utilization
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load the LoRA adapter
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(model, lora_model_path).to(device).eval()

def generate_responses_batched(prompts):
    text_batch = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        text_batch.append(text)

    model_inputs = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True).to(model.device)

    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
    end_time = time.time()
    generation_time = end_time - start_time

    generated_responses = []
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
        generated_token_ids = output_ids[len(input_ids):]
        response = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        generated_responses.append(response)

    average_generation_time = generation_time / len(prompts)
    return generated_responses, generation_time, average_generation_time

test_instructions = [
    "Explain the differences between supervised and unsupervised machine learning.",
    "Create a short story about a robot discovering emotions.",
    "Summarize the key features of quantum computing."
]

print("\n" + "=" * 50 + "\nTesting LoRA fine-tuned model WITHOUT quantization (for debugging)\n" + "=" * 50)
responses, total_generation_time, average_generation_time = generate_responses_batched(test_instructions)

for i, prompt in enumerate(test_instructions):
    print(f"\nTest {i + 1}:")
    print(f"Prompt: {prompt}")
    print("\nGenerated Response:")
    print(responses[i])
    print("-" * 50)

print(f"\nTotal Generation Time for {len(test_instructions)} samples: {total_generation_time:.4f} seconds")
print(f"Average Generation Time per sample: {average_generation_time:.4f} seconds")

# Total Generation Time for 3 samples: 14.3472 seconds
# Average Generation Time per sample: 4.7824 seconds
