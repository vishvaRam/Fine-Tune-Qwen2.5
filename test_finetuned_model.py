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
    device_map = "cpu"
else:
    device_map = "auto"

# Load the base model with 4-bit quantization
print(f"Loading base model {base_model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config={
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    },
    device_map=device_map,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load the LoRA adapter
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(model, lora_model_path)

# Set to evaluation mode
model.eval()

# Function to generate a response and calculate time
def generate_response(instruction, context=""):
    prompt = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,  # Reduced max_new_tokens
            do_sample=False,  # Greedy decoding
        )
    end_time = time.time()

    generation_time = end_time - start_time
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_only = response.split("### Response:")[1].strip()
    return response_only, generation_time

# Test with some examples
test_instructions = [
    {
        "instruction": "Explain the differences between supervised and unsupervised machine learning.",
        "context": ""
    },
    {
        "instruction": "Create a short story about a robot discovering emotions.",
        "context": "The robot's name is Alex-7 and it works in a human household."
    },
    {
        "instruction": "Summarize the key features of quantum computing.",
        "context": ""
    }
]

# Generate and print responses for test instructions
print("\n" + "=" * 50 + "\nTesting fine-tuned model\n" + "=" * 50)
for i, test in enumerate(test_instructions):
    print(f"\nTest {i + 1}:")
    print(f"Instruction: {test['instruction']}")
    if test['context']:
        print(f"Context: {test['context']}")

    response, generation_time = generate_response(test['instruction'], test['context'])
    print("\nGenerated Response:")
    print(response)
    print(f"\nGeneration Time: {generation_time:.4f} seconds")
    print("-" * 50)


# Total Generation Time for 3 samples: 17.7721 seconds
# Average Generation Time per sample: 5.9240 seconds
