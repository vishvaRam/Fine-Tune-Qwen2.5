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
    device_map={"": 0} if torch.cuda.is_available() else "cpu", # Explicitly map to GPU 0 if available
    trust_remote_code=True,
    # attn_implementation="flash_attention_2"
)

# Compile the model for potential speedup (requires PyTorch 2.0+)
if torch.cuda.is_available():
    model = torch.compile(model)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load the LoRA adapter
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(model, lora_model_path).to(device)
model.eval()

# Function to generate a response and calculate time (now handles batched inputs)
def generate_responses(instructions, contexts=None):
    if contexts is None:
        contexts = [""] * len(instructions)
    prompts = [f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n"
               for instruction, context in zip(instructions, contexts)]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=512,
            do_sample=False,
        )
    end_time = time.time()

    generation_time = end_time - start_time
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    response_only = [res.split("### Response:")[1].strip() for res in responses]
    return response_only, generation_time

# Test with some examples (batched)
test_instructions_batch = [
    "Explain the differences between supervised and unsupervised machine learning.",
    "Create a short story about a robot discovering emotions.",
    "Summarize the key features of quantum computing."
]
test_contexts_batch = [
    "",
    "The robot's name is Alex-7 and it works in a human household.",
    ""
]

# Generate and print responses for batched test instructions
print("\n" + "=" * 50 + "\nTesting fine-tuned model (batched)\n" + "=" * 50)
responses, generation_time = generate_responses(test_instructions_batch, test_contexts_batch)
average_generation_time = generation_time / len(test_instructions_batch)

for i, test in enumerate(test_instructions_batch):
    print(f"\nTest {i + 1}:")
    print(f"Instruction: {test}")
    if test_contexts_batch[i]:
        print(f"Context: {test_contexts_batch[i]}")
    print("\nGenerated Response:")
    print(responses[i])
    print("-" * 50)

print(f"\nTotal Generation Time for {len(test_instructions_batch)} samples: {generation_time:.4f} seconds")
print(f"Average Generation Time per sample: {average_generation_time:.4f} seconds")


# Total Generation Time for 3 samples: 17.7721 seconds
# Average Generation Time per sample: 5.9240 seconds
