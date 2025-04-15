import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

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

print("\n" + "=" * 50 + "\nTesting base model (batched)\n" + "=" * 50)
responses, total_generation_time, average_generation_time = generate_responses_batched(test_instructions)

for i, prompt in enumerate(test_instructions):
    print(f"\nTest {i + 1}:")
    print(f"Prompt: {prompt}")
    print("\nGenerated Response:")
    print(responses[i])
    print("-" * 50)

print(f"\nTotal Generation Time for {len(test_instructions)} samples: {total_generation_time:.4f} seconds")
print(f"Average Generation Time per sample: {average_generation_time:.4f} seconds")


# Total Generation Time for 3 samples: 8.6223 seconds
# Average Generation Time per sample: 2.8741 seconds
