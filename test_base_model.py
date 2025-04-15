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

def generate_response(prompt):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    start_time = time.time()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    end_time = time.time()
    generation_time = end_time - start_time

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response, generation_time

test_instructions = [
    "Explain the differences between supervised and unsupervised machine learning.",
    "Create a short story about a robot discovering emotions.",
    "Summarize the key features of quantum computing."
]

print("\n" + "=" * 50 + "\nTesting base model\n" + "=" * 50)
for i, prompt in enumerate(test_instructions):
    print(f"\nTest {i + 1}:")
    print(f"Prompt: {prompt}")

    response, generation_time = generate_response(prompt)
    print("\nGenerated Response:")
    print(response)
    print(f"\nGeneration Time: {generation_time:.4f} seconds")
    print("-" * 50)
