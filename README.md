# FineTune-Qwen2.5-0.5B

This repository contains scripts and utilities for fine-tuning, testing, and merging LoRA adapters with the Qwen 2.5-0.5B model. The project leverages Hugging Face's `transformers` library and PEFT (Parameter-Efficient Fine-Tuning) techniques to efficiently fine-tune large language models.

---

## Features

- **Fine-Tuning with QLoRA**: Efficiently fine-tune the Qwen 2.5-0.5B model using LoRA adapters and 4-bit quantization.
- **Model Testing**: Test the base model, fine-tuned model, and merged model with various configurations.
- **Parameter Inspection**: Inspect and analyze the parameters of the base model.
- **Model Merging**: Merge LoRA adapters into the base model for deployment.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/FineTune-Qwen2.5-0.5B.git
   cd FineTune-Qwen2.5-0.5B
   ```
2. Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```
## Usage
1. Fine-Tuning the Model
To fine-tune the Qwen 2.5-0.5B model using LoRA adapter
  ```bash
  python train.py
  ```
This will save the fine-tuned LoRA adapters in the ./qwen2_5_dolly_qlora directory.

2. Merging LoRA Adapters into the Base Model
To merge the fine-tuned LoRA adapters into the base model, run:
  ```bash
  python merged_QLoRA_to_Base.py
  ```

## Dependencies
The project requires the following Python libraries:

- transformers
- peft
- torch
- datasets
- matplotlib
- bitsandbytes
  
For the full list of dependencies, see requirements.txt.

## Acknowledgments
- Qwen 2.5-0.5B: A large language model by Alibaba Cloud.
- Hugging Face Transformers: For providing the tools to work with large language models.
- PEFT: For enabling parameter-efficient fine-tuning.
