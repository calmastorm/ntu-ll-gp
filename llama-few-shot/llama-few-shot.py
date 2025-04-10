import torch
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import evaluate

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType

# Fixed randomness
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and tokenizer
model_name = "openlm-research/open_llama_3b_v2"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Soft prompt configuration
peft_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="Answer the question based on the context:",
    num_virtual_tokens=20,
    tokenizer_name_or_path=model_name,
)

model = get_peft_model(model, peft_config)
model.eval()

# Load dataset and metric
dataset = load_dataset("squad", split="validation")
metric = evaluate.load("squad")

# Few-shot examples
few_shot_examples = [
    {
        "context": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        "question": "Where is the Eiffel Tower located?",
        "answer": "The Eiffel Tower is located in Paris, France."
    },
    {
        "context": "Photosynthesis is a process used by plants to convert light energy into chemical energy.",
        "question": "What is the purpose of photosynthesis?",
        "answer": "The purpose of photosynthesis is to convert light energy into chemical energy."
    },
    {
        "context": "The Amazon River is the largest river by discharge volume of water in the world.",
        "question": "Which river has the largest discharge volume?",
        "answer": "The Amazon River has the largest discharge volume."
    }
]

def generate_answer(question, context):
    # Build few-shot prompt
    prompt = "Answer the following questions based on the given context.\n\n"
    
    # Add few-shot examples
    for example in few_shot_examples:
        prompt += f"Context: {example['context']}\n"
        prompt += f"Question: {example['question']}\n"
        prompt += f"Answer: {example['answer']}\n\n"
    
    # Add current question and context
    prompt += f"Context: {context}\n"
    prompt += f"Question: {question}\n"
    prompt += "Answer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)

    with torch.no_grad():
        outputs = model.base_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=32,
            do_sample=False,
            temperature=0.7,  # Added for slightly more diverse outputs
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Extract only the new generated text (after the prompt)
    output_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return output_text

# Inference
predictions = []
references = []

for sample in tqdm(dataset):
    context = sample["context"]
    question = sample["question"]
    true_answers = sample["answers"]["text"]
    answer_start = sample["answers"]["answer_start"]

    pred_text = generate_answer(question, context)

    predictions.append({
        "id": sample["id"],
        "prediction_text": pred_text
    })
    references.append({
        "id": sample["id"],
        "answers": {
            "text": true_answers,
            "answer_start": answer_start
        }
    })

# Evaluation
results = metric.compute(predictions=predictions, references=references)
print("Exact Match:", round(results["exact_match"], 2))
print("F1 Score:", round(results["f1"], 2))