import torch
import numpy as np
import random
from tqdm import tqdm
import evaluate
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
model.eval()

# Load dataset and metrics
dataset = load_dataset("squad", split="validation")
metric = evaluate.load("squad")

def generate_answer(question, context):
    # Carefully selected few-shot examples covering different question types
    few_shot_examples = [
        {
            "question": "What is the capital of France?",
            "context": "France is a country in Europe. Its capital is Paris.",
            "answer": "Paris"
        },
        {
            "question": "When was the Declaration of Independence signed?",
            "context": "The Declaration of Independence was signed on August 2, 1776.",
            "answer": "August 2, 1776"
        },
        {
            "question": "Why do leaves change color in autumn?",
            "context": "In autumn, leaves change color due to reduced chlorophyll production as daylight decreases.",
            "answer": "because chlorophyll production decreases with less daylight"
        }
    ]
    
    # Build optimized few-shot prompt
    prompt_parts = []
    for example in few_shot_examples:
        prompt_parts.append(f"question: {example['question']} context: {example['context']}\nanswer: {example['answer']}")
    
    # Add current question with clear separation
    prompt_parts.append(f"question: {question} context: {context}\nanswer:")
    prompt = "\n\n".join(prompt_parts)
    
    # Tokenize with attention to length
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    ).to(device)
    
    # Generate answer
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=32,
            num_beams=4,  # Slightly better than greedy for QA
            early_stopping=True,
            do_sample=False
        )
    
    # Clean and return answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer.split("\n")[0]  # In case model generates multiple lines

# Evaluation loop
predictions = []
references = []

for sample in tqdm(dataset, desc="Evaluating"):
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

# Calculate and print metrics
results = metric.compute(predictions=predictions, references=references)
print("\nFinal Results:")
print(f"Exact Match: {round(results['exact_match'], 2)}")
print(f"F1 Score: {round(results['f1'], 2)}")
print(f"Total Samples: {len(dataset)}")

# Optional: Print some examples
print("\nSample Predictions:")
for i in random.sample(range(len(dataset)), 3):
    print(f"\nQuestion: {dataset[i]['question']}")
    print(f"Context: {dataset[i]['context'][:150]}...")
    print(f"True Answer: {dataset[i]['answers']['text'][0]}")
    print(f"Predicted Answer: {predictions[i]['prediction_text']}")