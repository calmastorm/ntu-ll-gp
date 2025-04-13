import torch
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import evaluate
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType

# Fixed random seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and tokenizer
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Soft prompt configuration
peft_config = PromptTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="Answer the question based on the context. Examples:",
    num_virtual_tokens=20,
    tokenizer_name_or_path=model_name,
)

# Add soft prompt to model
model = get_peft_model(model, peft_config)
model.eval()

# Load SQuAD validation set
dataset = load_dataset("squad", split="validation")
metric = evaluate.load("squad")

# Optimized few-shot examples
few_shot_examples = [
    {
        "context": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        "question": "Who designed the Eiffel Tower?",
        "answer": "Gustave Eiffel"
    },
    {
        "context": "The Amazon rainforest covers most of the Amazon basin of South America.",
        "question": "Where is the Amazon rainforest located?",
        "answer": "Amazon basin of South America"
    },
    {
        "context": "The Great Wall of China is made of stone, brick, and other materials.",
        "question": "What materials were used to build the Great Wall?",
        "answer": "stone, brick, and other materials"
    }
]

def generate_answer(question, context):
    """Generate answer using PEFT prompt tuning with few-shot examples"""
    # Build optimized prompt
    prompt_lines = []
    
    # Add few-shot examples
    for example in few_shot_examples:
        prompt_lines.append(f"Question: {example['question']}")
        prompt_lines.append(f"Context: {example['context']}")
        prompt_lines.append(f"Answer: {example['answer']}\n")
    
    # Add current question with PEFT soft prompt
    prompt_lines.append(f"Now answer:")
    prompt_lines.append(f"Question: {question}")
    prompt_lines.append(f"Context: {context}")
    prompt_lines.append("Answer:")
    
    prompt = "\n".join(prompt_lines)
    
    # Tokenize with attention to length
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    ).to(device)

    # Generate with tuned parameters
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=64,
            do_sample=False
        )
    
    # Clean and return answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Fallback mechanism if answer is invalid
    if answer.lower() in ["true", "false", "yes", "no"] or len(answer) < 2:
        simple_prompt = f"question: {question} context: {context}"
        inputs = tokenizer(simple_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=64
            )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    return answer

# Evaluation loop
predictions = []
references = []

for sample in tqdm(dataset, desc="Evaluating"):
    pred_text = generate_answer(sample["question"], sample["context"])
    
    predictions.append({
        "id": sample["id"],
        "prediction_text": pred_text
    })
    references.append({
        "id": sample["id"],
        "answers": sample["answers"]
    })

# Calculate and display results
results = metric.compute(predictions=predictions, references=references)
print("\nFinal Evaluation Results:")
print(f"Exact Match: {results['exact_match']:.2f}")
print(f"F1 Score: {results['f1']:.2f}")
print(f"Number of samples: {len(dataset)}")

# Show some examples
print("\nSample Predictions:")
for i in random.sample(range(5), 3):
    print(f"\nQuestion: {dataset[i]['question']}")
    print(f"Context: {dataset[i]['context'][:100]}...")
    print(f"True Answer: {dataset[i]['answers']['text'][0]}")
    print(f"Predicted Answer: {predictions[i]['prediction_text']}")