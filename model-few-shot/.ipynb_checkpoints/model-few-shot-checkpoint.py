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
    task_type="SEQ_2_SEQ_LM",
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="Answer the question based on the context:",
    num_virtual_tokens=20,
    tokenizer_name_or_path="t5-base",
)

# Add soft prompt to model
model = get_peft_model(model, peft_config)
model.eval()

# Load SQuAD validation set
dataset = load_dataset("squad", split="validation")
metric = evaluate.load("squad")

# Define few-shot examples
few_shot_examples = [
    {
        "context": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
        "question": "Who designed the Eiffel Tower?",
        "answer": "Gustave Eiffel"
    },
    {
        "context": "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America.",
        "question": "Where is the Amazon rainforest located?",
        "answer": "Amazon basin of South America"
    },
    {
        "context": "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials, generally built along an east-to-west line across the historical northern borders of China.",
        "question": "What materials were used to build the Great Wall of China?",
        "answer": "stone, brick, tamped earth, wood, and other materials"
    }
]

def generate_answer(question, context):
    # Construct few-shot prompt
    prompt = ""
    for example in few_shot_examples:
        prompt += f"question: {example['question']} context: {example['context']}\nanswer: {example['answer']}\n\n"
    
    # Add the current question and context
    prompt += f"question: {question} context: {context}\nanswer:"
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=64,
            do_sample=False,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return decoded

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