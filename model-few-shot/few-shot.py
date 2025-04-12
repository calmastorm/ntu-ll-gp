import torch
from tqdm import tqdm
from datasets import load_dataset
import evaluate
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Set up environment
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
model.eval()

# Load dataset and metrics
dataset = load_dataset("squad", split="validation")
metric = evaluate.load("squad")

def generate_answer(question, context):
    """Optimized few-shot prompt for T5 with fallback to zero-shot"""
    # Try few-shot first
    few_shot_prompt = f"""
    Answer the question based on the context below. Examples:
    
    Question: What is the capital of France?
    Context: France is a European country.
    Answer: Paris
    
    Question: When was NASA founded?
    Context: The National Aeronautics and Space Administration was established in 1958.
    Answer: 1958
    
    Now answer:
    Question: {question}
    Context: {context}
    Answer:"""
    
    inputs = tokenizer(
        few_shot_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=32,
            num_beams=3,
            early_stopping=True
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Fallback to zero-shot if answer is nonsensical
    if answer.lower() in ["false", "true", "yes", "no"] or len(answer) < 2:
        zero_shot_prompt = f"question: {question} context: {context}"
        inputs = tokenizer(zero_shot_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=32
            )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    return answer

# Evaluation loop
predictions = []
references = []

for sample in tqdm(dataset, desc="Processing"):
    pred_text = generate_answer(sample["question"], sample["context"])
    
    predictions.append({
        "id": sample["id"],
        "prediction_text": pred_text
    })
    references.append({
        "id": sample["id"],
        "answers": sample["answers"]
    })

# Calculate metrics
results = metric.compute(predictions=predictions, references=references)
print(f"\nResults:")
print(f"Exact Match: {results['exact_match']:.2f}")
print(f"F1 Score: {results['f1']:.2f}")

# Show samples
print("\nExample Predictions:")
for i in range(3):
    print(f"\nQuestion: {dataset[i]['question']}")
    print(f"Context: {dataset[i]['context'][:100]}...")
    print(f"True Answer: {dataset[i]['answers']['text'][0]}")
    print(f"Predicted Answer: {predictions[i]['prediction_text']}")