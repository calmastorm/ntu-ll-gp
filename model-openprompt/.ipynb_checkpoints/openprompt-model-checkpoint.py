import torch
import numpy as np
import random
from tqdm import tqdm
import evaluate
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from openprompt.prompts import SoftTemplate
from openprompt import PromptForGeneration
from openprompt.data_utils import InputExample

# 固定随机
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model+tokenizer
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
plm = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
plm.eval()

# squad validation
dataset = load_dataset("squad", split="validation")
metric = evaluate.load("squad")

# soft template
template = SoftTemplate(
    model=plm,
    tokenizer=tokenizer,
    text='{"soft"} {"soft"} {"soft"} question: {"placeholder":"text_a"} context: {"placeholder":"text_b"} answer: {"mask"}',
    num_tokens=3,
    initialize_from_vocab=True
).to(device)

# prompt model
prompt_model = PromptForGeneration(
    plm=plm,
    template=template,
    freeze_plm=True
).to(device)
prompt_model.eval()

def generate_answer(question, context):
    example = InputExample(
        guid="",
        text_a=question,
        text_b=context
    )
    batch = template.wrap_one_example(example)
    batch = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = prompt_model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=32,
            do_sample=False,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return decoded

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

results = metric.compute(predictions=predictions, references=references)
print("Exact Match:", round(results["exact_match"], 2))
print("F1 Score:", round(results["f1"], 2))
