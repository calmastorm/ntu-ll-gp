import torch
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import evaluate

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType

# 固定随机
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型和分词器
model_name = "openlm-research/open_llama_3b_v2"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# 定义 soft prompt 配置
peft_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="Answer the question based on the context:",
    num_virtual_tokens=20,
    tokenizer_name_or_path=model_name,
)

# 加载 soft prompt，但推理时不使用
model = get_peft_model(model, peft_config)
model.eval()

# squad validation
dataset = load_dataset("squad", split="validation")
metric = evaluate.load("squad")

def generate_answer(question, context):
    prompt = f"question: {question}  context: {context}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model.base_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=32,    # causal LM用max_new_tokens
            do_sample=False,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return decoded

# 推理
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

# 评估
results = metric.compute(predictions=predictions, references=references)
print("Exact Match:", round(results["exact_match"], 2))
print("F1 Score:", round(results["f1"], 2))
