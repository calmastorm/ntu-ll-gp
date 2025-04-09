import torch
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import evaluate

from transformers import T5Tokenizer, T5ForConditionalGeneration
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
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# 定义 soft prompt 配置
peft_config = PromptTuningConfig(
    task_type="SEQ_2_SEQ_LM",
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="Your initialization text here",
    num_virtual_tokens=20,
    tokenizer_name_or_path="t5-base",
)

# 把 soft prompt 加到模型上
model = get_peft_model(model, peft_config)
model.eval()

# 加载SQuAD validation集
dataset = load_dataset("squad", split="validation")
metric = evaluate.load("squad")

def generate_answer(question, context):
    prompt = f"question: {question} context: {context}"
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
