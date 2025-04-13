import torch
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import evaluate
import matplotlib.pyplot as plt

from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler
from torch.optim import AdamW
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType

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
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# soft prompt配置
peft_config = PromptTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="Answer the question based on the context:",
    num_virtual_tokens=20,
    tokenizer_name_or_path=model_name,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 加载SQuAD数据
train_dataset = load_dataset("squad", split="train")
val_dataset = load_dataset("squad", split="validation")
metric = evaluate.load("squad")

# 超参数
num_epochs = 5
batch_size = 8
lr = 5e-4

# optimizer+scheduler
optimizer = AdamW(model.parameters(), lr=lr)
num_training_steps = (len(train_dataset) // batch_size) * num_epochs
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# -------- 记录loss --------
all_batch_loss = []

# 训练
model.train()
for epoch in range(num_epochs):
    pbar = tqdm(range(0, len(train_dataset), batch_size))
    for idx in pbar:
        batch = train_dataset[idx: idx + batch_size]
        
        inputs = tokenizer(
            [f"question: {batch['question'][i]} context: {batch['context'][i]}" for i in range(len(batch['question']))],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(device)

        labels = tokenizer(
            [batch['answers'][i]['text'][0] if batch['answers'][i]['text'] else "" for i in range(len(batch['question']))],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=32,
        ).input_ids.to(device)

        labels[labels == tokenizer.pad_token_id] = -100  # 忽略pad

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        all_batch_loss.append(loss.item())
        pbar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 保存训练后的soft prompt
model.save_pretrained("./peft_soft_prompt")

# -------- 画loss曲线，每10个batch取均值 --------
smooth_loss = []
for i in range(0, len(all_batch_loss), 10):
    smooth_loss.append(np.mean(all_batch_loss[i:i+10]))

plt.figure(figsize=(10,8))
plt.plot(smooth_loss)
plt.xlabel("Every 10 Batches")
plt.ylabel("Loss")
plt.title("Smoothed Training Loss Curve")
plt.grid(True)
plt.savefig("training_loss_curve_epoch.png")
plt.show()

# 推理阶段
model.eval()

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

# 在validation set推理并评估
predictions = []
references = []

for sample in tqdm(val_dataset):
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
