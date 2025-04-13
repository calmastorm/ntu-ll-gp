import torch
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import evaluate
import matplotlib.pyplot as plt

from transformers import LlamaTokenizer, LlamaForCausalLM, get_scheduler
from torch.optim import AdamW
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType

# 固定随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型和分词器
model_name = "openlm-research/open_llama_3b_v2"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 重要！不然padding时报错
model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto")

# Soft prompt配置
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="Answer the question based on the context:",
    num_virtual_tokens=20,
    tokenizer_name_or_path=model_name,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 加载数据
train_dataset = load_dataset("squad", split="train")
val_dataset = load_dataset("squad", split="validation")
metric = evaluate.load("squad")

# 超参数
num_epochs = 2
batch_size = 8
lr = 5e-4

# optimizer和scheduler
optimizer = AdamW(model.parameters(), lr=lr)
num_training_steps = (len(train_dataset) // batch_size) * num_epochs
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# 记录loss
all_batch_loss = []

# train
model.train()
for epoch in range(num_epochs):
    pbar = tqdm(range(0, len(train_dataset), batch_size))
    for idx in pbar:
        batch = train_dataset[idx: idx + batch_size]

        prompts = []
        answers = []
        for i in range(len(batch["question"])):
            q = batch["question"][i]
            c = batch["context"][i]
            a = batch["answers"][i]["text"][0] if batch["answers"][i]["text"] else ""
            prompts.append(f"question: {q} context: {c}")
            answers.append(a)

        # 拼接prompt+answer作为input
        full_texts = [p + " answer: " + a for p, a in zip(prompts, answers)]

        tokenized = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        labels = input_ids.clone()

        # 只学习answer部分，mask掉prompt部分
        for i in range(len(prompts)):
            prompt_tokens = tokenizer(prompts[i] + " answer:", add_special_tokens=False).input_ids
            prompt_len = len(prompt_tokens)
            labels[i, :prompt_len] = -100  # 忽略prompt的loss

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        all_batch_loss.append(loss.item())
        pbar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 保存soft prompt
model.save_pretrained("./peft_soft_prompt_llama_fixed")

# loss curve 说实话没什么锤子用
smooth_loss = []
for i in range(0, len(all_batch_loss), 10):
    smooth_loss.append(np.mean(all_batch_loss[i:i+10]))

plt.figure(figsize=(10,8))
plt.plot(smooth_loss)
plt.xlabel("Every 10 Batches")
plt.ylabel("Loss")
plt.title("Training Loss Curve (Smoothed)")
plt.grid(True)
plt.savefig("training_loss_curve_llama_fixed.png")
plt.show()

# infer
model.eval()

def generate_answer(question, context):
    prompt = f"question: {question} context: {context} answer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=64,
            do_sample=False,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return decoded

# infer on v set
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

# eval
results = metric.compute(predictions=predictions, references=references)
print("Exact Match:", round(results["exact_match"], 2))
print("F1 Score:", round(results["f1"], 2))
