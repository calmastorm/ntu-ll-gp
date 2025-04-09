import os

openprompt_path = "/home/msai/hu0023an/.conda/envs/msai_env/lib/python3.12/site-packages/openprompt/pipeline_base.py"

if not os.path.exists(openprompt_path):
    print("错误：找不到 OpenPrompt 的 pipeline_base.py 文件，请检查路径")
    exit(1)

with open(openprompt_path, "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "from transformers.generation_utils import GenerationMixin" in line:
        new_line = line.replace(
            "from transformers.generation_utils import GenerationMixin",
            "from transformers.generation.utils import GenerationMixin"
        )
        new_lines.append(new_line)
    elif "from transformers import AdamW, get_linear_schedule_with_warmup" in line:
        new_line = line.replace(
            "from transformers import AdamW, get_linear_schedule_with_warmup",
            "from transformers.optimization import AdamW, get_linear_schedule_with_warmup"
        )
        new_lines.append(new_line)
    else:
        new_lines.append(line)

with open(openprompt_path, "w") as f:
    f.writelines(new_lines)

print("✅ OpenPrompt所有兼容补丁应用成功！")
