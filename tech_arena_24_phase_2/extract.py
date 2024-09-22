import os
import shutil
import re
from datetime import datetime

from score import score_main

# 假设文件都在当前目录下
input_dir = './output'
current_dir = './'

# 创建一个以当前时间命名的目录
current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
output_dir = os.path.join(current_dir, current_time)

output_dir_score = os.path.join(output_dir, 'score')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(output_dir_score):
    os.makedirs(output_dir_score)

# 定义用于提取 seed 和分数的正则表达式
file_pattern = re.compile(r'(\d+)_([0-9.]+e\+\d+)\.json')

# 字典存储每个 seed 对应的最大分数文件
max_files = {}

# 遍历文件
for file_name in os.listdir(input_dir):
    match = file_pattern.match(file_name)
    if match:
        seed, score = match.groups()
        score = float(score)

        # 如果当前 seed 不存在，或者找到更大的分数文件
        if seed not in max_files or score > max_files[seed][1]:
            max_files[seed] = (file_name, score)

# 复制最大分数文件到目标目录，去掉分数后缀，只保留seed作为文件名
for seed, (file_name, _) in max_files.items():
    # 复制文件并重命名为 seed.json
    src_file = os.path.join(input_dir, file_name)
    dst_file = os.path.join(output_dir, f"{seed}.json")  # 只保留 seed 作为文件名
    shutil.copy2(src_file, dst_file)
    shutil.copy2(src_file, output_dir_score)  # 复制到 score 目录

print(f"最大分数文件已复制到 {output_dir}，文件名只保留 seed。")

# score_main(output_dir)
