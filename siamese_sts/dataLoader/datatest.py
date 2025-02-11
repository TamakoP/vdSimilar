import os
import re

def extract_cve_info(file_name):
    # 正则表达式匹配 CVE ID 和类型
    match = re.match(r'(CVE-\d+-\d+)#(vul|fix)', file_name)
    if match:
        return match.group(1), match.group(2)
    return None, None

def traverse_directory(directory):
    cve_dict = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            cve_id, type_ = extract_cve_info(file)
            if cve_id and type_:
                # 如果字典中没有这个 CVE ID，就添加它
                if cve_id not in cve_dict:
                    cve_dict[cve_id] = {'vul': [], 'fix': []}
                # 根据类型添加文件名到字典
                cve_dict[cve_id][type_].append(file)
    return cve_dict

def generate_pairs(cve_dict):
    similarity_pairs = []
    difference_pairs = []

    for cve_id, files in cve_dict.items():
        vul_files = files['vul']
        fix_files = files['fix']

        # 生成差异对：一个易受攻击的函数和一个已修补的函数
        for vul_file in vul_files:
            for fix_file in fix_files:
                difference_pairs.append((vul_file, fix_file))

        # 生成相似对：两个不同版本的易受攻击的函数
        if len(vul_files) > 1:
            vul_files_copy = vul_files.copy()  # 创建副本以避免在迭代中修改列表
            for i in range(len(vul_files_copy)):
                for j in range(i + 1, len(vul_files_copy)):
                    similarity_pairs.append((vul_files_copy[i], vul_files_copy[j]))

    return similarity_pairs, difference_pairs

def read_file(file_path):
    file_prefix= r'D:\auto\siamese-nn-semantic-text-similarity\siamese_sts\siamese_dataset\new_linux_data'
    file_path = file_prefix+r'\\'+file_path
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess_code(code):
    # 这里可以添加去除注释、格式化代码等预处理步骤
    return code

def encode_code_for_codebert(code):
    inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True)
    return inputs

def get_code_embeddings(inputs):
    outputs = model(**inputs)
    return outputs.last_hidden_state

def process_file(file_path):
    code = read_file(file_path)
    #code = preprocess_code(code)
    inputs = encode_code_for_codebert(code)
    embeddings = get_code_embeddings(inputs)
    return embeddings
# 使用示例
LINUX_PTH = r'D:\auto\siamese-nn-semantic-text-similarity\siamese_sts\siamese_dataset\new_linux_data'
cve_linux_files = traverse_directory(LINUX_PTH)
similarity_pairs, difference_pairs = generate_pairs(cve_linux_files)

from transformers import AutoTokenizer, AutoModel
import torch

# 初始化 CodeBERT 的 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(r"D:\auto\siamese-nn-semantic-text-similarity\siamese_sts\codeBert")
model = AutoModel.from_pretrained(r"D:\auto\siamese-nn-semantic-text-similarity\siamese_sts\codeBert")

for pair in similarity_pairs:
    a=process_file(pair[0])
    b=process_file(pair[1])