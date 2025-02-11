import os
import re

import pandas as pd
from siamese_sts.dataLoader.preprocess import preprocess
import logging
import torch
import numpy as np
from siamese_sts.dataLoader.dataset import SiameseDataset
from siamese_sts.dataLoader.vectorizer import CodeVectorizer
import transformers
from sklearn.model_selection import KFold
from torch.nn.utils.rnn import pad_sequence
logging.basicConfig(level=logging.INFO)

"""
For loading Linux data  and preprocessing
"""


class VDSData:
    DATA_FILE_PTH = r'D:\auto\siamese-nn-semantic-text-similarity\siamese_sts/siamese_dataset/'

    def __init__(
        self,
        dataset_name,
        #stopwords_path="siamese_sts/data_loader/stopwords-en.txt",
        model_name="lstm",
        max_sequence_len=1240,
        pretrained_model_name="codeBert",
        normalization_const=5.0,
        normalize_labels=False,
        embedding_size=64
    ):
        """
        Loads data into memory and create vocabulary from text field.
        """
        '''
        vulns = {'cveId_version' : [] }
        '''
        self.vulns = {}
        self.fixs = {}
        self.normalization_const = normalization_const
        self.normalize_labels = normalize_labels
        self.pretrained_model_name = pretrained_model_name
        self.model_name = model_name
        self.max_sequence_len = max_sequence_len
        self.dataset_name = dataset_name
        self.vectorizer = CodeVectorizer(embedding_size)
        ## load data file into memory
        self.load_data(dataset_name)


        ## create vocabulary over entire dataset before train/test split
       # self.create_vocab()

    def generate_pairs(self, cve_dict):
        similarity_pairs = []
        difference_pairs = []
        features = []
        labels = []

        for cve_id, files in cve_dict.items():
            vul_files = files['vul']
            fix_files = files['fix']

            # 生成差异对：一个易受攻击的函数和一个已修补的函数
            for vul_file in vul_files:

                for fix_file in fix_files:
                    similarity_pairs.append((vul_file, fix_file, "0"))
                    '''features.append((vul_file, fix_file))
                    labels.append("0")'''

            # 生成相似对：两个不同版本的易受攻击的函数
            if len(vul_files) > 1:
                vul_files_copy = vul_files.copy()  # 创建副本以避免在迭代中修改列表
                for i in range(len(vul_files_copy)):

                    for j in range(i + 1, len(vul_files_copy)):
                        difference_pairs.append((vul_files_copy[i], vul_files_copy[j],"1"))
                        '''
                        features.append((vul_files_copy[i], vul_files_copy[j]))
                        labels.append("1")'''

        return similarity_pairs, difference_pairs

    @staticmethod
    def extract_cve_info(file_name):
        # 正则表达式匹配 CVE ID 和类型
        match = re.match(r'(CVE-\d+-\d+)#(vul|fix)#(linux-\d+\.\d+\.\d+)', file_name)
        if match:
            return match.group(1), match.group(2), match.group(3)
        return None, None, None

    def read_file(self, file_path):
        file_path = VDSData.DATA_FILE_PTH + self.dataset_name + r'\\' + file_path
        codes = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                stripped = line.strip()
                if stripped:  # 跳过空行
                    # 使用正则表达式分割行为单词或符号
                    tokens = re.findall(r'\b[\w\.]+|[^\w\s]|\SCSI_SENSE_BUFFERSIZE', stripped)
                    codes.extend(tokens)
        return codes
    def traverse_directory(self):
        dataset_pth = VDSData.DATA_FILE_PTH+self.dataset_name
        '''
        cve_dict = {'cve_id':{'vul': [], 'fix': []}
        '''
        cve_dict = {}
        for root, dirs, files in os.walk(dataset_pth):
            for file in files:
                cve_id, type_, version = VDSData.extract_cve_info(file)
                if cve_id and type_:
                    # 如果字典中没有这个 CVE ID，就添加它
                    if cve_id not in cve_dict:
                        cve_dict[cve_id] = {'vul': [], 'fix': []}
                    # 根据类型添加文件名到字典
                    code = self.read_file(file)
                    if self.max_sequence_len<len(code):
                        self.max_sequence_len = len(code)
                    #code = preprocess(code)
                    cve_dict[cve_id][type_].append(code)
                    if type_ == 'vul':

                        self.vectorizer.add_gadget(code)
                        self.vulns[cve_id+'_'+version] = code
                    if type_ == 'fix':

                        self.vectorizer.add_gadget(code)
                        self.fixs[cve_id+'_'+version] = code
        return cve_dict
    def load_data(self, dataset_name):
        """
        Reads data set file from disk to memory using pandas
        """

        logging.info("loading and preprocessing data...")
        cve_dict = self.traverse_directory()   # 已经clean完
        self.vectorizer.train_model()
        # pairs = [(f1,f2,label)]
        self.simi_pairs, self.diff_pairs = self.generate_pairs(cve_dict)
        self.simi_pairs_tensor = self.data2tensors(self.simi_pairs)
        self.diff_pairs_tensor = self.data2tensors(self.diff_pairs)

        logging.info("reading and preprocessing data completed...")

    def cross_validation(self, num_folds=10):
        features = [pair for pair in self.simi_pairs_tensor + self.diff_pairs_tensor]  # 提取特征
        labels = [pair[2] for pair in self.simi_pairs + self.diff_pairs]  # 提取标签
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)  # 设置 random_state 以获得可重现的结果
        self.datasets = []

        for train_index, test_index in kf.split(features, labels):
            # 使用生成的索引划分训练集和测试集
            train_data = [features[i] for i in train_index]
            test_data = [features[i] for i in test_index]
            train_labels = [labels[i] for i in train_index]
            test_labels = [labels[i] for i in test_index]

            # 将特征和标签合并为元组 (feature, label)
            train_dataset = list(train_data)
            test_dataset = list(test_data)

            self.datasets.append((train_dataset, test_dataset))
        return self.datasets






    def data2tensors(self, pairs):
        """
        Converts raw data sequences into vectorized sequences as tensors
        """
        processed_pairs = []

        for pair in pairs:

            vectorized_fun_1 = torch.from_numpy(np.array(self.vectorizer.vectorize(pair[0])))
            vectorized_fun_2 = torch.from_numpy(np.array(self.vectorizer.vectorize(pair[1])))

            fun1_length = torch.LongTensor([len(pair[0])])
            fun2_length = torch.LongTensor([len(pair[1])])
            padded_sent1_tensor = self.pad_sequences(vectorized_fun_1)
            padded_sent2_tensor = self.pad_sequences(vectorized_fun_2)
            lable_tensor = torch.FloatTensor([float(pair[2])])

            processed_pairs.append((padded_sent1_tensor,padded_sent2_tensor,lable_tensor,fun1_length,fun2_length))

            '''sequence_1_length = len(vectorized_fun_1)
            sequence_2_length = len(vectorized_fun_2)

            if sequence_1_length <= 0 or sequence_2_length <= 0:
                continue

            padded_sent1_tensor, sents1_length = self.pad_sequences(vectorized_fun_1,
                                                                    torch.LongTensor(sequence_1_length))
            padded_sent2_tensor, sents2_length = self.pad_sequences(vectorized_fun_2,
                                                                    torch.LongTensor(sequence_2_length))

            targets = torch.FloatTensor([pair[2]])

            if self.normalize_labels:
                targets = targets / self.normalization_const

            processed_pairs.append(
                (padded_sent1_tensor, padded_sent2_tensor, targets, sents1_length, sents2_length, pair[0], pair[1]))'''

        return processed_pairs

    def pad_sequences(self, vector):


        padded_sequence = torch.zeros((self.max_sequence_len), dtype=torch.long)

        # 将原始向量复制到填充后的张量的相应位置
        padded_sequence[:len(vector):] = torch.tensor(vector, dtype=torch.long)

        # 返回填充后的张量
        return padded_sequence





def main():
    data = VDSData("new_linux_data")




if __name__ == "__main__":
    main()