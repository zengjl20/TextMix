#!/usr/bin/python
# -*- coding:utf-8 -*-
import collections
import logging
import os
import sys
import math
import random
import requests
import json

import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader, TensorDataset
from flask import Flask
from flask import request, Response, make_response


app = Flask("__name__")


@app.route('/sample', methods=['POST'])
def sample_result():
    try:
        if request.method == 'POST':
            req = request.get_data().decode("utf-8")
            req_dict = json.loads(req)
            print(req_dict)
            labeled_inds = req_dict["labeled_inds"]
            output_dir = req_dict["output_dir"]
            X_original = req_dict["X_original"]
            Y_original = req_dict["Y_original"]

            labeled_inds = [i for i in labeled_inds if Y_original[i] != -1]

            length = len(X_original)
            candidate_inds = [i for i in range(length) if i not in labeled_inds]
            sample_result = calculate_uncertainty(method='entropy',
                                                  best_model_path=output_dir,
                                                  annotations_per_it=5,
                                                  candidate_inds=candidate_inds,
                                                  X_original=X_original)

            sample_result = [int(i) for i in sample_result]
            print("sample_result: {}".format(sample_result))
            response_data = {"sample_inds": sample_result}
            print(response_data)

    except Exception as e:
        print(e)
        response_data = {
            "sample_inds": "Json error"
        }
    rst = make_response(json.dumps(response_data, indent=2, ensure_ascii=False))
    rst.headers['X-Frame-Options'] = 'DENY'
    return rst


def calculate_uncertainty(method, annotations_per_it,
                          best_model_path=None,
                          candidate_inds=None,
                          X_original=None):
    """
    输入：
    :参数 method: 基于不确定性筛选策略的名称. 可选项:
        - 'random'
        - 'least confidence'
        - 'entropy'
    :参数 annotations_per_it: 每一轮次筛选数量
    :参数 candidate_inds: 未标注样本的序号列表，例：[1, 3, 4, 6, ..., 2341]
    :参数 best_model_path: 最优模型参数地址
    :参数 X_original: 所有样本，size=[n]
    输出：
    :参数 sample_inds: 筛选的样本序号
    """

    # 参数定义
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    if best_model_path is not None:
        model = BertForSequenceClassification.from_pretrained(best_model_path)
    else:
        model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)

    # 数据初始化
    init_unlabeled_data = len(candidate_inds)

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-chinese",
        use_fast=True,
    )

    X_input = list(np.asarray(X_original)[candidate_inds])
    if method not in ['random']:
        encodings = tokenizer(X_input,  truncation=True, padding=True, max_length=64)
    else:
        encodings = tokenizer(X_original, truncation=True, padding=True, max_length=64)
    dataset = ASRDataset(encodings)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 模型预测
    model.to(device)
    model.eval()

    preds = None
    for batch in dataloader:
        with torch.no_grad():
            inputs = {"input_ids": batch['input_ids'].to(device), "attention_mask": batch['attention_mask'].to(device)}
            outputs = model(**inputs)
            logits = outputs[0]

        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

    logits = torch.tensor(preds)

    # 参数验证
    if method not in ['random']:
        if type(logits) is list and logits != []:
            assert init_unlabeled_data == logits[0].size(0), "logits {}, inital unlabaled data {}".format(
                logits[0].size(0), init_unlabeled_data)
        elif type(logits) != []:
            assert init_unlabeled_data == len(logits)

    # 计算信息度
    if method == 'least_conf':
        uncertainty_scores = least_confidence(logits)
    elif method == 'entropy':
        uncertainty_scores = entropy(logits)
    elif method == 'random':
        pass
    else:
        raise ValueError('Acquisition function {} not implemented yet check again!'.format(method))

    # 筛选样本
    if method == 'random':
        sampled_ind = np.random.choice(init_unlabeled_data, annotations_per_it, replace=False)
    else:
        sampled_ind = np.argpartition(uncertainty_scores, -annotations_per_it)[-annotations_per_it:]

    X_unlab = np.asarray(X_original, dtype='object')[candidate_inds]

    new_samples = np.asarray(X_unlab, dtype='object')[sampled_ind]

    assert len(new_samples) == annotations_per_it, 'len(new_samples)={}, annotatations_per_it={}'.format(len(new_samples), annotations_per_it)

    if method != 'random':
        sampled_ind = list(np.array(candidate_inds)[sampled_ind])
    else:
        sampled_ind = list(sampled_ind)
    return sampled_ind


def least_confidence(logits):
    if type(logits) is list:
        logits_B_K_C = torch.stack(logits, 1)  # prob
    else:
        logits_B_K_C = logits.unsqueeze(1)  # det
    least_conf_ = variation_ratios(logits_B_K_C)
    return least_conf_.cpu().numpy()


def entropy(logits):
    if type(logits) is list:
        logits_B_K_C = torch.stack(logits, 1)  # prob
    else:
        logits_B_K_C = logits.unsqueeze(1)  # det

    logits_ = logit_mean(logits_B_K_C, dim=1, keepdim=False)
    entropy_scores_ = -torch.sum((torch.exp(logits_) * logits_).double(), dim=-1, keepdim=False)
    return entropy_scores_.cpu().numpy()


def logit_mean(logits, dim: int, keepdim: bool = False):
    r"""Computes $\log \left ( \frac{1}{n} \sum_i p_i \right ) =
    \log \left ( \frac{1}{n} \sum_i e^{\log p_i} \right )$.

    We pass in logits.
    """
    return torch.logsumexp(logits, dim=dim, keepdim=keepdim) - math.log(logits.shape[dim])


class ASRDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings

    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


if __name__ == '__main__':
    # 此部分开始为模型输入的demo
    # 具体来讲，只需要输入所有样本的文本list，已标注样本的序号list两项
    # 此脚本目前还需要输入未标注样本的id列表，后续可优化
    X_original = []
    with open('./afqmc_1', 'r') as f:
        for line in f.readlines():
            X_original.append(line)

    length = len(X_original)
    candidate_inds = list(np.random.choice(length, int(length * 2 / 3), replace=False))
    labeled_inds = [i for i in range(length) if i not in candidate_inds]

    print(calculate_uncertainty(method='entropy', annotations_per_it=50,
                                best_model_path='./models/',
                                candidate_inds=candidate_inds,
                                X_original=X_original))