#!/usr/bin/python
# -*- coding:utf-8 -*-
import collections
import logging
import os
import sys
import math
import csv
import random
import requests
import json

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoConfig, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, TensorDataset
from flask import Flask
from flask import request, Response, make_response


app = Flask("__name__")


@app.route('/training', methods=['POST'])
def train_result():
    try:
        if request.method == 'POST':
            req = request.get_data().decode("utf-8")
            req_dict = json.loads(req)
            print(req_dict)
            labeled_inds = req_dict["labeled_inds"]
            output_dir = req_dict["output_dir"]
            X_original = req_dict["X_original"]
            Y_original = req_dict["Y_original"]

            labeled_inds = [i for i in range(len(Y_original)) if Y_original[i] != -1]

            acc = train_save_model(output_dir=output_dir,
                                   labeled_inds=labeled_inds,
                                   X_original=X_original,
                                   Y_original=Y_original)
            print("acc: {}".format(acc))
            response_data = {"acc": int(acc)}
            print(response_data)

    except Exception as e:
        print(e)
        response_data = {
            "acc": "Json error"
        }
    rst = make_response(json.dumps(response_data, indent=2, ensure_ascii=False))
    rst.headers['X-Frame-Options'] = 'DENY'
    return rst


def train_save_model(output_dir=None,
                     labeled_inds=None,
                     X_original=None,
                     Y_original=None):
    """
    输入：
    :参数 labeled_inds: 已标注样本的序号列表，例：[0, 2, 5, 7, ..., 2342]
    :参数 output_dir: 模型参数地址
    :参数 X_original: 所有样本的文本，size=[n]
    :参数 Y_original: 所有样本的标签，size=[n]。未标注样本的标签可先用其他label代替，例如二分类标签为0、1。未标注样本可打上标签-1
    输出：
    :output_dir目录下会保存一个最优模型。
    """

    # 参数定义
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)

    # 数据初始化
    init_labeled_data = len(labeled_inds)  # before selecting data for annotation

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-chinese",
        use_fast=True,
    )

    # 训练集验证集划分
    X_input = list(np.asarray(X_original)[labeled_inds])
    Y_input = list(np.asarray(Y_original)[labeled_inds])
    x_train, x_test, train_label, test_label = train_test_split(X_input[:],
                                                                Y_input[:], test_size=0.2, stratify=Y_input[:])
    # 加载dataloader
    train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
    test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)

    train_dataset = ASRTrainDataset(train_encodings, train_label)
    test_dataset = ASRTrainDataset(test_encodings, test_label)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 模型训练，评估
    model.to(device)

    optim = AdamW(model.parameters(), lr=2e-5)

    best_eval_accuracy = 0
    for epoch in range(5):
        model.train()
        total_train_loss = 0
        iter_num = 0
        total_iter = len(train_dataloader)

        for batch in train_dataloader:
            optim.zero_grad()
            inputs = {"input_ids": batch['input_ids'].to(device),
                      "attention_mask": batch['attention_mask'].to(device),
                      "labels": batch["labels"].to(device)}
            outputs = model(**inputs)
            loss = outputs[0]
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 参数更新
            optim.step()

            iter_num += 1
            if iter_num % 100 == 0:
                print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (
                    epoch, iter_num, loss.item(), iter_num / total_iter * 100))

        print("Epoch: %d, Average training loss: %.4f" % (epoch, total_train_loss / len(train_dataloader)))

        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        for batch in test_dataloader:
            with torch.no_grad():
                inputs = {"input_ids": batch['input_ids'].to(device),
                          "attention_mask": batch['attention_mask'].to(device),
                          "labels": batch["labels"].to(device)}
                outputs = model(**inputs)
                labels = batch['labels'].to(device)
            loss = outputs[0]
            logits = outputs[1]

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
        if avg_val_accuracy > best_eval_accuracy:
            model.save_pretrained(output_dir)
            print("Save the best model to output_dir")
        best_eval_accuracy = max(best_eval_accuracy, avg_val_accuracy)
        print("Accuracy: %.4f" % avg_val_accuracy)
        print("Average testing loss: %.4f" % (total_eval_loss / len(test_dataloader)))

    return best_eval_accuracy


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


class ASRTrainDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    # 此部分为模型输入的demo
    # 具体来讲，需要输入所有样本的文本list，所有样本的标签list。格式见函数注释部分。
    # 此脚本目前还需输入已标注样本的id列表，后续可优化。

    Y_original = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 0, 0]
    label_inds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

    X_original = ['喂你好。喂你好。喂美女。哎。你的车怎么停到人门口人家司机不好上下货啊。哦，好，那我那个下去叫他一下。你中午都，中，你中午都被罚了，你那边车位你们那边住的地方，那么多车位给的题目大门口。不好意思不好意思我，我那个现在到下面去，他不好意思啊。', '导这，何支。喂，你好。喂。啊您好你，来一下，我能货让进门。嗯好，吃是吗。六，S，S九七柜东妈仪下车我们要卸会。好好好好马上再去，行。好好好嗯。好，嗯先生。', '你个。机。喂你好哪里。啊你那个，电，帮我离一下。哎这个就是八二幺的这个车麻烦你帮我一一下。我现在没个选职我的进讲。嗯。不是靠你不是不是已经全靠，他很靠强了吗。嗯。一。好好，嗯。哦，啊。', '你个。机。喂你好哪里。啊你那个，电，帮我离一下。哎这个就是八二幺的这个车麻烦你帮我一一下。我现在没个选职我的进讲。嗯。不是靠你不是不是已经全靠，他很靠强了吗。嗯。一。好好，嗯。哦，啊。', '喂你好。是免C八零TV二这宝马车主是吧。你这个车灯和车门都没锁啊。五车灯跟车门没锁是吧。对车灯也没关，我们是达理酒店是，这个管部的。哦哦，好好好好。行麻烦你下来关一下啊。嗯嗯嗯。', '没接了，我听人赶了三。哎。喂。喂，喂。嗯，嗯。你那个来来那个车挪一下吧。这个学校门口这个车，这个是一个通道嘛。撤。还哈。好好行好。哦，叔是明天。好好。好好好。', '喂你好。嗯你这个车。就是过，八点之前要一早不要开，天车这边，被打路的。哦哦，知道知道。', '喂。在上给我这个家的人。嗯，嗯需要。嗯需要。我这个讲，听想业听到听到几大的。想爱。放电话呀。没想。我送备呢。我给赢。哦，看他三人人呀。哎呀，什地话。我上九点也没上病啊。九零吧。哦哦。我花M的。喂你换没用了还不下开。你找席手机这等单啊。你打哦。喂。哦，那我当你回K那就边我，定家的。嗯，说。哦，让我把百案发一给你再想给要匙。说，话，我恰办话，的。哦，啊。你好。', '没。啊这个，免息，S，三幺七的，三幺七。是的的是他是我是你的。啊，事。对对对。你身份交完通过不能停啊能叫他机车也要郑桥区拉垃圾进不了。啊行行行好嘞好嘞好嘞。我赶紧来一下。真牛逼呀你听了。', '这个我擦了。喂，喂六。喂。日西，八个比三，一爷掐走了。哪快点能说话点说快点能算。那你些家都，刷工电课啊不会带，让来得着是，打行K再回回回QK。好，我在东你们，的上啊。白K个号。给打工了，一百能。短短信下反馈下号。哦好我我这天的，我的，个。反正就说。哦N。哦你啥抓工定可啊哈。啊好好好行嘞，嗯。', '喂。在前退了。这电话上没开十万。谁的可以的算了。啊我，没亏我再再说的，啊。啊现在在刷看我看下视频说这很，两，听，听了下了个，家听个我为他肚蛋干。哦喂好。', '喂你好。嗯，这部，白色的宝马，BC，五A三六S自里的车呢。对你，你说。叫他把催挪一下啊我这边就交紧了。好，行。好。等现在现在下安走，让我们站上推下去了好。好。好拜拜。', '喂。你这个车下雨下我们放掉来了。我们电视施工呢车要你下。我是他是一下事请盖的。', '嘿，你好。哎，你好我是嫌外社区的你这个车来这边要一调来这边要花车位你这个车站我们挡住了不行啊。哦我我看一下看是谁开，他到那边哈。三八LM五，车的赶快过来，挪一下好。嗯好好，我通知他们一下哈。', '你别忘。喂你好。喂你好这个免息。嗯。这个。井，C，一三六。三六DZ七是你的车是吧。哎对对对对报，是不是。喂你好老驾把车一下你这个车刚好在我们，我们今天这边做管的。管到清，清清，清洗啊，刚诉你这车这个，把警盖给压到了。好好。好，好好好好的好的好的，嗯。嗯行好嘞好嘞谢谢啊哎。嗯，嗯。', '他，他房的拿话也不问我问我问。个电话去把老看问。喂。快来打航都，这款，喂。嗯。不。你是不车不能停在这边向王通道你呢下来一下哈。好了。不是干。', '喂。喂，嗯。哎你好哎麻烦你这个车来挪一下。我们这边在施工对。在哪里呀。人民医院这边啊你不是三NB借联是你的吗。在人民医院。嗯，嗯。车在哪里你自己的车都不知道。哦，被有朋友开去了。那你你来来来了搂了，换了好多天哪都都他们都我功能都不知道我今天在看到他，才。我才那个枣。你们车你们手手机号也不留一个。好好。好好好好好听停了好几天了是吗。哦你来开吧我现在这边。叫你你你给来赶快来开着吧好我这边在施工，哦。好，嗯。嗯嗯嗯。', '这个，八点三六六个吗。要钱说的然货北还没打就搞的，按。爱了两茶麻烦机器叫的趴掉高了我。好好好好。嗯嗯嗯好。', '喂你好。我八。喂好。喂你好。喂你好啊米的六二二四这不课是你的吗。呃是我的是我的马上金融哈。赶时下来看看下来看这边要开盘了，业主都在去了。好拜拜，好好拜拜。', '喂你好。我八。喂好。喂你好。喂你好啊米的六二二四这不课是你的吗。呃是我的是我的马上金融哈。赶时下来看看下来看这边要开盘了，业主都在去了。好拜拜，好好拜拜。', '天送外递款喽。哦那，这个，是，个，五六S的这个六吗。四E五六SC呀。那的听老通道了。等得了得要钱考通到行。是啊。我才我在老回个在老婆行快车。', '喂。免时在，TQK可以的吗。啊行。那点多话挺会了靠，起来跟过来给我，回电呀。哦好好好。', '这个我擦了。喂，喂六。喂。日西，八个比三，一爷掐走了。哪快点能说话点说快点能算。那你些家都，刷工电课啊不会带，让来得着是，打行K再回回回QK。好，我在东你们，的上啊。白K个号。给打工了，一百能。短短信下反馈下号。哦好我我这天的，我的，个。反正就说。哦N。哦你啥抓工定可啊哈。啊好好好行嘞，嗯。', '喂。在前退了。这电话上没开十万。谁的可以的算了。啊我，没亏我再再说的，啊。啊现在在刷看我看下视频说这很，两，听，听了下了个，家听个我为他肚蛋干。哦喂好。', '喂你好。嗯，这部，白色的宝马，BC，五A三六S自里的车呢。对你，你说。叫他把催挪一下啊我这边就交紧了。好，行。好。等现在现在下安走，让我们站上推下去了好。好。好拜拜。', '喂。你这个车下雨下我们放掉来了。我们电视施工呢车要你下。我是他是一下事请盖的。', '嘿，你好。哎，你好我是嫌外社区的你这个车来这边要一调来这边要花车位你这个车站我们挡住了不行啊。哦我我看一下看是谁开，他到那边哈。三八LM五，车的赶快过来，挪一下好。嗯好好，我通知他们一下哈。', '你别忘。喂你好。喂你好这个免息。嗯。这个。井，C，一三六。三六DZ七是你的车是吧。哎对对对对报，是不是。喂你好老驾把车一下你这个车刚好在我们，我们今天这边做管的。管到清，清清，清洗啊，刚诉你这车这个，把警盖给压到了。好好。好，好好好好的好的好的，嗯。嗯行好嘞好嘞谢谢啊哎。嗯，嗯。']

    labeled_inds = [i for i in range(len(Y_original)) if Y_original[i] != -1]

    print(train_save_model(output_dir='./models/',
                            labeled_inds=labeled_inds,
                            X_original=X_original,
                            Y_original=Y_original))