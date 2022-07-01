#!/usr/bin/python
# -*- coding:utf-8 -*-
import json
import os
import time
import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import datetime,time
import datetime
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel
from torch import optim
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
from torch.utils.data import random_split
from sklearn import metrics
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import sys
from log import Logger
from flask import Flask
from flask import request, Response, make_response
import requests


id = 1 #sys.argv[1]
# with open('/opt/input/context_{}.txt'.format(id), 'r') as f:
#     data = f.readlines()
#     data = ''.join(data).split('$$')
os.environ['CUDA_VISIBLE_DEVICE']='1,2,3,4'
model_name = './bert-base-chinese'
cache_dir = 'E:/2 - PythonProject/8 - nlp/8 - bert_trans/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)
Debug = False
epoch = 10
Pretrainpath = '/home/test/'


app = Flask("__name__")

log_path = os.path.join(os.path.dirname(__file__), 'logs{}.log'.format(datetime.datetime.now().strftime("%Y-%m-%d")))
log = Logger(log_path).logger
@app.route('/multi_label', methods=['POST'])
def multi_label_result():
    req_dict = {}
    try:
        if request.method == 'POST':
            req = request.get_data().decode("utf-8")
            req_dict = json.loads(req)
            log.debug(req_dict)
            print(req_dict)
            id = req_dict["id"]
            print(id.isalnum())
            if len(id) > 32:
                response_data = {
                   "result": "id error"
                }
            else:
                if id.isalnum() is False:
                   response_data = {
                      "result": "id error"
                   }
                else:
                   context = req_dict["context"]

                   pred = test(context)
                   pred_txt = pred_result(pred)

                   print("'id:' ", id, ", result: ", pred_txt)

                   response_data = {
                      "id": "",
                      "context": "",
                      "result": ""
                   }

                   response_data["id"] = id
                   response_data["context"] = context
                   response_data["result"] = pred_txt
                   print(response_data)

    except Exception as e:
        print(e)
        log.error(e)
        log.err
        response_data = {
            "result": "Json error"
        }
    rst = make_response(json.dumps(response_data, indent=2,ensure_ascii=False))
    rst.headers['X-Frame-Options'] = 'DENY' 
    #return json.dumps(response_data, indent=2, ensure_ascii=False)
    return rst

def read_test_file(filename):
    df = pd.read_excel(filename, sheet_name='总表', keep_default_na=False)
    train_data = df['录转内容'].values
    return train_data

def data_process(train_data, train_label):
    # label: 是=1， 否=0
    dataset = []

    for i in range(len(train_data)):
        ds = {
            'input_token_ids': '',
            'token_ids_labels': ''
        }
        ds['input_token_ids'] = train_data[i]
        ds['token_ids_labels'] = (1 if train_label[i] == '是' else 0)
        dataset.append(ds)

    return dataset

def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def predict(logits):
    # res = torch.argmax(logits, 1)
    accuracy_th = 0.5
    res = logits > accuracy_th
    return res

def string_format(str):
    str = str.replace("\"",'').replace("n1:",'').replace("n0:",'').replace('\n','。').replace('[','').replace(']','').replace(' ','')
    if len(str) > 510:
        str = str[:510]
    elif len(str) < 510:
        for i in range(510-len(str)):
            str += '[PAD]'
    return str


def diacls_dataset(datas, labels=None):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenized_text = [tokenizer.tokenize(string_format(i)) for i in datas]
    input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
    for i in range(len(input_ids)):
        j = input_ids[i]
        if len(j) < 510:
            input_ids[i].extend([0]*(510-len(j)))
        input_ids[i].insert(0, 101)
        input_ids[i].extend([102])

    # input_labels = [[0,1] if i == '是' else [1,0] for i in train_label ]
    # input_labels = [[0,1] if i == '是否有效' else [1,0] for i in train_label ]
    #[预约上门，客户报障，撤销服务，其他]
    # input_labels = [[0, 1] if i == '预约上门' or i == '客户报障' or i == '撤销服务' or i == '业务咨询' or  i == '告知服务' else [1, 0] for i in train_label]
    input_labels = []
    if labels is not None:
        for label in labels:
            labs = [0,0,0]
            for i in label.split('+'):
                if i == '预约上门':
                    labs = np.sum([[1,0,0],labs], axis=0).tolist()
                elif i == '客户报障':
                    labs = np.sum([[0,1,0],labs], axis=0).tolist()
                elif i == '撤销服务' or i == '其他':
                    labs = np.sum([[0,0,1],labs],axis=0).tolist()
            input_labels.append(labs)

        train_set = TensorDataset(torch.LongTensor(input_ids),
                                  torch.FloatTensor(input_labels))
    else:
        train_set = TensorDataset(torch.LongTensor(input_ids))

    return train_set


class dia_cls(nn.Module):
    def __init__(self):
        super(dia_cls, self).__init__()
        self.model = BertModel.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.dropout = nn.Dropout(0.1).to(device)
        self.l1 = nn.Linear(768, 3).to(device)

    def forward(self, x, attention_mask=None):
        outputs = self.model(x, attention_mask=attention_mask)
        x = outputs[1]
        x = x.view(-1, 768)
        x = self.dropout(x)
        x = self.l1(x)
        return x


def test(ori_data):
    # print("start....")

    ori_set = diacls_dataset(ori_data)
    sigmoid = nn.Sigmoid().to(device)

    # path = '/home/module/module/ep6loss0.0407547764480114'
    path = './ep6loss0.0407547764480114'
    dia_cls_model = torch.load(path).to(device)
    test_loader = DataLoader(dataset=ori_set, batch_size=50, shuffle=False)

    with torch.no_grad():
        dia_cls_model.eval()
        pred_all, true_all, txt_all = [], [], []
        for batch_idx, data in enumerate(test_loader):
            test_input_ids = data[0].to(device)
            # test_label = data[1]
            test_output = dia_cls_model(test_input_ids).to(device)
            pred_list = predict(sigmoid(test_output)).cpu()
            pred_all.extend(pred_list.numpy().tolist())

        # print(pred_all)
    return pred_all


def pred_result(pred):
    pred = np.array(pred).astype(int).tolist()
    pred_txt = []
    for a in pred:
        txt = ''
        for i in range(len(a)):
            if a[i] == 1 and i == 0:
                if txt != '':
                    txt += '+'
                txt += '预约上门'
            elif a[i] == 1 and i == 1:
                if txt != '':
                    txt += '+'
                txt += '客户报障'
            elif a[i] == 1 and i == 2 and len(txt) == 0:
                # if txt != '':
                #     txt += '+'
                txt += '其他'

        if txt == '':
            txt = '其他'

        pred_txt.append(txt)
    return pred_txt


if __name__ == '__main__':
    # data = ['"n0: 喂，","n1: 喂你好","n0: 那个帮我装宽带你还是什么时候过来，","n1: 哪个地方了吧","n0: 就是大盘路80号这里？","n1: 啥时才是81号吗","n0: 80号，","n1: 到也就是，","n0: 嗯嗯嗯","n1: 大直街80号是吧","n0: 嗯告诉3208号么我？","n1: 噢，那把你那个算下午了，下午一两点两点两点两点不少了吧，","n0: 原来呢你就能能能能我就在我我打一下，","n1: 嗯","n0: 多少分钟","n1: 啊等一下没了吧你是两点钟是吧，","n0: 你流量都过了我现在去我老婆过去坐下这么"',
    #         '"n0: 噢，","n1: 喂，你好我移动公司装宽带的噢，十一十一点钟11点钟过来装啊","n0: 嗯，好嘛好嘛好的再见，好您看"',
    #         '"n1: 喂，你好，你给我打电话说我没听到不好意思那个","n0: ，我是移动的，您那边地址是","n1: 噢，就是口袋小，第三个小时以后，","n0: 第三项是吧那个就是可能没这么快了，南这边嘞，因为在之前给您打电话的时候，您电话没接，然后我就约了几家，因为我约了用了满多，或者可能就没这么快再来您这边了，你再耐心等等那个就是说我在你们这边的时候我我再给你打吧，","n1: 嗯，","n0: 发给您吧，唉，","n1: 大概多久","n0: 嗯不太，","n1: 投诉","n0: 嗯，这个","n1: 不是说20号吗？那个通知发下来，","n0: 对对对我会在20号以前过来帮您装好的了","n1: 噢噢好的","n0: 嗯"',
    #         '"n0: 喂，","n1: ，嗯","n0: 嗯，你好我是那个移动宽带呢，","n1: 哦哦","n0: 然后你那个没坑，村那边是打电话跟你说有个宽带升级是吧","n1: 嗯是的","n0: 唉，然后这边后台给您升级升级好了","n1: 噢申请好了，","n0: 那到时候你网络什么有问题你记下我的号码但是网络什么有问题直接打我电话就行了啊","n1: 噢，知道就是讲这个地方是吧对对对对对，噢，你你给我申请一下告诉我你自动的是吧对对对噢，就那个包都不能使了这个，","n0: 猫不用换了，你本身那个是接到猫啊","n1: 噢，我不用了","n0: 嗯，对，那反正有问题到时候打我电话就行了啊"',
    #         '"n0: 噢，","n1: 喂，你好我移动公司装宽带的噢，十一十一点钟11点钟过来装啊","n0: 嗯，好嘛好嘛好的再见，好您看"',
    #         "n0: 喂你好，','n1: ，嗯你好','n0: 喂你那边宽带不好用了是吧','n1: 对？','n0: 啊啊，哪里啊','n1: 那个泰安西路85号？','n0: 85号几楼啊，','n1: 三楼3057','n0: 嗯好的好的你明天在不在明天密码','n1: 嗯，明天10点之后再办，','n0: 好的好的没问题，我让师傅会帮您做好过来帮您修好的麻烦是亮红灯是吧8月16号，然后每一条噢，那现在没事，没事谢谢，没校对，每天都收不过来啊，','n1: 嗯，好，','n0: 嗯，好好好，再见",
    #         "n1: 啊，唉，你好，我那个移动宽带的啊，啊，唉，你那个嗯，双溪路340','n0: 5号庄家移动宽带啊','n1: 噢，是？','n0: 哪个位置啊，','n1: 就像河里边上那个大盒里边三个点，就是那个扣了那个那个那个，','n0: 南山西那个小区门口啊','n1: 对对对，就这里不是有个家庭一样通过的吗？你现在过来装吗？我','n0: 现在不是现在晚上8点过了我就是白天联系你都不用接听的，没人接电话','n1: 不是我是你4点半以后打我电话四点四点钟以后打我电话吧白天要装机？','n0: 那你要那什么时候休息呢，','n1: 我我就是我在店里的我都在店里的就是四点钟以后才会到店里','n0: 您办理的','n1: 嗯，嗯','n0: 那你说一下什么时候被你这样子4点半以后我们五五点钟你你要拉线，要配数据的吗，','n1: 噢那你要几点钟过来装啊','n0: 你看一下要不中午的时候有没有时间啦？在一二点钟','n1: 我是起不来啊我都我做夜宵我都通宵的唉','n0: 噢那我下午大概两三点钟好吧，','n1: 要不要不三点过吧3点半左右吧，','n0: 噢，因为要配数字的，我怕等下玩那个，','n1: 嗯','n0: 数据如果有问题的话你等一下又装不上','n1: 这样的你什么时候过来装，明天还是什么时候，','n0: 那你就明天吧，我本来今天过来这样打你电话没人接的，','n1: 唉呀，我今天那个我我那个上面那个，然后我昨天办号码那个人跟我说我说4点半以后他说可以的，可以吗？我就说10点半以后了那我说怎么今天我四点钟就到店里了，然后等当时怎么还没到','n0: 那个是业务员，业务员，他受理业务的它，嗯都有时间，','n1: 可能不知道啊，嗯','n0: 要安心','n1: 对','n0: 电脑上装了，','n1: 噢，好那我明天那个要不我明天3点半，嗯3点半多一点到店里可以吧','n0: 嗯，尽量稍微少一点呢，','n1: 那我等我到地我给你打电话吧','n0: 嗯，行行行，尽量搞稍微少一点",
    #          "n0: 噢喂你好，','n1: ，嗯？','n0: 您好，我这边是移动宽带，中国好像给您打过电话的吧，嗯嗯一是今天远吗，','n1: 嗯，','n0: 今天也有集团了','n1: 有市场什么呢','n0: 噢68那那我明天过去看一下吧？因为那单子转过来转过去，耽误了好久，我们现在都下班了好吧','n1: 嗯是我是我们搞错了不用看了是我们搞错了我明天过去好了先开通了吧','n0: 开通噢，您那边是怎么回事','n1: 把它报停了我忘了，','n0: 不，','n1: 刚才你们客服打电话过来，我跟他说过了','n0: 噢，也就是你那边是已经报停停机了是吧，','n1: 对对对对，是我们搞错了不好意思啊，到时你们工作的失误，对不起啊','n0: 噢那我先把就把这个单子给报掉唉','n1: 啊，对对对对','n0: 那是改明天吧，今天时间早','n1: 一次，','n0: 噢，我改的号",
    #         "n0: 啊，','n1: 喂','n0: 我家宽带又没网了是吧','n1: 嗯，对对对','n0: 哦没有网络就是对的啊那个','n1: 什么网址','n0: 啊？那房子把把主线拆掉了','n1: 那那怎么弄','n0: 怎么弄本嘛，本人没收到就修修不了就没办法了15，','n1: 不了，我这个就就没网了','n0: 嗯，','n1: 唉','n0: 能修我们尽量收嘛收不了我们还能上天啊，你这是不是嗯，','n1: 那你们移动不可能就这样，你肯定要牵线过来呀，','n0: 权限我们也就是今年尽力而为的经理给我没办法但是那边今天两台挖机在那里推房子嘛，全部把那线给抢了，','n1: 我我怀疑也是这个问题，那那假如是大概哪个答复我有没有往你说一下吧要，','n0: 噢我们明天会安排人去修的，明天要有个大工程也修修好几百米啊，这个全部是电线杆上的大写','n1: 我知道我开了今天外面再开房子我怀疑就是这个问题，他我回来的时候我家里人说下午就没有了应该就是网速的问题','n0: 对啊还有就是说你们有问题直接打电话给我嘛，不要打那个10086嘛，猫上有电话的。','n1: 噢那我没注意我，我没注意好的好的','n0: 因为你们打打了一次还好打两次第二次的话我要我要扣200块了','n1: 噢好的那我现在这个，','n0: 200那几十个人200我的送不了啊，','n1: 嗯那我知道那那那这个电话可以联系你了','n0: 嗯，对对对','n1: 噢好的那你尽量帮我那抢修吧好吧因为这两个要注意一段时间呢没网的话也不行，就是你，','n0: 加几几十家上百家呀，','n1: 嗯好的好的','n0: 这个刚才我们都很郁闷死了现在，','n1: 好的那你明天过来看一下吧噢，','n0: 嗯我们知道的，我们知道的都主要是现在打电话没网的都是都是这个原因，','n1: 嗯，好的好的好就这样说','n0: 嗯，好好",
    #         ]
    # pred = test(data)
    # print(pred_result(pred))

    app.run(host="0.0.0.0", port=8080, debug="True")


