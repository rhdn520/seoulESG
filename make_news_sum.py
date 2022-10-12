# -*- coding: utf-8 -*-
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import math
import pandas as pd
import numpy as np

from kiwipiepy import Kiwi
kiwi = Kiwi()
# import kss

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim

from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

from torch.nn.init import xavier_uniform_

import pytorch_lightning as pl
# from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

import seaborn as sns
# from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

from itertools import count
from os.path import exists
import requests
import csv
import datetime
import json
import re
# from kiwipiepy import Kiwi
# kiwi = Kiwi()
# kiwi.load_user_dictionary('user_dict.txt')
with open(f'dictionary/gov-public-agent-list.csv', 'r', newline = '', encoding='utf-8-sig') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        kiwi.add_user_word(row['name'])
with open('dictionary/corporation-list.csv', 'r', newline='', encoding='utf-8-sig') as csvfile:
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader):
        if(row['name'].count(' ') > 0): #기업명에 띄어쓰기 포함된 경우 제외
            continue
        kiwi.add_user_word(row['name'])
from datetime import datetime
from datetime import timedelta
from collections import Counter
# import inquirer



MAX_TOKEN_COUNT = 512
BERT_MODEL_NAME = 'kpfbert/'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]

class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        if (self.use_final_linear):
            self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None, predefined_graph_1=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query), \
                                    self.linear_keys(query), \
                                    self.linear_values(query)

                key = shape(key)
                value = shape(value)

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat(
                            (layer_cache["self_keys"].to(device), key),
                            dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat(
                            (layer_cache["self_values"].to(device), value),
                            dim=2)
                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        key, value = self.linear_keys(key), \
                                     self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache["memory_keys"], \
                                     layer_cache["memory_values"]
                    layer_cache["memory_keys"] = key
                    layer_cache["memory_values"] = value
                else:
                    key, value = self.linear_keys(key), \
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18) # how can i fix it to use fp16...

        # 3) Apply attention dropout and compute context vectors.

        attn = self.softmax(scores)

        if (not predefined_graph_1 is None):
            attn_masked = attn[:, -1] * predefined_graph_1
            attn_masked = attn_masked / (torch.sum(attn_masked, 2).unsqueeze(2) + 1e-9)

            attn = torch.cat([attn[:, :-1], attn_masked.unsqueeze(1)], 1)

        drop_attn = self.dropout(attn)
        if (self.use_final_linear):
            context = unshape(torch.matmul(drop_attn, value))
            output = self.final_linear(context)
            return output
        else:
            context = torch.matmul(drop_attn, value)
            return context

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)

class ExtTransformerEncoder(nn.Module):
    def __init__(self, hidden_size=768, d_ff=2048, heads=8, dropout=0.2, num_inter_layers=2):
        super(ExtTransformerEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, hidden_size)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(hidden_size, heads, d_ff, dropout)
            for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.wo = nn.Linear(hidden_size, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~mask) 

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores

class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def gelu(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


    def forward(self, x):
        inter = self.dropout_1(self.gelu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)

class Summarizer(pl.LightningModule):

    def __init__(self, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.max_pos = 512
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME) #, return_dict=True)
        self.ext_layer = ExtTransformerEncoder()
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.loss = nn.BCELoss(reduction='none')
    
        for p in self.ext_layer.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, segs, clss, labels=None): #, input_ids, attention_mask, labels=None):
        
        mask_src = ~(src == 0) #1 - (src == 0)
        mask_cls = ~(clss == -1) #1 - (clss == -1)

        top_vec = self.bert(src, token_type_ids=segs, attention_mask=mask_src)
        top_vec = top_vec.last_hidden_state
        
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()

        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        
        loss = 0
        if labels is not None:
            loss = self.loss(sent_scores, labels)
            
            loss = (loss * mask_cls.float()).sum() / len(labels)
        
        return loss, sent_scores
    
    def step(self, batch):

        src = batch['src']
        if len(batch['labels']) > 0 :
            labels = batch['labels']
        else:
            labels = None
        segs = batch['segs']
        clss = batch['clss']
        
        loss, sent_scores = self(src, segs, clss, labels)    
        
        return loss, sent_scores, labels

    def training_step(self, batch, batch_idx):

        loss, sent_scores, labels = self.step(batch)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        
        return {"loss": loss, "predictions": sent_scores, "labels": labels}

    def validation_step(self, batch, batch_idx):
        
        loss, sent_scores, labels = self.step(batch)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        
        return {"loss": loss, "predictions": sent_scores, "labels": labels}

    def test_step(self, batch, batch_idx):
        
        loss, sent_scores, labels = self.step(batch)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        
        return {"loss": loss, "predictions": sent_scores, "labels": labels}

    def acc_loss(self, outputs):
        total_loss = 0
        hit_cnt = 0
        for outp in outputs:
            labels = outp['labels'].cpu()
            predictions, idxs = outp['predictions'].cpu().sort()
            loss = outp['loss'].cpu()
            for label, idx in zip(labels, idxs):
                for i in range(1,3):
                    if label[idx[-i-1]] == 1 : 
                        hit_cnt += 1

            total_loss += loss
            
        avg_loss = total_loss / len(outputs)
        acc = hit_cnt / (3*len(outputs)*len(labels))
        
        return acc, avg_loss
        
    def training_epoch_end(self, outputs):
        
        acc, avg_loss = self.acc_loss(outputs)
        
        print('acc:', acc, 'avg_loss:', avg_loss)
        
        self.log('avg_train_loss', avg_loss, prog_bar=True, logger=True)

    def validation_epoch_end(self, outputs):
        
        acc, avg_loss = self.acc_loss(outputs)
        
        print('val_acc:', acc, 'avg_val_loss:', avg_loss)
        
        self.log('avg_val_loss', avg_loss, prog_bar=True, logger=True)

    def test_epoch_end(self, outputs):
        
        acc, avg_loss = self.acc_loss(outputs)
        
        print('test_acc:', acc, 'avg_test_loss:', avg_loss)
        
        self.log('avg_test_loss', avg_loss, prog_bar=True, logger=True)

        return
        
    # def configure_optimizers(self):
        
    #     optimizer = AdamW(self.parameters(), lr=2e-5)

    #     steps_per_epoch=len(train_df) // BATCH_SIZE
    #     total_training_steps = steps_per_epoch * N_EPOCHS
        
    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=steps_per_epoch,
    #         num_training_steps=total_training_steps
    #     )

    #     return dict(
    #         optimizer=optimizer,
    #         lr_scheduler=dict(
    #             scheduler=scheduler,
    #             interval='step'
    #         )
    #     )

trained_model = Summarizer.load_from_checkpoint(
    "checkpoint/best-checkpoint.ckpt"
)

def data_process(text):
    # 문장 분리 하고,
    sents = kiwi.split_into_sents(text)
    # sents = kss.split_sentences(text)
    
    #데이터 가공하고,
    tokenlist = []
    for sent in sents:
        tokenlist.append(tokenizer(
            text = sent.text,
            add_special_tokens = True)) #, # Add '[CLS]' and '[SEP]'

    src = [] # 토크나이징 된 전체 문단
    labels = []  # 요약문에 해당하면 1, 아니면 0으로 문장수 만큼 생성
    segs = []  #각 토큰에 대해 홀수번째 문장이면 0, 짝수번째 문장이면 1을 매핑
    clss = []  #[CLS]토큰의 포지션값을 지정

    odd = 0

    for tkns in tokenlist:

        if odd > 1 : odd = 0
        clss = clss + [len(src)]
        src = src + tkns['input_ids']
        segs = segs + [odd] * len(tkns['input_ids'])
        odd += 1

        #truncation
        if len(src) == MAX_TOKEN_COUNT:
            break
        elif len(src) > MAX_TOKEN_COUNT:
            src = src[:MAX_TOKEN_COUNT - 1] + [src[-1]]
            segs = segs[:MAX_TOKEN_COUNT]
            break

    #padding
    if len(src) < MAX_TOKEN_COUNT:
        src = src + [0]*(MAX_TOKEN_COUNT - len(src))
        segs = segs + [0]*(MAX_TOKEN_COUNT - len(segs))

    if len(clss) < MAX_TOKEN_COUNT:
        clss = clss + [-1]*(MAX_TOKEN_COUNT - len(clss))

    return dict(
        sents = sents, #정답 출력을 위해...
        src = torch.tensor(src),
        segs = torch.tensor(segs),
        clss = torch.tensor(clss),
    )

def summarize_test(text):
    data = data_process(text.replace('\n',''))
    
    #trained_model에 넣어 결과값 반환
    _, rtn = trained_model(data['src'].unsqueeze(0), data['segs'].unsqueeze(0), data['clss'].unsqueeze(0))
    rtn = rtn.squeeze()
    
    # 예측 결과값을 받기 위한 프로세스
    rtn_sort, idx = rtn.sort(descending = True)
    rtn_sort = rtn_sort.tolist()
    idx = idx.tolist()

    end_idx = rtn_sort.index(0)

    rtn_sort = rtn_sort[:end_idx]
    idx = idx[:end_idx]
    
    if len(idx) > 6:
        rslt = idx[:6]
    else:
        rslt = idx
        
    summ = []
    # print(' *** 입력한 문단의 요약문은 ...')
    for i, r in enumerate(rslt):
        summ.append(data['sents'][r])
        # print('[', i+1, ']', summ[i])

    # return summ
    return sorted(summ,key=lambda x:x.start)


def main():
    test_context = '''
    조주완(사진) LG전자(066570) 최고경영자(CEO) 사장이 취임 후 처음으로 협력사 간담회에 참석해 상생 협력을 위한 다양한 지원을 약속했다. \n \n\n LG전자는 최근 조 사장이 협력사 모임인 ‘협력회’ 임원들과 취임 후 첫 간담회를 가졌다고 8일 밝혔다. 간담회에는 LG전자에서 조 사장과 왕철민 구매·SCM경영센터장(전무) 등이, 협력회에서 임원단인 협력사 대표 8명이 각각 참석했다. \n \n\n 조 사장은 간담회에서 “자동화 시스템 구축, 공급망 다각화 등 제조 경쟁력을 확보하고 급변하는 시장 환경에 선제적으로 대응할 수 있도록 협력사에 실질적으로 도움이 되는 지원을 지속해서 펼쳐 상생 협력을 더욱 강화하겠다”고 약속했다. \n \n\n LG전자는 협력사 생산성 향상이 상생과 지속 가능한 성장을 위한 방안이라고 판단해 다양한 지원 활동을 펼치고 있다. 회사의 자동화 시스템 전문가를 협력사에 파견해 스마트공장 구축 노하우를 전수하고 있다. 또 로봇프로세스자동화(RPA) 기술을 도입할 수 있도록 도와 지난해 협력사 직원 중 80명 이상의 관련 전문가를 육성하고 176개 RPA를 업무에 도입하는 성과를 냈다.
    '''

    # a = kiwi.split_into_sents(test_context)
    # print(a)
    rtn = summarize_test(test_context)
    print(rtn)
    exit()



if __name__ == "__main__":
    main()







