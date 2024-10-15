import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,precision_score,recall_score
from preprocessing import MyCustomData

#토픽 모델링의 결과를 이용하여 해당 클러스터가 정상 텍스트로 구성된 클러스터인지 마약 매매 관련 텍스트로 구성된 클러스터인지 판단하는 함수
#모델은 이전에 학습한 모델을 이용함.

def cluster_decision(topic_result,model,batch_size,cls_tokenizer,is_test = False):
    #tokenizer
    #cls_tokenizer = 'google-bert/bert-base-multilingual-cased'
    #sequence length
    seq_len = 128#sequence length
    
    topic_test = MyCustomData(topic_result,cls_tokenizer,max_len = seq_len)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    topic_test = DataLoader(topic_test,
                            batch_size = batch_size,
                            num_workers=0)
    cls_result = []
    with torch.no_grad():
        for batch_idx,data in tqdm(enumerate(topic_test)):
    
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)

            out = model(input_ids,attention_mask = attention_mask)
            out = torch.sigmoid(out)
            out = out.cpu().detach().numpy()
            #print('out 1: ',out)
            out = np.argmax(out,axis = 1)
            #print('out 2 : ',out.tolist())
            cls_result += out.tolist()
    topic_result['cls_result'] = cls_result
    if is_test:
        return topic_result
    #각 텍스트마다 0과 1로 구분한 뒤에 하나의 토픽에 대해 1이 50% 이상이면 해당 토픽은 마약 매매 관련 토픽으로 판단
    #이를 통해 해당 토픽에 대한 label을 부여함.
    MAX_topic_num = max(topic_result['topic'])
    topic_result['vote'] = np.zeros(len(topic_result),dtype = int)
    for i in range(-1,MAX_topic_num+1):
        one_topic = topic_result[topic_result['topic'] == i]
        topic_result.loc[topic_result['topic'] == i,'vote'] = np.where(np.sum(one_topic['cls_result']) > len(one_topic['cls_result'])/2,1,0)
    return topic_result