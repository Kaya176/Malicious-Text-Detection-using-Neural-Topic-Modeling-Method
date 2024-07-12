import torch
import torch.nn as nn
from transformers import AutoModel
import math

class Classification_model(nn.Module):

    def __init__(self,batch_size,num_cls,seq_len,model_name = "google-bert/bert-base-multilingual-cased"):
        super(Classification_model,self).__init__()
        self.num_cls = num_cls
        self.batch_ize = batch_size
        self.seq_len = seq_len
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop1 = nn.Dropout(0.1)

        #fnn layer
        self.fnn_layer = nn.Linear(self.bert.config.hidden_size*self.seq_len,256)
        #cnn layer
        self.cnn = nn.Conv1d(in_channels= self.bert.config.hidden_size,out_channels=256,kernel_size= 2, stride = 1)
        self.pool = nn.MaxPool1d(2,stride = 1)
        self.cnn_to_fnn = nn.Linear(self._calcualte_output_feature_cnn(),256)
        #lstm layer
        self.lstm = nn.LSTM(self.bert.config.hidden_size,256,num_layers = 3,dropout = 0.2,batch_first = True)
        #classification layer
        self.linear_1 = nn.Linear(256,64)
        self.linear_2 = nn.Linear(64,self.num_cls)
    
    def _calcualte_output_feature_cnn(self):
        out_conv = (self.seq_len-1*(2-1)-1)+1
        out_conv = math.floor(out_conv)
        out_pool = (out_conv-1*(2-1)-1)+1
        out_pool = math.floor(out_pool)
        return out_pool * 256
    

    def forward_cnn(self,bert_output):
        conv_output = self.cnn(bert_output)
        conv_output = nn.functional.sigmoid(conv_output)
        conv_output = self.pool(conv_output)
        conv_output = torch.flatten(conv_output,1)
        return conv_output

    def forward_fnn(self,bert_output):
        fc_output = self.fnn_layer(bert_output)
        return fc_output

    def forward_lstm(self,bert_output):
        lstm_output,_ = self.lstm(bert_output)
        lstm_output = lstm_output[:,-1,:]
        return lstm_output

    def forward(self,input_ids,attention_mask,add_layer = None):
        x = self.bert(input_ids = input_ids,attention_mask = attention_mask)
        x = x.last_hidden_state.transpose(1,2)

        if add_layer == 'cnn':
            output = self.forward_cnn(x)
            output = self.cnn_to_fnn(output)

        elif add_layer == 'lstm':
            output = self.forward_lstm(x)
        else:
            x = torch.flatten(x,1)
            output = self.forward_fnn(x)

        output = self.linear_1(output)
        output = self.drop1(output)
        output = nn.functional.relu(output)
        output = self.linear_2(output)
        return output