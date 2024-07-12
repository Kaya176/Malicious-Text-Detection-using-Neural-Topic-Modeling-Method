#library import
import pandas as pd
import string
import numpy as np
import re
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class MyCustomData(Dataset):

    def __init__(self,data,tokenizer,max_len):
        self.punc = string.punctuation
        self.punc_table = dict((ord(char),u" ") for char in self.punc)
        self.df = data
        #data preprocessing
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_len = max_len

    def _remove(self,text):
        #remove link and images
        text = text.split()
        text = " ".join([s for s in text if 'http' not in s and 'pic' not in s])

        #data cleaning
        text = re.sub("-"," ",text)
        text = re.sub(r"@", " ",text)
        text = re.sub("\'", "", text)
        text = text.translate(self.punc_table)
        text = text.replace("..", "").strip()
        text = text.lower()
        return ' '.join([t for t in text.split() if t != "" and t.find(" ") == -1])
    
    def __getitem__(self, index):
        #get one data(line)
        line = self.df.iloc[index]
        #simple preprocessing
        text = line['text']
        label = line['label']

        text = self._remove(text)
        onehot = np.zeros(2)
        onehot[label] = 1

        tokenized_text = self.tokenizer.encode_plus(text,
                                                    max_length = self.max_len,
                                                    pad_to_max_length = True,
                                                    truncation = True,
                                                    return_tensors = 'pt')
        
        result = {"input_ids":tokenized_text['input_ids'].squeeze(0),
                  "attention_mask":tokenized_text['attention_mask'].squeeze(0),
                  "label":onehot}
        
        return result
    
    def __len__(self):
        return len(self.df)