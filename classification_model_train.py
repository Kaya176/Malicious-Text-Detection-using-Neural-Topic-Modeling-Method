import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,precision_score,recall_score
from preprocessing import MyCustomData

def clssification_train(model,train_dataset,batch_size,epochs,cls_tokenizer):

    #load data
    #input : train data & valid data
    #input type : dataframe
    #output : fine-tuned model
    #cls_tokenizer = 'google-bert/bert-base-multilingual-cased'
    seq_len = 128
    #print(train_dataset)
    skf = StratifiedKFold(n_splits = 5,shuffle = True,random_state = 42)
    for fold,(train_data,valid_data) in enumerate(skf.split(train_dataset,train_dataset['label'])):

        train = train_dataset.iloc[train_data]
        valid = train_dataset.iloc[valid_data]

        train_data = MyCustomData(train,cls_tokenizer,max_len = seq_len)
        valid_data = MyCustomData(valid,cls_tokenizer,max_len = seq_len)
        
        train_data = DataLoader(train_data,
                            batch_size = batch_size,
                            num_workers=0)

        valid_data = DataLoader(valid_data,
                            batch_size = batch_size,
                            num_workers=0)
        
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        optimizer = optim.AdamW(model.parameters(),lr = 1e-6)
        loss_fn = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            total_loss = 0

            model.train()
            for batch_idx,data in tqdm(enumerate(train_data)):

                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                
                label = torch.tensor(data['label'],dtype = torch.float,device = device)

                out = model(input_ids,attention_mask = attention_mask)
                optimizer.zero_grad()

                loss = loss_fn(out,label)
                loss.backward()

                optimizer.step()
                total_loss += loss.detach().item()

                if batch_idx % 100 == 0:
                    print(f"Epoch : {epoch+1} | train loss : {loss:.4f}")

            #validation
            with torch.no_grad():
                predictions = []
                labels = valid['label'].tolist()
                for batch in valid_data:
                    #model input
                    input_ids = batch['input_ids'].to(device)
                    mask = batch['attention_mask'].to(device)
                    #answer
                    #model prediction
                    y_pred = model(input_ids = input_ids,attention_mask = mask)
                    y_pred = y_pred.cpu().numpy()
                    predictions += list(np.argmax(y_pred,axis=1))
                
                #print validation result
                print(f"Accuracy : {accuracy_score(labels,predictions):.4f} | Precision : {precision_score(labels,predictions):.4f}| Recall : {recall_score(labels,predictions):.4f}")
                
    return model