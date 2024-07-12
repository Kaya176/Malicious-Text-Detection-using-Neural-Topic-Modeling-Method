import load_data
from preprocessing import MyCustomData
from transformers import AutoConfig,AutoTokenizer, AutoModel
from preprocessing import MyCustomData
from ClassificationModel import Classification_model
from classification_model_train import clssification_train
from topic_modeling import topic_modeling
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")
file_name = "DRUG_original_data.xlsx"
#Argument parser
parser = ArgumentParser()
parser.add_argument("--file_name",type = str,default = "DRUG_original_data.xlsx")
parser.add_argument("--cls_ratio",type = float,default = 0.2)
parser.add_argument("--topic_ratio",type = float,default = 0.8)
parser.add_argument("--test_ratio",type = float,default = 0.2)
parser.add_argument("--batch_size",type = int,default = 16)
parser.add_argument("--num_cls",type = int,default = 2)
parser.add_argument("--seq_len",type = int,default = 128)
parser.add_argument("--epochs",type = int,default = 3)
parser.add_argument("--cls_tokenizer",type = str,default = "google-bert/bert-base-multilingual-cased")
parser.add_argument("--cls_model_name",type = str,default = "google-bert/bert-base-multilingual-cased")
args = parser.parse_args()

#parameters

cls_ratio = 0.2
topic_ratio = 1-cls_ratio
test_ratio = 0.2

batch_size = 16
num_cls = 2
seq_len = 128

epochs = 3
#huggning face의 auto tokenizer와 auto model을 사용하였기 때문에 다른 모델로 변경 가능함.
cls_tokenizer = "google-bert/bert-base-multilingual-cased"
cls_model_name = "google-bert/bert-base-multilingual-cased"

#dataset
dataset = load_data.load_data(file_name,cls_ratio,test_ratio)

cls_data = dataset.get_cls_data()
topic_data = dataset.get_topic_data()
test_data = dataset.get_test_data()

####train_classification model
#clssification model - data
cls_train_data = MyCustomData(cls_data,cls_tokenizer,max_len = seq_len)
#classifciation model - model
clssification_model = Classification_model(batch_size = batch_size,num_cls = num_cls,seq_len = seq_len,model_name = cls_model_name)

#clssitication model - train
print("Start training classification model...")
cls_fine_tuned = clssification_train(clssification_model,cls_data,batch_size,epochs=epochs)
print("Finish training classification model...")
#topic modeling
print("Start topic modeling...")
topic_data = topic_modeling(topic_data,cls_fine_tuned,cls_tokenizer,seq_len)
print("Finish topic modeling...")
#topic modeling의 각 cluster를 cls 모델을 이용하여 정확도를 체크

#test dataset을 이용하여 모델을 평가
