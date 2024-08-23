import load_data
from preprocessing import MyCustomData
from transformers import AutoConfig,AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from preprocessing import MyCustomData
from ClassificationModel import Classification_model
from classification_model_train import clssification_train
from topic_modeling import topic_modeling
from classification_model_valid import cluster_decision
from argparse import ArgumentParser
import warnings
from sklearn.metrics import accuracy_score,precision_score,recall_score
warnings.filterwarnings("ignore")
file_name = "DRUG_original_data.xlsx"
#Argument parser
parser = ArgumentParser()
parser.add_argument("--file_name",type = str,default = "DRUG_original_data.xlsx")
parser.add_argument("--cls_ratio",type = float,default = 0.1)
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

epochs = 1
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
#topic modeling
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
topic_model = BERTopic(embedding_model=embedding_model)
#clssitication model - train
print("Start training classification model...")
cls_fine_tuned = clssification_train(clssification_model,cls_data,batch_size,epochs=epochs)
print("Finish training classification model...")
#topic modeling
print("Start topic modeling...")
topic_data,topic_model = topic_modeling("0.8_topic_data.csv",topic_model)
print("Finish topic modeling...")
#topic modeling의 각 cluster를 cls 모델을 이용하여 정확도를 체크
result = cluster_decision(topic_data,cls_fine_tuned,batch_size)
print("precision : ",precision_score(result['label'],result['vote']))
print("recall : ",recall_score(result['label'],result['vote']))
print("accuracy : ",accuracy_score(result['label'],result['vote']))
print("Finish classification model evaluation...")
result.to_csv("result.csv",index = False)
#test dataset을 이용하여 모델을 평가함.
#위의 과정을 하나의 모델로 만들고, 테스트 데이터 넣으면 바로 결과가 나오도록 함
#테스트 데이터 -> 토픽 모델링을 이용하여 클러스터화 -> 각 클러스터에 대해 cls 모델을 이용하여 판단 -> 결과 출력
#테스트 데이터에 대한 정확도, 재현율, 정밀도를 출력함.
test_topic_result,_ = topic_modeling("0.2_test_data.csv",topic_model,test= True)
test_result = cluster_decision(test_topic_result,cls_fine_tuned,batch_size)
print("precision : ",precision_score(test_result['label'],test_result['vote']))
print("recall : ",recall_score(test_result['label'],test_result['vote']))
print("accuracy : ",accuracy_score(test_result['label'],test_result['vote']))
print("Finish test data evaluation...")
#단순히 분류 모델의 성능을 출력
print("classification model evaluation...")
test_data_cls = cluster_decision(test_data,cls_fine_tuned,batch_size,is_test = True)
print("precision : ",precision_score(test_data_cls['label'],test_data_cls['cls_result']))
print("recall : ",recall_score(test_data_cls['label'],test_data_cls['cls_result']))
print("accuracy : ",accuracy_score(test_data_cls['label'],test_data_cls['cls_result']))
print("Finish classification model evaluation...")

#readme 작성하기
#github에 업로드하기