import pandas as pd
from sklearn.model_selection import train_test_split
#데이터를 나누고, 샘플링하는 클래스
#1. Positive와 Negative의 비율을 2:1로 맞춘다.
#2. test data를 나눈다.
#3. 남은 데이터는 classification data와 topic modeling data로 나눈다.

class load_data:

    def __init__(self,file_name,cls_ratio,test_ratio):
        self.data_path = "./data/"
        self.file = pd.read_excel(self.data_path+file_name)
        #dataset
        self.train_data = None
        self.cls_data = None
        self.topic_data = None
        self.Test_data = None #for final test
        self.undersampling_ratio = 2
        #do sampling
        self.file = self._sampling()
        #divide data
        self.train_data,self.test_data = self.divide_data(test_ratio)
        #divide cls and topic data
        self.topic_data,self.cls_data = self.divide_cls_topic(cls_ratio)
        #reset index
        # self.test_data.reset_index(inplace = True)
        # self.cls_data.reset_index(inplace = True)
        # self.topic_data.reset_index(inplace = True)
        #save
        self.test_data.to_csv(self.data_path + str(test_ratio)+"_"+"test_data.csv",index = False)
        self.cls_data.to_csv(self.data_path + str(cls_ratio)+"_"+"cls_data.csv",index = False)
        self.topic_data.to_csv(self.data_path + str(1-cls_ratio)+"_"+"topic_data.csv",index = False)
    def data_augmentation(self):
        return
    
    def _sampling(self):
        positive = self.file[self.file['label']== 0]
        negative = self.file[self.file['label']== 1]
        
        pos_sample = positive.sample(n = len(negative)*self.undersampling_ratio)
        result = pd.concat([pos_sample,negative])
        result = result.sample(frac = 1)
        return result
    
    def divide_data(self,test_ratio):
        #divide data into train and test data
        train_data,test_data = train_test_split(self.file,test_size = test_ratio,random_state = 2000,stratify=self.file['label'])
        #test_data = test_data.reset_index()
        return train_data,test_data
    
    def divide_cls_topic(self,cls_ratio):
        #divide data into classification and topic modeling data
        topic_data,cls_data = train_test_split(self.train_data,test_size = cls_ratio,random_state = 2001,stratify=self.train_data['label'])
        cls_data = cls_data.reset_index(drop = True)
        topic_data = topic_data.reset_index(drop = True)
        return topic_data,cls_data
    
    def get_cls_data(self):
        return self.cls_data
    
    def get_topic_data(self):
        return self.topic_data
    
    def get_test_data(self):
        return self.test_data

if __name__ == "__main__":
    file_name = "DRUG_original_data.xlsx"
    cls_ratio = 0.2
    test_ratio = 0.2
    data_class = load_data(file_name,cls_ratio,test_ratio)

    cls_test = data_class.get_cls_data()
    topic_test = data_class.get_topic_data()
    test_data = data_class.get_test_data()
    print(cls_test.head())
    #각 데이터의 라벨 비율이 잘 나눠졌는지 출력
    print("cls_data : ",cls_test['label'].value_counts())
    print("topic_data : ",topic_test['label'].value_counts())
    print("test_data : ",test_data['label'].value_counts())
    #나중에 추가할 사항.
    #1. augmentation
    #2. 비율을 0.1~0.9까지 바꿔가며 실험헤야하기 때문에 이를 자동화하는 코드를 추가해야함.