from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import pandas as pd

def topic_modeling(topic_data_path,topic_model,test = False):
    PATH_BASE = './data/'
    #data
    topic_data = pd.read_csv(PATH_BASE + topic_data_path)
    text_data = topic_data['text'].tolist()

    #embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    #topic_model = BERTopic(embedding_model=embedding_model)

    #fit model
    if test:
        topics, probs= topic_model.transform(text_data)
    else:
        topics, probs = topic_model.fit_transform(text_data)

    #result
    #결과로 나온 텍스트마다의 topic을 부여하고, 이 dataframe을 저장한다.
    #이후, 이 topic cluster마다 이전에 학습했던 classification model을 이용하여
    #각 클러스터마다 label을 부여한다.

    topic_data['topic'] = topics
    #topic_data.to_csv(PATH_BASE + topic_data_path,index = False)
    
    return topic_data,topic_model