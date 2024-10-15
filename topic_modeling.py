from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import pandas as pd

def topic_modeling(topic_data_path,topic_model,test = False):
    PATH_BASE = './data/'
    #data
    topic_data = pd.read_csv(PATH_BASE + topic_data_path)
    text_data = topic_data['text'].tolist()

    if test:
        topics, probs= topic_model.transform(text_data)
    else:
        topics, probs = topic_model.fit_transform(text_data)

    topic_data['topic'] = topics
    
    return topic_data,topic_model