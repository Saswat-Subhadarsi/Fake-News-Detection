import numpy as np
import pandas as pd
import streamlit as st
import re
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords

news_df = pd.read_csv('WELFake_Dataset.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['title'] + ' '+ news_df['text']
ps = PorterStemmer()
stopwords = set(stopwords.words('english'))
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', " ", content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content=[ps.stem(word) for word in stemmed_content if not word in stopwords]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content
news_df['content'] = news_df['content'].apply(stemming)
X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 1)
model = LogisticRegression()
model.fit(X_train, y_train)
train_y_pred = model.predict(X_train)
test_y_pred = model.predict(X_test)


st.title('Fake News')
input_text = st.text_input("Enter a news article")

def prediction(input_text):
    input_data = vector.transform(stemming([input_text]))
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred==1:
        st.write("The News is Fake")
    else:
        st.write("The News is Real")

st.write("Dataset link : https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification")
