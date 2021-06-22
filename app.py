import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import nltk
import joblib
from nltk.corpus import stopwords
nltk.download('stopwords')

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import string 
import keras
from keras.preprocessing import text,sequence
import dill

voc_size = 10000
news_title = ""
sent_length= 50
model = keras.models.load_model('model/model.h5')

st.title("Fake News Detection")

nav = st.sidebar.radio("Navigation",["HOME", "CHECK YOUR NEWS"])

if nav == "HOME":
    st.image("images//news.jpg", width = 500)

    st.header("Many a times these days we find ourselves caught up with misinformation due to coming in touch with fake news on the internet. This application will help you to stay away from such scams. Hope you find it useful. Thanks for using!")
    st.subheader("This video shows the results of a research conducted by MIT, showing the spread and impact of FAKE NEWS!!")
    video_file = open('images//Video.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

elif nav == "CHECK YOUR NEWS":
    st.header("CHECK YOUR NEWS HERE!")
    news_title = st.text_area('Enter your news title below')

X = list()
X.append(news_title)
tokenizer = open('model/tokenizer.pkl', 'rb')
tokenized = joblib.load(tokenizer)
max_len = 300
tokenized_pred = tokenized.texts_to_sequences(X)
X = sequence.pad_sequences(tokenized_pred, maxlen=max_len)

prediction = model.predict_classes(X)

    
if st.button("Detect"):
        if prediction[0] == 1:
          st.success("Your news is FAKE!")
        else:
            st.success("Your news is REAL!")
