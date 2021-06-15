import streamlit as st
import pandas as pd
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

corpus = []
voc_size = 10000
sent_length= 50
model = keras.models.load_model('model/model.h5')

st.title("Fake News Detection")

nav = st.sidebar.radio("Navigation",["HOME", "CHECK YOUR NEWS"])

if nav == "HOME":
    st.image("images//news.jpg", width = 500)

    st.header("Many a times these days we find ourselves caught up with misinformation due to coming in touch with fake news on the internet. This application will help you to stay away from such scams. Hope you find it useful. Thanks for using!")
    st.subheader("This video shows the results of a research conducted by MIT, showing the spread and impact of FAKE NEWS!!")
    video_file = open('images//video.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    graph = st.selectbox("*DO YOU WANNA SEE A VISUAL REPRESENTATION OF OUR DATASET ?*", ["Yes", "No"])

    if graph == "Yes":
       chart_data = pd.DataFrame(
       [0,1,0,0,1],
       ['News1', 'News2', 'News3', 'News4', 'News5'])
       st.bar_chart(chart_data)
       st.write("Here 1 means the news is fake, 0 means news is real")
       st.write("This bargraph is just a basic representation of our dataset")

    if graph == "No":
        st.subheader("OKAY SO GO TRY OUR MODEL OUT!")

elif nav == "CHECK YOUR NEWS":
    st.header("CHECK YOUR NEWS HERE!")
    news_title = st.text_area('Enter your news title below')
    X = []
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
