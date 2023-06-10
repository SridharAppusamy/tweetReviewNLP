from flask import Flask,render_template,request
import pickle
import flask
import pickle
import pandas as pd
import numpy as np
import nltk
import string
import re


# Removing URLs
def remove_url(text):
    return re.sub(r"http\S+", "", text)

#Removing Punctuations
def remove_punct(text):
    new_text = []
    for t in text:
        if t not in string.punctuation:
            new_text.append(t)
    return ''.join(new_text)


#Tokenizer
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')



#Removing Stop words
from nltk.corpus import stopwords

def remove_sw(text):
    new_text = []
    for t in text:
        if t not in stopwords.words('english'):
            new_text.append(t)
    return new_text

#Lemmatizaion
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def word_lemmatizer(text):
    new_text = []
    for t in text:
        lem_text = lemmatizer.lemmatize(t)
        new_text.append(lem_text)
    return new_text

#Use pickle to load in the pre-trained model.
with open(f'model/twitter_predictions.pkl', 'rb') as f:
    model = pickle.load(f)

with open(f'model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app=Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return "<h1>Hello World<h1/>"


@app.route('/predict', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return (render_template('index.html'))

    if request.method == 'POST':
        tweet = request.form['tweet']

    df = pd.DataFrame([tweet], columns=['tweet'])

    df['tweet'] = df['tweet'].apply(lambda t: remove_url(t))

    df['tweet'] = df['tweet'].apply(lambda t: remove_punct(t))

    df['tweet'] = df['tweet'].apply(lambda t: tokenizer.tokenize(t.lower()))

    df['tweet'] = df['tweet'].apply(lambda t: remove_sw(t))

    df['tweet'] = df['tweet'].apply(lambda t: word_lemmatizer(t))

    final_text = df['tweet']

    final_text.iloc[0] = ' '.join(final_text.iloc[0])

    final_text = vectorizer.transform(final_text)

    prediction = model.predict(final_text)

    return render_template('index.html', result=prediction, original_input={'Mobile Review': tweet})


if __name__ == '__main__':
    app.run(debug=True)