import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle


import re
import nltk
from nltk.util import pr
stemmer = nltk.SnowballStemmer('english')
from nltk.corpus import stopwords
import string
Stopword = set(stopwords.words('english'))

df = pd.read_csv('labeled_data.csv')
print(df.head())

df['labels'] = df['class'].map({0:"Hate Speech", 1:"Offensive Language", 2:"Normal"})
print(df.head())

df = df[['tweet', 'labels']]
print(df.head())

def clean (text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+','',text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split() if word not in Stopword]
    text = " ".join(text)
    return text

df['tweet'] = df['tweet'].apply(clean)

print(df.head())


x = np.array(df['tweet'])
y = np.array(df['labels'])

cv = CountVectorizer()
x = cv.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
clf.score(x_test, y_test)

test_data = 'this is my dog and this is my bitch'
df = cv.transform([test_data]).toarray()
print(clf.predict(df))

with open('hate-speech.pkl', 'wb') as f:
    pickle.dump(clf, f)

