#prepare a model for sentiment analysis os movies dataset

#imorting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score,recall_score,f1_score
import joblib

#import libraries for nlp
import nltk #natural language tool
import re #regular expression

from nltk.corpus import stopwords #library for importing stopwords

#download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

print(stop_words)

#load the dataset into dataframe
df = pd.read_csv('IMDB Dataset.csv')

df.shape

df.info()

df["review"].value_counts()

df["sentiment"].value_counts()

df["sentiment"] = df["sentiment"].map({"positive":1,"negative":0})

#clean text
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]"," ",text).lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

#apply the clean text fuction
df["cleaned_review"] = df["review"].apply(clean_text)

df["cleaned_review"]

#feature extraction
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned_review"])

y = df["sentiment"]

#divide the dataset into train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#train the model
model = MultinomialNB()
model.fit(X_train, y_train)

#make predictions
y_pred = model.predict(X_test)

#calculate the performance metrics
accuracy = accuracy_score(y_pred,y_test)
precision = precision_score(y_pred, y_test)
recall = recall_score(y_pred,y_test)
f1 = f1_score( y_pred,y_test)
cm = confusion_matrix(y_pred,y_test)
cr = classification_report(y_pred,y_test)

print("The accuracy is :", accuracy)
print("The precision is :", precision)
print("The recall is :" ,recall)
print("The f1 score is :", f1)
print("*********The confusion matrix is********* :", cm)
print("*************The classification report is********** :" ,cr)

joblib.dump(model,"sentiment_analysis_model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")