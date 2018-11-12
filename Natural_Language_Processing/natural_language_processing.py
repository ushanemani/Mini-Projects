# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 08:29:06 2018
@author: uknemani
"""
import pandas as pd

dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

# here the data is already preprocessed.
# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#creating Bag of WORDS model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X= cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values
  
# Split the dataset into train n test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state =0)


#creating model
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix
models = [GaussianNB() ]
for model in models:
  model.fit(X_train,y_train)
  y_pred = model.predict(X_test)
  cm= confusion_matrix(y_pred,y_test)
  print(cm)