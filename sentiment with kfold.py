# -*- coding: utf-8 -*-


import pandas as pd
import nltk
import re
from sklearn import *
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics



def data_preprocessing(tweets):
    ProcessedTweets = []
    for each in tweets:
        if type(each) is str:
            tweets = each.lower() #convert to lowercase
        tweets = re.sub(r'(http|https|ftp)://[a-zA-Z0-9\\./]+', '', tweets) #remove URL
        tweets = re.sub(r'(\s)@\w+', r'', tweets) #remove handles
        tweets = re.sub(r'@(\w+)', r'', tweets) #remove handles
        tweets = re.sub('<[^<]+?>', '', tweets)  # remove html
        tweets = re.sub(r'[<>!#@$:.,%\?-]+', r' ', tweets) #remove punctuation and special characters
        tweets = re.sub(r'(.)\1{1,}',r'',tweets) #remove repeated characters
        stop_words = set(stopwords.words('english'))
        word_tokens = tweets.split()
        filtered_sentence = [w for w in word_tokens if not w in stop_words] #removing stopwords
        #word_tokens = filtered_sentence.split()
        ps = PorterStemmer()
        stemmed_sentence =[]
        for w in filtered_sentence:
            stemmed_sentence.append(ps.stem(w))
        stemmed_sentence = " ".join(stemmed_sentence)
        #print (tweets)
        #print ('\n')
        ProcessedTweets.append(stemmed_sentence)
    return ProcessedTweets


xls_romney = pd.read_excel("training-Obama-Romney-tweets.xlsx",sheet_name='Romney')
xls_romney = xls_romney[(xls_romney['Class'].isin((1,-1,0)))]
romney_tweet = xls_romney['Anootated tweet'].tolist()
romney_class = xls_romney['Class'].tolist()
romney_tweet = data_preprocessing(romney_tweet)
tfidf = TfidfVectorizer(stop_words='english',min_df=5) #vectorisation
tweet_vectors = tfidf.fit_transform(romney_tweet)



#Accuracy using Naive Bayes Model
NB = MultinomialNB()
preds = model_selection.cross_val_predict(NB, tweet_vectors, romney_class, cv=10)

print('\nNaive Bayes')
print('Accuracy Score: ',metrics.accuracy_score(romney_class,preds)*100,'%',sep='')
labels = [1,-1]
precision = metrics.precision_score(romney_class,preds,average=None,labels=labels)
recall = metrics.recall_score(romney_class,preds,average=None,labels=labels)
f1Score = metrics.f1_score(romney_class,preds,average=None,labels=labels)
lbl = ['positive', 'negative']
for i in range(2):
    print("Precision of %s class: %f" %(lbl[i],precision[i]*100))
    print("Recall of %s class: %f" %(lbl[i],recall[i]*100))
    print("F1-Score of %s class: %f" %(lbl[i],f1Score[i]*100),"\n")






