# -*- coding: utf-8 -*-
"""
IST 736 - Team 5: Courtney Smith, Debra Kernstock, Sukhpal Dhillon, and Teresa Cameron

Final Project: US Airlines Sentiment Analysis

Description: 
This program performs data exploration and builds algorithms, SVM and Mulitnomial 
Naive Bayes, to classify airline tweets sentiment. LDA is also used to find topics 
on additional airline tweet data obtained via the Twitter API.

Program Sections:
    Section 1: Import Libraries and Packages
    Section 2: Data Import
    Section 3: Data Exploration
    Section 4: Business Questions
    Section 5: SVM Models
    Sectoin 6: Naive Bayes Models
    Section 7: LDA - Twitter API Data

This program was coded in Google Colab and the original file is located at
    https://colab.research.google.com/drive/1pyg0HkDQ8NwQT2QEhozd8iq3DE5hPyb4
"""

# =============================================================================
# Section 1 - Import Libraries and Packages
# =============================================================================

## import modules
import pandas as pd
import numpy as np
import os
from scipy.stats import itemfreq
import random as rd

# vectorization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# models
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

# measuring results
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
import prettytable
from wordcloud import WordCloud, STOPWORDS

# set seaborn style
sns.set(style="ticks", palette="Set2") # set global color palette
sns.set_style("whitegrid")


# =============================================================================
# Section 2 - Data Import
# =============================================================================

# read in data
tweets = pd.read_csv('https://raw.githubusercontent.com/ceejux/IST-736/master/Tweets.csv')
tweets.dtypes
tweets.head(5)


# =============================================================================
# Section 3 - Data Exploration
# =============================================================================

# plot review count by sentiment
sns.countplot(x='airline_sentiment',  data=tweets)
plt.ylabel('Count')
plt.title('Review Count by Airline Sentiment Label')
plt.xlabel('Airline Sentiment Label')
plt.show()

# calculate review text length
tweets['review_length'] = tweets['text'].astype(str).apply(len)

# Tweet length
fig, ax = plt.subplots()
sns.distplot(tweets['review_length'], kde = False)
plt.ylabel('Count')
plt.xlabel('Tweet Length')
plt.title('Distribution of Tweet Length')
plt.show()

# review legnth -facetgrid
g = sns.FacetGrid(tweets, col = 'airline_sentiment', hue = 'airline_sentiment')
g.map(sns.distplot, 'review_length').set_xlabels('Tweet Length')
plt.show()

## word cloud
# convert review column into text
review_text = " ".join(review for review in tweets.text)

# Create stopword list:
stopwords = set(STOPWORDS)

# Create wordcloud object
wordcloud = WordCloud(stopwords=stopwords,max_words=250, background_color="white", colormap = "viridis").generate(review_text)

# display word cloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


# =============================================================================
# Section 4 - Business Questions
# =============================================================================

# Question 1 - V1: Which airlines have the most positive and negative tweets?
sns.countplot(x='airline', hue = 'airline_sentiment', data=tweets)
plt.ylabel('Tweet Count')
plt.title('Count of Sentiment by Airline')
plt.xlabel('Airline ')
plt.legend(loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.show()

# Question 1 - V2: Which airlines have the most positive and negative tweets?
## Tweet Sentiment Distribution - Overall
SentimentTypeCount = tweets['airline_sentiment'].value_counts() # Summarize tweet sentiment by counts
SentimentTypePercentage = tweets['airline_sentiment'].value_counts(normalize=True).round(1) *100 # Summarize tweet sentiment by percentage rounded to 1 decimal

print(SentimentTypeCount)
# graph1_SentimentType = SentimentTypeCount.plot.bar(title = 'Sentiment Type by Count')

print(SentimentTypePercentage)
graph1_SentimentTypePercentage = SentimentTypePercentage.plot.pie(title = 'Sentiment Type by Percentage')

## Tweet Sentiment Distribution - By Airline 
AirlineSentimentCount = tweets.groupby(['airline'])['airline_sentiment'].value_counts() # Summarize sentiment by airline via count

print(AirlineSentimentCount)
graph1_AirlineSentimentCount = AirlineSentimentCount.unstack(fill_value=0).plot.bar(title = 'Sentiment by Airline by Count')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()
#####


# Question 2 - V1: What is the ratio of positive and negative tweets for each airline?
tweets2 = pd.pivot_table(tweets,index=["airline", 'airline_sentiment'],values=["tweet_id"],aggfunc='count')

# calculate % of total for each airline
tweets_pcts = tweets2.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
tweets_pcts = tweets_pcts.reset_index()

# plot
sns.barplot(x='airline', y = 'tweet_id', hue = 'airline_sentiment' ,data = tweets_pcts, ci=None)
plt.ylabel('%')
plt.title('Ratio of Positive/Negative/Neutral Tweets for each Airline')
plt.xlabel('Airline')
plt.legend(loc='best')
plt.xticks(rotation=45, ha='right')
plt.show()

# Question 2 - V2: What is the ratio of positive and negative tweets for each airline?
AirlineSentimentPercentage = tweets.groupby(['airline'])['airline_sentiment'].value_counts(normalize=True).round(4) *100  # Summarize sentiment by airline via percentage rounded to 2 decimals

print(AirlineSentimentPercentage)
graph2_AirlineSentimentPercentage = AirlineSentimentPercentage.unstack(fill_value=0).plot.bar(title = 'Sentiment by Airline by Percentage')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()
#####


# Question 3: Are there variables that may influence a tweet’s sentiment? (e.g., tweet location, time of day, time zone, or day of the week)

## Time of Day
print('Number of Tweet Time Stamps:',tweets.tweet_created.nunique())

tweets['tweet_created'] = pd.to_datetime(tweets['tweet_created'])  # Convert starttime from data type object to datetime
tweets['TweetTime'] = tweets['tweet_created'].dt.hour # Retrieve hour in which tweet was sent

SentimentTimeCount = tweets.groupby(['airline_sentiment'])['TweetTime'].value_counts()
print(SentimentTimeCount)
graph3_SentimentTimeCount = SentimentTimeCount.unstack(fill_value=0).plot.bar(title = 'Sentiment Count by Time of Day')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()

## Day of the Week
tweets['tweet_created'] = pd.to_datetime(tweets['tweet_created'])  # Convert starttime from data type object to datetime
tweets['WeekDay'] = tweets['tweet_created'].dt.dayofweek

SentimentDayCount = tweets.groupby(['WeekDay'])['airline_sentiment'].value_counts()
print(SentimentDayCount)
graph3_SentimentDayCount = SentimentDayCount.unstack(fill_value=0).plot.bar(title = 'Sentiment Count by Day')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()

## Tweet Location
print('Number of Locations:',tweets.tweet_location.nunique())
SentimentLocationCount = tweets.groupby(['airline_sentiment'])['tweet_location'].value_counts()
print(SentimentLocationCount)
# graph3_SentimentLocationCount = SentimentLocationCount.plot.bar(title = 'Sentiment Count by Location')

## Time Zone
print('Number of Time Zones:',tweets.user_timezone.nunique())
SentimentZoneCount = tweets.groupby(['airline_sentiment'])['user_timezone'].value_counts()
print(SentimentZoneCount)
# graph3_SentimentZoneCount = SentimentZoneCount.plot.bar(title = 'Sentiment Count by Time Zone')
#####


# Question 4 - V1: What is the most frequent negative reason given and is it specific to an airline?
tweets3 = pd.pivot_table(tweets,index=['negativereason'],values=["tweet_id"],aggfunc='count').sort_values(by = 'tweet_id', ascending = False)

tweets3 = tweets3.reset_index()

# plot
sns.barplot(x='negativereason', y = 'tweet_id', data = tweets3, ci=None)
plt.ylabel('Tweet Count')
plt.title('Most Frequent Negative Reasons for Tweets')
plt.xlabel('Reason')
plt.xticks(rotation=45, ha='right')
plt.show()


# Question 4 - V2: What is the most frequent negative reason given and is it specific to an airline?
## Negative Tweet Reason Distribution - Overall
NegativeReasonCount = tweets['negativereason'].value_counts() # Summarize negative reasons by counts
print(NegativeReasonCount)
graph4_NegativeReasonCount = NegativeReasonCount.plot.bar(title = 'Negative Tweet Reasons by Count')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()

## Negative Tweet Reason Distribution - By Airline 
AirlineNegativeReasonCount = tweets.groupby(['airline'])['negativereason'].value_counts() # Summarize negative tweets reasons by airline via count
AirlineNegativeReasonPercentage = tweets.groupby(['airline'])['negativereason'].value_counts(normalize=True).round(4) *100  # Summarize negative tweets reasons by airline via percentage rounded to 2 decimals

print(AirlineNegativeReasonCount)
print(AirlineNegativeReasonPercentage)

graph4_AirlineNegativeReasonCount = AirlineNegativeReasonCount.unstack(fill_value=0).sort_values('airline').plot.bar(title = 'Negative Tweet Reasons by Count')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()

graph4_AirlineNegativeReasonPercentage = AirlineNegativeReasonPercentage.unstack(fill_value=0).plot.bar(title = 'Negative Tweet Reasons by Percentage')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()
#####


# =============================================================================
# Section 5 - SVM Models
# =============================================================================

# prepare the data - countvectorizer
y=tweets['airline_sentiment'].values
X=tweets['text'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(X_train[0])
print(y_train[0])
print(X_test[0])
print(y_test[0])

# Check how many training examples in each category. this is important to see whether the data set is balanced or skewed
unique, counts = np.unique(y_train, return_counts=True)
print(np.asarray((unique, counts)))


# Print out the category distribution in the test data set. 
training_labels = set(y_train)
print(training_labels)
training_category_dist = itemfreq(y_train)
print(training_category_dist)


#  unigram term frequency vectorizer, set minimum document frequency to 5
unigram_count_vectorizer = CountVectorizer(encoding='latin-1', binary=False, min_df=5, stop_words='english')


## vectorize the data
# fit vocabulary in training documents and transform the training documents into vectors
X_train_vec = unigram_count_vectorizer.fit_transform(X_train)

# check the content of a document vector
print(X_train_vec.shape)
print(X_train_vec[0].toarray())

# check the size of the constructed vocabulary
print(len(unigram_count_vectorizer.vocabulary_))

# print out the first 10 items in the vocabulary
print(list(unigram_count_vectorizer.vocabulary_.items())[:10])

# check word index in vocabulary
print(unigram_count_vectorizer.vocabulary_.get('place'))

#  Vectorize the test data
X_test_vec = unigram_count_vectorizer.transform(X_test)

# print out #examples and #features in the test set
print(X_test_vec.shape)



SVM_Model1=LinearSVC(C=1)
SVM_Model1.fit(X_train_vec, y_train)


print("SVM prediction:\n", SVM_Model1.predict(X_test_vec))
print("Actual:")
print(y_test)


SVM_matrix = confusion_matrix(y_test, SVM_Model1.predict(X_test_vec), labels=['negative', 'neutral', 'positive'])
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")


target_names = ['negative', 'neutral', 'positive']
print('SVM Results - LinearSVC (CountVectorizer - Term Frequency):\n', classification_report(y_test, SVM_Model1.predict(X_test_vec), target_names=target_names))


# try same model with different C value
SVM_Model1=LinearSVC(C=100)
SVM_Model1.fit(X_train_vec, y_train)


print("SVM prediction:\n", SVM_Model1.predict(X_test_vec))
print("Actual:")
print(y_test)


SVM_matrix = confusion_matrix(y_test, SVM_Model1.predict(X_test_vec), labels=['negative', 'neutral', 'positive'])
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")


target_names = ['negative', 'neutral', 'positive']
print('SVM Results - LinearSVC (CountVectorizer - Term Frequency):\n', classification_report(y_test, SVM_Model1.predict(X_test_vec), target_names=target_names))


# TF-IDF
y=tweets['airline_sentiment'].values
X=tweets['text'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(X_train[0])
print(y_train[0])
print(X_test[0])
print(y_test[0])

# Check how many training examples in each category. this is important to see whether the data set is balanced or skewed
unique, counts = np.unique(y_train, return_counts=True)
print(np.asarray((unique, counts)))


# Print out the category distribution in the test data set. 
training_labels = set(y_train)
print(training_labels)
training_category_dist = itemfreq(y_train)
print(training_category_dist)



#  unigram tfidf vectorizer, set minimum document frequency to 5
unigram_tfidf_vectorizer = TfidfVectorizer(encoding='latin-1', use_idf=True, min_df=5, stop_words='english')



## vectorize the data
# fit vocabulary in training documents and transform the training documents into vectors
X_train_vec = unigram_tfidf_vectorizer.fit_transform(X_train)

# check the content of a document vector
print(X_train_vec.shape)
print(X_train_vec[0].toarray())

# check the size of the constructed vocabulary
print(len(unigram_count_vectorizer.vocabulary_))

# print out the first 10 items in the vocabulary
print(list(unigram_count_vectorizer.vocabulary_.items())[:10])

# check word index in vocabulary
print(unigram_count_vectorizer.vocabulary_.get('place'))

#  Vectorize the test data
X_test_vec = unigram_count_vectorizer.transform(X_test)

# print out #examples and #features in the test set
print(X_test_vec.shape)



SVM_Model1=LinearSVC(C=1)
SVM_Model1.fit(X_train_vec, y_train)

print("SVM prediction:\n", SVM_Model1.predict(X_test_vec))
print("Actual:")
print(y_test)



SVM_matrix = confusion_matrix(y_test, SVM_Model1.predict(X_test_vec), labels=['negative', 'neutral', 'positive'])
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")


target_names = ['negative', 'neutral', 'positive']
print('SVM Results - LinearSVC (CountVectorizer - TF-IDF):\n', classification_report(y_test, SVM_Model1.predict(X_test_vec), target_names=target_names))


# =============================================================================
# Section 5b: Other SVM Kernels
# =============================================================================

## RBF
SVM_Model2=SVC(C=100, kernel='rbf', 
                           verbose=True, gamma="auto")

SVM_Model2.fit(X_train_vec, y_train)

print("SVM prediction:\n", SVM_Model2.predict(X_test_vec))
print("Actual:")
print(y_test)

SVM_matrix = confusion_matrix(y_test, SVM_Model2.predict(X_test_vec), labels=['negative', 'neutral', 'positive'])
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")

target_names = ['negative', 'neutral', 'positive']
print('SVM  Results - SVC RBF Kernel Results:\n', classification_report(y_test, SVM_Model2.predict(X_test_vec), target_names=target_names))



## POLY
SVM_Model3=SVC(C=100, kernel='poly',degree=3,
                           gamma="auto", verbose=True)

print(SVM_Model3)
SVM_Model3.fit(X_train_vec, y_train)

print("SVM prediction:\n", SVM_Model3.predict(X_test_vec))
print("Actual:")
print(y_test)

SVM_matrix = confusion_matrix(y_test, SVM_Model3.predict(X_test_vec), labels=['negative', 'neutral', 'positive'])
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")


target_names = ['negative', 'neutral', 'positive']
print('SVM Results - SVC Poly Kernel Results:\n', classification_report(y_test, SVM_Model3.predict(X_test_vec), target_names=target_names))


# =============================================================================
# Section 6: Naive Bayes Models
# =============================================================================

# Vectorize data
# MNB
unigram_count_vectorizer = CountVectorizer(encoding='latin-1', binary=False, min_df = 5, stop_words='english')
# Bernoulli
unigram_bool_vectorizer = CountVectorizer(encoding='latin-1', binary=True, min_df = 5, stop_words='english')
# tfidf
unigram_tfidf_vectorizer = TfidfVectorizer(encoding='latin-1', use_idf=True, min_df = 5, stop_words='english')


# Set classifiers
nb_clf = MultinomialNB()
b_clf = BernoulliNB()

# Fit vocabulary in training documents and transform the training documents into vectors
# MNB
X_train_vec = unigram_count_vectorizer.fit_transform(X_train)
# Bernoulli
X_train_vecB = unigram_bool_vectorizer.fit_transform(X_train)
# tfifd
X_train_vect = unigram_tfidf_vectorizer.fit_transform(X_train)

# Vectorize the test data
# MNB
X_test_vec = unigram_count_vectorizer.transform(X_test)
# Bernoulli
X_test_vecB = unigram_bool_vectorizer.transform(X_test)
# tfidf
X_test_vect = unigram_tfidf_vectorizer.transform(X_test)

# use the training data to train the models
# MNB
nb_clf.fit(X_train_vec,y_train)
# Bernoulli
b_clf.fit(X_train_vecB, y_train)
# tfidf
nb_clf.fit(X_train_vect, y_train)

# test the classifier on the test data set, print accuracy score
print('Multinomial Naive Bayes Accuracy Score is: ',nb_clf.score(X_test_vec,y_test))
print('Bernoulli Naive Bayes Accuracy Score is: ',b_clf.score(X_test_vecB,y_test))
print('Multinomial TFIDF Naive Bayes Accuracy Score is: ',nb_clf.score(X_test_vect,y_test))

from sklearn.metrics import confusion_matrix
# MNB
y_predMNB = nb_clf.fit(X_train_vec, y_train).predict(X_test_vec)
cmMNB = confusion_matrix(y_test,y_predMNB, labels = ['positive', 'neutral', 'negative'])
print('The Multinomial Naive Bayes Confusion Matrix is:\n ', cmMNB)

# Bernoulli
y_predB = b_clf.fit(X_train_vecB, y_train).predict(X_test_vecB)
cmB = confusion_matrix(y_test,y_predB, labels = ['positive', 'neutral', 'negative'])
print('The Bernoulli Naive Bayes Confusion Matrix is: \n ', cmB)

# tfidf
y_predT = nb_clf.fit(X_train_vect, y_train).predict(X_test_vect)
cmT = confusion_matrix(y_test,y_predT, labels = ['positive', 'neutral', 'negative'])
print('The Multinomial Naive Bayes TFIDF Confusion Matrix is:\n ', cmT)

# Classification report
from sklearn.metrics import classification_report
# MNB
target_names = ['positive', 'neutral', 'negative']
print('The Multinomial Naive Bayes Classification Report is: \n', classification_report(y_test, y_predMNB, target_names = target_names))

# Bernoulli
print('The Bernoulli Naive Bayes Classification Report is: \n', classification_report(y_test, y_predB, target_names = target_names))

# tfidf
print('The Multinomial Naive Bayes TFIDF Classification Report is: \n', classification_report(y_test, y_predT, target_names = target_names))


# =============================================================================
# Summarize Model Results
# =============================================================================
# put model results into a single table
import prettytable

pretty = prettytable.PrettyTable()
pretty.field_names = ['Model', 'Precision', 'Recall', 'Accuracy']  # To add the names of the columns or field name
pretty.add_row(['SVM - LinearSVC (Term Frequency)', '72%', '71%', '71%'])
pretty.add_row(['SVM (TF-IDF)', '76%', '75%', '75%'])
pretty.add_row(['SVM (RBF Kernel)', '73%', '73%', '73%'])
pretty.add_row(['SVM (Poly Kernel)', '39%', '63%', '63%'])
pretty.add_row(['MNB (Term Frequency)', '75%', '76%', '76%'])
pretty.add_row(['Bernoulli Naive Bayes', '77%', '77%', '77%'])
pretty.add_row(['MNB (TF-IDF)', '75%', '74%', '74%'])

pretty.align["Model"] = "l"

print("\n Model Results (Weighted Avg.):")
print(pretty)


# =============================================================================
# Section 7: LDA - Twitter API Data
# =============================================================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topic Modeling with Twitter API

@author: teresacameron

"""
##############################################################################
#   Get Tweets
##############################################################################

import tweepy as tw
import pandas as pd

# Twitter api data
consumer_key = 'yEjRX09e9g6CfIbXP4619iY5G'
consumer_secret = '7fGe03SvLFHVhSBswBbqRXBp7xSg3aYpyTW0cYO1tkK6yf423k'
access_token = '1034798053268631552-RxbEOAKx719zcm9BgTN8hB63gSKAV3'
access_token_secret = 'xLDu1sNN21nSIGqfmQwX25EhZCDOSCv7dXtSauBO7RZLw'

# Connect to Twitter API using the secrets
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# Delta
# Define the search term and the date as variables
search_words = '@Delta'

# Collect tweets
tweets = tw.Cursor(api.search, tweet_mode = 'extended',
              q=search_words,
              lang="en").items(250)

# Create list of Twitter
DeltaTweet_data = [[tweet.user.screen_name, tweet.full_text, tweet.created_at, tweet.user.location] for tweet in tweets]
DeltaTweetDF = pd.DataFrame(DeltaTweet_data, columns = ['UserName', 'Tweet', 'TimeDateStamp', 'Location'])
#print(DeltaTweetDF.Tweet)
DeltaTweetDF['Airline'] = 'Delta' 

# United
# Define the search term and the date as variables
search_words = '@United'

# Collect tweets
tweets = tw.Cursor(api.search, tweet_mode = 'extended',
              q=search_words,
              lang="en").items(250)

# Create list of Twitter data
UnitedTweet_data = [[tweet.user.screen_name, tweet.full_text, tweet.created_at, tweet.user.location] for tweet in tweets]
UnitedTweetDF = pd.DataFrame(UnitedTweet_data, columns = ['UserName', 'Tweet', 'TimeDateStamp', 'Location'])
#print(UnitedTweetDF.Tweet)
UnitedTweetDF['Airline'] = 'United'

# American Airlines
# Define the search term and the date as variables
search_words = '@AmericanAir'

# Collect tweets
tweets = tw.Cursor(api.search, tweet_mode = 'extended',
              q=search_words,
              lang="en").items(250)

# Create list of Twitter data
AATweet_data = [[tweet.user.screen_name, tweet.full_text, tweet.created_at, tweet.user.location] for tweet in tweets]
AATweetDF = pd.DataFrame(AATweet_data, columns = ['UserName', 'Tweet', 'TimeDateStamp', 'Location'])
#print(AATweetDF.Tweet)
AATweetDF['Airline'] = 'American Air'

# Southwest
# Define the search term and the date as variables
search_words = '@southwest'

# Collect tweets
tweets = tw.Cursor(api.search, tweet_mode = 'extended',
              q=search_words,
              lang="en").items(250)

# Create list of Twitter data for American Airlines
SWTweet_data = [[tweet.user.screen_name, tweet.full_text, tweet.created_at, tweet.user.location] for tweet in tweets]
SWTweetDF = pd.DataFrame(SWTweet_data, columns = ['UserName', 'Tweet', 'TimeDateStamp', 'Location'])
#print(SWTweetDF.Tweet) 
SWTweetDF['Airline'] = 'Southwest'

# US Airways
# Define the search term and the date as variables
search_words = '@USAirways'

# Collect tweets
tweets = tw.Cursor(api.search, tweet_mode = 'extended',
              q=search_words,
              lang="en").items(250)

# Create list of Twitter data
USATweet_data = [[tweet.user.screen_name, tweet.full_text, tweet.created_at, tweet.user.location] for tweet in tweets]
USATweetDF = pd.DataFrame(USATweet_data, columns = ['UserName', 'Tweet', 'TimeDateStamp', 'Location'])
#print(USATweetDF.Tweet) 
USATweetDF['Airline'] = 'US Airways'

# Virgin America
# Define the search term and the date as variables
search_words = '@VirginAmerica'

# Collect tweets
tweets = tw.Cursor(api.search, tweet_mode = 'extended',
              q=search_words,
              lang="en").items(250)

# Create list of Twitter data
VATweet_data = [[tweet.user.screen_name, tweet.full_text, tweet.created_at, tweet.user.location] for tweet in tweets]
VATweetDF = pd.DataFrame(VATweet_data, columns = ['UserName', 'Tweet', 'TimeDateStamp', 'Location'])
#print(VATweetDF.Tweet) 
VATweetDF['Airline'] = 'Virgin America'


# Combine dataframes of different airlines
AirTweets = pd.concat([DeltaTweetDF, UnitedTweetDF, AATweetDF, SWTweetDF, USATweetDF, 
                       VATweetDF], ignore_index=True, sort=False, axis=0)

###############################################################################
# Clean Tweets
###############################################################################

# Drop duplicates
AirTweets = AirTweets.drop_duplicates(subset = 'Tweet', keep = 'last')
# AirTweets.dtypes

#conda install -c anaconda gensim

from nltk.corpus import stopwords
en_stop_words = set(stopwords.words('english'))

from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk
nltk.download('stopwords')
import pandas as pd
import re
import math
import matplotlib.pyplot as plt

# Write function to clean tweets and create dataframe with original tweet,
# preprocessed Tweet, and tokenized tweet
# Source: https://towardsdatascience.com/topic-modeling-with-latent-dirichlet-allocation-by-example-3b22cd10c835

def clean_tweets(df=AirTweets, 
                 tweet_col='Tweet'):
    
    df_copy = df.copy()
    
    # drop rows with empty values
    df_copy.dropna(inplace=True)
    
    # lower the tweets
    df_copy['preprocessed_' + tweet_col] = df_copy[tweet_col].str.lower()
    
    # filter out stop words and URLs
    en_stop_words = set(stopwords.words('english'))
    extended_stop_words = en_stop_words | \
                        {
                            '&amp;', 'rt',                           
                            'th','co', 're', 've', 'kim', 'daca', 'rt' #, '@delta', '@americanair', 'flight', '@southwest', '@southwestair', 'airlines', '@united', 'united', 'delta', '@virginamerica'
                        }
    url_re = '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'        
    df_copy['preprocessed_' + tweet_col] = df_copy['preprocessed_' + tweet_col].apply(lambda row: ' '.join([word for word in row.split() if (not word in extended_stop_words) and (not re.match(url_re, word))]))
    
    # tokenize the tweets
    tokenizer = RegexpTokenizer('[a-zA-Z]\w+\'?\w*')
    df_copy['tokenized_' + tweet_col] = df_copy['preprocessed_' + tweet_col].apply(lambda row: tokenizer.tokenize(row))
    
    return df_copy

# Clean tweets
air_tweets_clean1 = clean_tweets(AirTweets)
air_tweets_clean1.head()

# Save as csv file
air_tweets_clean1.to_csv(r'/Users/teresacameron/Documents/House/MADS/Text_Mining/Project/AirTweets1.csv', index = False, header = True)

# Create dataframe of tweets without hashtags

def clean_tweets(df=AirTweets, 
                 tweet_col='Tweet'):
    
    df_copy = df.copy()
    
    # drop rows with empty values
    df_copy.dropna(inplace=True)
    
    # lower the tweets
    df_copy['preprocessed_' + tweet_col] = df_copy[tweet_col].str.lower()
    
    # filter out stop words and URLs
    en_stop_words = set(stopwords.words('english'))
    extended_stop_words = en_stop_words | \
                        {
                            '&amp;', 'rt',                           
                            'th','co', 're', 've', 'kim', 'daca', 'rt', '@delta', '@americanair', 'flight', '@southwest', '@southwestair', 'airlines', '@united', 'united', 'delta', '@virginamerica'
                        }
    url_re = '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'        
    df_copy['preprocessed_' + tweet_col] = df_copy['preprocessed_' + tweet_col].apply(lambda row: ' '.join([word for word in row.split() if (not word in extended_stop_words) and (not re.match(url_re, word))]))
    
    # tokenize the tweets
    tokenizer = RegexpTokenizer('[a-zA-Z]\w+\'?\w*')
    df_copy['tokenized_' + tweet_col] = df_copy['preprocessed_' + tweet_col].apply(lambda row: tokenizer.tokenize(row))
    
    return df_copy

# Clean tweets
air_tweets_clean2 = clean_tweets(AirTweets)
air_tweets_clean2.head()

# Save as csv file
air_tweets_clean2.to_csv(r'/Users/teresacameron/Documents/House/MADS/Text_Mining/Project/AirTweets2.csv', index = False, header = True)


###############################################################################
#Analyze tweets
###############################################################################

# Write function to get most frequent tweet
# With hashtags
def get_most_freq_words(str, n=None):
    vect = CountVectorizer().fit(str)
    bag_of_words = vect.transform(str)
    sum_words = bag_of_words.sum(axis=0) 
    freq = [(word, sum_words[0, idx]) for word, idx in vect.vocabulary_.items()]
    freq =sorted(freq, key = lambda x: x[1], reverse=True)
    return freq[:n]
  
most_freq1 = get_most_freq_words([ word for tweet in air_tweets_clean1.tokenized_Tweet for word in tweet],25)

# Without hashtags 
most_freq2 = get_most_freq_words([ word for tweet in air_tweets_clean2.tokenized_Tweet for word in tweet],25)

# Create dataframes
most_freq1DF = pd.DataFrame(most_freq1, columns = ['words', 'count'])
most_freq2DF = pd.DataFrame(most_freq2, columns = ['words', 'count'])

# Create horizontal bar graph of 25 most frequently used words
# With hashtags
fig, ax = plt.subplots(figsize=(8,8))
most_freq1DF.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color='orange')
ax.set_title('Top Word Frequency in Tweets with Hashtags')
plt.show()

# Without hashtags
ig, ax = plt.subplots(figsize=(8,8))
most_freq2DF.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color='navy')
ax.set_title('Top Word Frequency in Tweets without Hashtags')
plt.show()

# LDA
# Build a dictionary where for each tweet, each word has its own id.
# With hashtags
tweets_dictionary1 = Dictionary(air_tweets_clean1.tokenized_Tweet)
# Without hashtags
tweets_dictionary2 = Dictionary(air_tweets_clean2.tokenized_Tweet)

# build the corpus i.e. vectors with the number of occurence of each word per tweet
tweets_corpus1 = [tweets_dictionary1.doc2bow(tweet) for tweet in air_tweets_clean1.tokenized_Tweet]
tweets_corpus2 = [tweets_dictionary2.doc2bow(tweet) for tweet in air_tweets_clean2.tokenized_Tweet]

# compute coherence to visualize best number of topics
# With hashtags
tweets_coherence = []
for nb_topics in range(1,36):
    lda = LdaModel(tweets_corpus1, num_topics = nb_topics, id2word = tweets_dictionary1, passes=10)
    cohm = CoherenceModel(model=lda, corpus=tweets_corpus1, dictionary=tweets_dictionary1, coherence='u_mass')
    coh = cohm.get_coherence()
    tweets_coherence.append(coh)

# visualize coherence
plt.figure(figsize=(10,5))
plt.plot(range(1,36),tweets_coherence)
plt.title('Topic Coherence: Determining Optimal Number of Topic with Hashtags')
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score");

# Without hashtags
tweets_coherence = []
for nb_topics in range(1,36):
    lda = LdaModel(tweets_corpus2, num_topics = nb_topics, id2word = tweets_dictionary2, passes=10)
    cohm = CoherenceModel(model=lda, corpus=tweets_corpus2, dictionary=tweets_dictionary2, coherence='u_mass')
    coh = cohm.get_coherence()
    tweets_coherence.append(coh)

# visualize coherence
plt.figure(figsize=(10,5))
plt.plot(range(1,36),tweets_coherence)
plt.title('Topic Coherence: Determining Optimal Number of Topic without Hashtags')
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score");

# Run LDA
# With hashtags
k=3
tweets_lda1 = LdaModel(tweets_corpus1, num_topics = k, id2word = tweets_dictionary1, passes=10)

import matplotlib.gridspec as gridspec
def plot_top_words1(lda=tweets_lda1, nb_topics=k, nb_words=10):
    top_words = [[word for word,_ in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]
    top_betas = [[beta for _,beta in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]

    gs  = gridspec.GridSpec(round(math.sqrt(k))+1,round(math.sqrt(k))+1)
    gs.update(wspace=0.5, hspace=0.5)
    plt.figure(figsize=(20,15))
    for i in range(nb_topics):
        ax = plt.subplot(gs[i])
        plt.barh(range(nb_words), top_betas[i][:nb_words], align='center',color='orange', ecolor='black')
        ax.invert_yaxis()
        ax.set_yticks(range(nb_words))
        ax.set_yticklabels(top_words[i][:nb_words])
        plt.title("Topic "+str(i))
        
  
plot_top_words1()

# Without hashtags
k=4
tweets_lda2 = LdaModel(tweets_corpus2, num_topics = k, id2word = tweets_dictionary2, passes=10)

def plot_top_words2(lda=tweets_lda2, nb_topics=k, nb_words=10):
    top_words = [[word for word,_ in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]
    top_betas = [[beta for _,beta in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]

    gs  = gridspec.GridSpec(round(math.sqrt(k))+1,round(math.sqrt(k))+1)
    gs.update(wspace=0.5, hspace=0.5)
    plt.figure(figsize=(20,15))
    for i in range(nb_topics):
        ax = plt.subplot(gs[i])
        plt.barh(range(nb_words), top_betas[i][:nb_words], align='center',color='navy', ecolor='black')
        ax.invert_yaxis()
        ax.set_yticks(range(nb_words))
        ax.set_yticklabels(top_words[i][:nb_words])
        plt.title("Topic "+str(i))
  
plot_top_words2()


# =============================================================================
# IST 736 Final Project - End of Code
# =============================================================================