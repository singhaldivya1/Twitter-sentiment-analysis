# Twitter-sentiment-analysis

An example of sentiment analysis on Twitter using positive(+1) , negative(-1) and neutral(0) opinions as labels to analyse the tweets.

To complete the analysis exploits few python libraries:

- Panda
- NLTK
- Scikit-learn


We first convert the tweets into data frames using panda.

Pre Processing

The preprocessing scripts modifies the tweets content in order to make possible the further analysis.

Any url,twitter handles,Html tags are removed
Any additional white spaces is removed
Any unnecessary repeated letters are removed for eg. "I am Happyyyyyy" is made into "I am Happy".

Feature Extractor

Before building a classifier we need to extract all the features (word) contained in the tweet text.

Moreover, is necessary to remove any stop words. We also strip out punctuation, digits, and any symbols that might be still in the tweet text.

A list of the most common English stopwords is imported from NLTK corpus and used for stopword removal.

Stemming is done using Porter Stemmer Algorithm


Classifier

Multinmial Naive Bayesian Classifier.

The analysis follows the following steps:

The tweets dataset are divided into a train dataset, which will be used to train the classifier, and a test dataset, used to test it. 10- fold cross validiation strategy is used to divide the data.

From each tweet will be extracted a feature-set

We use NLTK to describe each tweets in terms of the features it contains. Indeed, we create a list of words ordered by frequency.

We trains a NaiveBayesClassifier with such a dataset

We test the classifier using the test data and we process in the same way of the training dataset.

Results

Here the results of the analysis for the example:

Accuracy Score: 53.6830561141%

Precision of positive class: 55.481526

Recall of positive class: 55.414398

F1-Score of positive class: 55.447942 

Precision of negative class: 55.032619

Recall of negative class: 61.446410

F1-Score of negative class: 58.062930 
