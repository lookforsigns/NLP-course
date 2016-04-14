import collections
import itertools
import numpy
import scipy
import pickle
from pprint import pprint
import json

## language package
import nltk
import nltk.data
from nltk.classify import NaiveBayesClassifier
#import nltk.sentiment
#from nltk.sentiment import SentimentAnalyzer
#from nltk.sentiment.util import *

## machine learning
from sklearn import svm, grid_search, cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

## init

wnLemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()
multinomial_classifier = MultinomialNB()

## vars
topic_category_index = {1:"autot",2:"terveys",3:"julkkikset",4:"tunteet"}
emotion_index = {1:"Sadness",2:"Love",3:"Fear",4:"Certainty",5:"Other",0:"None"}
docs_new = ["He hates me :(","Martina had hulluuuskohtaus today","today I fear a hero", "could toyota be good toudellakaan","perhaps we say good night","I love him","Today husband made me VHH pizza"]

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

## subprograms

def preprocess( text ):

    #tokens = nltk.word_tokenize( text )
    #tokens = [wnLemmatizer.lemmatize(x).lower() for x in tokens]

    #turn the text content into numerical feature vectors
    #bag of words representation
    # assign a fixed integer id to each word occurring in any document of the set (eg. build a dictionary)
    # for each document #i count the occurrences of each word w and store it in X[i,j] as
    # value of feature #j where j is the index of word w in the dictionary
    # n_features is the # of distinct words in the corpus
    # all tokens including non-characters should be coded as NUM
    # all words not in dictionary might be later coded as FINNISH

    X_train_counts = count_vect.fit_transform( text )
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    return X_train_tfidf

def trainingCategorySample( data, labels, testList ):
    data_train, data_test, labels_train, labels_test = cross_validation.train_test_split( data, labels, test_size = .25 )
    classifier = multinomial_classifier.fit(data_train, labels_train)

    X_new_counts = count_vect.transform( testList )
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = classifier.predict(X_new_tfidf)

    return predicted

def trainingCategory( data, labels):
    data_train, data_test, labels_train, labels_test = cross_validation.train_test_split( data, labels, test_size = .25 )
    
    text_clf = Pipeline([("vect",count_vect),("tfidf",tfidf_transformer),("clf",multinomial_classifier)])
    text_clf = text_clf.fit(data_train,labels_train)

    docs_test = data_test
    predictedTest = text_clf.predict(docs_test)
    outcome = numpy.mean(predictedTest == labels_test)

    return outcome

## third round (not working yet)
def learn( data, labels ):

    estimator = svm.SVC()
    grid = [
        {'C': numpy.arange( 0.5 , 10, .5 ), 'gamma': numpy.arange( .0001, .1, .0005) , 'kernel': ['rbf', 'sigmoid'] },
    ]

    model = grid_search.GridSearchCV( estimator , grid, cv = 10, verbose = 5 )

    data = numpy.array( data )
    labels = numpy.array( labels )

    ## separate train and test
    data_train, data_test, labels_train, labels_test = cross_validation.train_test_split( data, labels, test_size = .25 )

    model.fit( data, labels )

    pickle.dump( model, open('model.svm', 'w') )

    print("Test result")
    print (model.score( data_train, labels_train ))
    print ("")
    print ("Test result")
    print (model.score( data_test, labels_test ))

def predict( textline ):

    model = pickle.load( open('alpha/model.svm') )
    data = numpy.array( preprocess( textline ) )
    return model.predict( [ data ] )


## main

if __name__ == "__main__":

    ## teach with data

    def _int(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    ## first round

    d = json.load(open( 'D://dataForLearning.json' ))
    d = d[1:] #remove first row that is column labels
    d = [x for x in d if x["text"] is not "" ] # include rows where text cell is not empty
    topics = [int(x["topic"]) for x in d ] # int topic values (topic categories)

    label1 = [int(x["Label1"]) if x["Label1"] is not "" else 0 for x in d ] # int label values where Label is not empty, otherwise make them 0
    label2 = [int(x["Label2"]) if x["Label2"] is not "" else 0 for x in d ] # int label values where Label is not empty, otherwise make them 0
    label3 = [int(x["Label3"]) if x["Label3"] is not "" else 0 for x in d ] # int label values where Label is not empty, otherwise make them 0
    label4 = [int(x["Label4"]) if x["Label4"] is not "" else 0 for x in d ] # int label values where Label is not empty, otherwise make them 0
    label5 = [int(x["Label5"]) if x["Label5"] is not "" else 0 for x in d ] # int label values where Label is not empty, otherwise make them 0

    data = [( x['text'] ) for x in d]
    print("Data rows after first cleaning: "+str(len(data)))
    print("Topic rows after first cleaning: "+str(len(topics)))
    print("Emotion labels after first cleaning: "+str(len(label1)))

    ## predict discussion topic - sample sentences then whole data performance

    tfidf = preprocess(data)
    predictionSample = trainingCategorySample(tfidf,topics,docs_new)
    print("\nSample set topic category prediction performance:")
    for doc,item in zip(docs_new,predictionSample):
        print("%r => %s" % (doc, topic_category_index[item]))
    print("Prediction quality: "+str(trainingCategory(data,topics)))

    ## predict emotion labels - sample sentences then whole data performance

    data = [( x['text'] ) for x in d]
    tfidf = preprocess(data)
    
    predictionSampleEmotion = trainingCategorySample(tfidf,label1,docs_new)
    print("\nSample set emotion category prediction performance for Label1")
    for doc,item in zip(docs_new,predictionSampleEmotion):
        print("%r => %s" % (doc, emotion_index[item]))
    print("Prediction quality: "+str(trainingCategory(data,label1)))
    
    data = [( x['text'] ) for x in d]
    tfidf = preprocess(data)
    
    predictionSampleEmotion = trainingCategorySample(tfidf,label2,docs_new)
    print("\nSample set emotion category prediction performance for Label2")
    for doc,item in zip(docs_new,predictionSampleEmotion):
        print("%r => %s" % (doc, emotion_index[item]))
    print("Prediction quality: "+str(trainingCategory(data,label2)))

    data = [( x['text'] ) for x in d]
    tfidf = preprocess(data)
    
    predictionSampleEmotion = trainingCategorySample(tfidf,label3,docs_new)
    print("\nSample set emotion category prediction performance for Label3")
    for doc,item in zip(docs_new,predictionSampleEmotion):
        print("%r => %s" % (doc, emotion_index[item]))
    print("Prediction quality: "+str(trainingCategory(data,label3)))

    data = [( x['text'] ) for x in d]
    tfidf = preprocess(data)
    
    predictionSampleEmotion = trainingCategorySample(tfidf,label4,docs_new)
    print("\nSample set emotion category prediction performance for Label4")
    for doc,item in zip(docs_new,predictionSampleEmotion):
        print("%r => %s" % (doc, emotion_index[item]))
    print("Prediction quality: "+str(trainingCategory(data,label4)))

    data = [( x['text'] ) for x in d]
    tfidf = preprocess(data)
    
    predictionSampleEmotion = trainingCategorySample(tfidf,label5,docs_new)
    print("\nSample set emotion category prediction performance for Label5")
    for doc,item in zip(docs_new,predictionSampleEmotion):
        print("%r => %s" % (doc, emotion_index[item]))
    print("Prediction quality: "+str(trainingCategory(data,label5)))

    ## second round

    d = json.load(open( 'D://dataForLearning.json' ))
    d = d[1:] #remove first row that is column labels
    d = [x for x in d if x["text"] is not "" ] # include rows where text cell is not empty

    # partial data: include only rows with emotion tag
    d = [x for x in d if x["Label1"] is not "" or x["Label2"] is not "" or x["Label3"] is not "" or x["Label4"] is not "" or x["Label5"] is not ""] 
    ##d = [x for x in d if x["Label1"] is not ""] # testing: include rows with sample emotion tag only
    topics = [int(x["topic"]) for x in d ] # int topic values (topic categories)

    label1 = [int(x["Label1"]) if x["Label1"] is not "" else 0 for x in d ] # int label values where Label is not empty, otherwise make them 0
    label2 = [int(x["Label2"]) if x["Label2"] is not "" else 0 for x in d ] # int label values where Label is not empty, otherwise make them 0
    label3 = [int(x["Label3"]) if x["Label3"] is not "" else 0 for x in d ] # int label values where Label is not empty, otherwise make them 0
    label4 = [int(x["Label4"]) if x["Label4"] is not "" else 0 for x in d ] # int label values where Label is not empty, otherwise make them 0
    label5 = [int(x["Label5"]) if x["Label5"] is not "" else 0 for x in d ] # int label values where Label is not empty, otherwise make them 0

    data = [( x['text'] ) for x in d]
    print("\nData rows after second cleaning: "+str(len(data)))
    print("Topic rows after second cleaning: "+str(len(topics)))
    print("Emotion labels after second cleaning: "+str(len(label1)))

    ## predict topic labels - sample sentences then partial data performance
 
    tfidf = preprocess(data)
    predictionSample = trainingCategorySample(tfidf,topics,docs_new)
    print("\nSample set topic category prediction performance:")
    for doc,item in zip(docs_new,predictionSample):
        print("%r => %s" % (doc, topic_category_index[item]))
    print("Prediction quality: "+str(trainingCategory(data,topics)))

    ## predict emotion labels - sample sentences then partial data performance

    data = [( x['text'] ) for x in d]
    tfidf = preprocess(data)
    
    predictionSampleEmotion = trainingCategorySample(tfidf,label1,docs_new)
    print("\nSample set emotion category prediction performance for Label1")
    for doc,item in zip(docs_new,predictionSampleEmotion):
        print("%r => %s" % (doc, emotion_index[item]))
    print("Prediction quality: "+str(trainingCategory(data,label1)))
    
    data = [( x['text'] ) for x in d]
    tfidf = preprocess(data)
    
    predictionSampleEmotion = trainingCategorySample(tfidf,label2,docs_new)
    print("\nSample set emotion category prediction performance for Label2")
    for doc,item in zip(docs_new,predictionSampleEmotion):
        print("%r => %s" % (doc, emotion_index[item]))
    print("Prediction quality: "+str(trainingCategory(data,label2)))

    data = [( x['text'] ) for x in d]
    tfidf = preprocess(data)
    
    predictionSampleEmotion = trainingCategorySample(tfidf,label3,docs_new)
    print("\nSample set emotion category prediction performance for Label3")
    for doc,item in zip(docs_new,predictionSampleEmotion):
        print("%r => %s" % (doc, emotion_index[item]))
    print("Prediction quality: "+str(trainingCategory(data,label3)))

    data = [( x['text'] ) for x in d]
    tfidf = preprocess(data)
    
    predictionSampleEmotion = trainingCategorySample(tfidf,label4,docs_new)
    print("\nSample set emotion category prediction performance for Label4")
    for doc,item in zip(docs_new,predictionSampleEmotion):
        print("%r => %s" % (doc, emotion_index[item]))
    print("Prediction quality: "+str(trainingCategory(data,label4)))

    data = [( x['text'] ) for x in d]
    tfidf = preprocess(data)
    
    predictionSampleEmotion = trainingCategorySample(tfidf,label5,docs_new)
    print("\nSample set emotion category prediction performance for Label5")
    for doc,item in zip(docs_new,predictionSampleEmotion):
        print("%r => %s" % (doc, emotion_index[item]))
    print("Prediction quality: "+str(trainingCategory(data,label5)))

    ## third round - get nltk.sentimentAnalyzer to work

    #data = [preprocess( x['text'] ) for x in d] # nltk way of doing it - preprocess text column: lemmatize all row-sentences and return as list
