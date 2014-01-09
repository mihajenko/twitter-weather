#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Miha Jenko'
__copyright__ = "Copyright 2013, Miha Jenko"
__credits__ = ["Christopher Potts"]
__license__ = "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License: http://creativecommons.org/licenses/by-nc-sa/3.0/"

###############################  INSTRUCTIONS  ###############################
# Check comments as they may contain additional instructions.
# Set train data, test data, output file path strings.
# Get data from: https://www.kaggle.com/c/crowdflower-weather-twitter/data
traindatafp = 'train.csv'
testdatafp = 'test.csv'
outputdatafp = 'prediction.csv'
# Change Ridge's alpha parameter at your own discretion (use CV!)
##############################################################################

import time
# data handlers
import pandas as pd
import numpy as np
# model saver
# from sklearn.externals import joblib # uncomment if you wish to save the prediction models to a file in working dir
# vectorizers, regressors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import Normalizer


# for pretty printing
def tweetprint(rowindex, finalmat):
	print rowindex
	row = iter(finalmat)
	i = 0
	s = ""
	while i < len(rowindex):
		if rowindex[i].isupper():
			cell = "%00.1f" % (row.next() * 100) + "%"
			s += cell
			if len(cell) == 4:
				i += 4
			else:
				i += 5

		else:
			s += " "
			i += 1
	print s

# start of user input
option = int(raw_input("Predict confidence score on: [1] Kaggle data  [2] I will input my own tweets: "))

print "Getting data ..."
start = time.time()
td1 = pd.read_csv(traindatafp, encoding='utf-8')
td1size = len(td1)
if option is 1:
	td2 = pd.read_csv(testdatafp, encoding='utf-8')
	td2size = len(td2)
print "Completed in " + str(time.time() - start) + "s \n"

print "TF-IDF & fitting ..."
start = time.time()
tfidf = TfidfVectorizer(max_features=None, ngram_range=(1, 3), analyzer='word',
                        token_pattern=r'''(?:[<>]?[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]|[\)\]\(\[dDpP/\:\}\{@\|\\][\-o\*\']?[:;=8][<>]?)|(?:(?:\+?[01][\-\s.]*)?(?:[\(]?\d{3}[\-\s.\)]*)?\d{3}[\-\s.]*\d{4})|(?:@[\w_]+)|(?:\#+[\w_]+[\w\'_\-]*[\w_]+)|(?:[a-z][a-z'\-_]+[a-z])|(?:[+\-]?\d+[,/.:-]\d+[+\-]?)|(?:[\w_]+)|(?:\.(?:\s*\.){1,})|(?:\S)''',
                        norm='l2', use_idf=True, smooth_idf=True, dtype=np.float)
tfidf.fit(td1['tweet'])
# joblib.dump(tfidf,'tfidf.pkl',compress=9) # uncomment if you wish to save the tf-idf model

traindata = tfidf.transform(td1['tweet'])
clf1 = Ridge(alpha=0.421)  # determine alpha by playing around with grid search and cross-validation
clf1.fit(traindata, td1.iloc[:, 4:9])
# joblib.dump(clf1,'clfs.pkl',compress=9)  # uncomment if you wish to save the "sentiment" model
clf2 = Ridge(alpha=0.8)  # determine alpha by playing around with grid search and cross-validation
clf2.fit(traindata, td1.iloc[:, 9:13])
# joblib.dump(clf2,'clfw.pkl',compress=9) # uncomment if you wish to save the "when" model
clf3 = Ridge(alpha=0.613)  # determine alpha by playing around with grid search and cross-validation
clf3.fit(traindata, td1.iloc[:, 13:28])
# joblib.dump(clf3,'clfk.pkl',compress=9)  # uncomment if you wish to save the "kind" model
print "Completed in " + str(time.time() - start) + "s \n"

if option is 2:
	tweets = []
	while 1:
		tweet = raw_input("Tweet, please (q for quit): ")
		if tweet in "q":
			break
		else:
			tweets.append(tweet)
	testdata = tfidf.transform(np.array(tweets))
	td2size = len(tweets)
else:
	testdata = tfidf.transform(td2['tweet'])

print "Classifying ..."
start = time.time()

print "S ..."
smodel = clf1.predict(X=testdata)

print "W ..."
wmodel = clf2.predict(X=testdata)

print "K ..."
kmodel = clf3.predict(X=testdata)
print "Completed in " + str(time.time() - start) + "s \n"

print "Clipping and L1 normalization ..."
smodel = np.clip(smodel, 0, 1)
wmodel = np.clip(wmodel, 0, 1)
kmodel = np.clip(kmodel, 0, 1)
smodel = Normalizer(norm='l1').transform(smodel)
wmodel = Normalizer(norm='l1').transform(wmodel)
# no need for normalizing the K labels model, as each label is independent
print "Completed in " + str(time.time() - start) + " s\n"


# set output filename
if option is 1:
	print "Finishing ..."
	start = time.time()
	twid = np.matrix(td2['id']).transpose()
	finalmat = np.concatenate((twid, smodel, wmodel, kmodel), axis=1)
	col = '%i,' + '%f,' * 9 + '%f,' * 14 + '%f'
	np.savetxt(outputdatafp, finalmat, col, delimiter=",")
	with file(outputdatafp, 'r') as original:
		data = original.read()
	original.close()
	with file(outputdatafp, 'w') as modified:
		modified.write("id,s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15\n" + data)
	modified.close()
	print "Completed in " + str(time.time() - start) + "s \n"
else:
	np.set_printoptions(precision=2)
	finalmat = np.concatenate((smodel, wmodel, kmodel), axis=1)
	for j in xrange(td2size):
		print "\n\nTWEET:  " + tweets[j] + "\n"
		print "Sentiment"
		rowindex = "I can't tell | Negative | Neutral / author is just sharing information | Positive | Tweet not related to weather condition"
		tweetprint(rowindex, finalmat[j, 0:5])

		print "\nWhen?"
		rowindex = "Current (same day) weather | Future (forecast) | I can't tell | Past weather"
		tweetprint(rowindex, finalmat[j, 5:9])

		print "\nWhat kind?"
		rowindex = "Clouds | Cold | Dry | Hot | Humid | Hurricane | I can't tell | Ice | Other | Rain | Snow | Storms | Sun | Tornado | Wind"
		tweetprint(rowindex, finalmat[j, 9:])