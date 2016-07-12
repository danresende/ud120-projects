#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn.cross_validation import train_test_split
from sklearn import tree, metrics

features_train, features_test, labels_train, labels_test = train_test_split(
	features, labels, test_size = 0.3, random_state = 42)

clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)

predictions = clf.predict(features_test)

print 'Precision:', metrics.precision_score(labels_test, predictions)
print 'Recall:', metrics.precision_score(labels_test, predictions)
print 'Accuracy:', clf.score(features_test, labels_test)

joseph = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0

for i in range(len(true_labels)):
	if joseph[i] == 1 and true_labels[i] == 1:
		true_pos += 1
	elif joseph[i] == 0 and true_labels[i] == 0:
		true_neg += 1
	elif joseph[i] == 1 and true_labels[i] == 0:
		false_pos += 1
	elif joseph[i] == 0 and true_labels[i] == 1:
		false_neg += 1

precision = (float(true_pos) / float(true_pos + false_pos))
recall = (float(true_pos) / float(true_pos + false_neg))

print 'true_pos: %i' % true_pos
print 'true_neg: %i' % true_neg
print 'false_pos: %i' % false_pos
print 'false_neg: %i' % false_neg
print 'precision: %f' % precision
print 'recall: %f' % recall
print 'F1: %f' % ((precision * recall) / (precision + recall))
