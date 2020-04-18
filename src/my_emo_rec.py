"""
@author: Danyal

The following code classifies piece of music as one of 
the four emotions mentioned in the document
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import csv
# from sklearn.cross_validation import train_test_split

data = pd.read_csv('../My_features.csv')
feature = data.ix[:, 'tempo':]
featureName = list(feature)
color = ['red' if l==1 else 'green' if l==2 else 'blue' if l==3 else 'orange' for l in data['label']]

for name in featureName:
    feature[name] = (feature[name]-feature[name].min())/(feature[name].max()-feature[name].min())

plt.style.use('ggplot')

array = np.array(data)

features = feature.values
labels = data.ix[:, 'class'].dropna()
# test_size = 0.20
# random_seed = 7

# train_d, test_d, train_l, test_l = train_test_split(features, labels, test_size=test_size, random_state=random_seed)

train_d, train_l = features, labels


emot_list = ["Joy", "Fear", "Anger", "Sadness"]
fields = [[],[]]
for emo in emot_list:
	print(emo)
	data = pd.read_csv('../Test_Features/' + emo + '_features.csv')
	name_list = data.ix[:, 'song_name']
	feature = data.ix[:, 'tempo':]
	featureName = list(feature)
	color = ['red' if l==1 else 'green' if l==2 else 'blue' if l==3 else 'orange' for l in data['label']]

	for name in featureName:
	    feature[name] = (feature[name]-feature[name].min())/(feature[name].max()-feature[name].min())

	plt.style.use('ggplot')

	array = np.array(data)

	test_d = feature.values
	test_l = data.ix[:, 'class'].dropna()

	result = []
	xlabel = [i for i in range(1, 11)]
	# for neighbors in range(1, 11):
	# kNN = KNeighborsClassifier(n_neighbors=6)
	kNN = svm.SVC()
	# kNN = svm.LinearSVC()
	# kNN = RandomForestClassifier()
	# kNN = AdaBoostClassifier()
	kNN.fit(train_d, train_l)
	prediction = kNN.predict(test_d)
	for i,name in enumerate(name_list):
		fields[0].append(name_list[i])
		fields[1].append(prediction[i])
	print(prediction)
	print(name_list)
	# fields[0] = fields[0] + name_list
	# fields[1] = fields[1] + prediction
	print("Accuracy : ", accuracy_score(prediction, test_l)*100)

with open('Infer.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)  
    csvwriter.writerows(fields)  
    # csvwriter.writerows(rows)
	# plt.figure(figsize=(10, 10))
	# plt.xlabel('kNN Neighbors for k=1,2...10')
	# plt.ylabel('Accuracy Score')
	# plt.title('kNN Classifier Results')
	# plt.ylim(0, 100)
	# plt.xlim(0, xlabel[len(xlabel)-1]+1)
	# plt.plot(xlabel, result)
	# plt.savefig('1-fold 10NN Result.png')
	# plt.show()