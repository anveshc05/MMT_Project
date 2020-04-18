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
from sklearn.decomposition import PCA

# from sklearn.cross_validation import train_test_split

# data = pd.read_csv('../My_features.csv')
data = pd.read_csv('../Emotion_features.csv')
col_list = []
for col in data.columns:
    col_list.append(col)
print(col_list)
# print(data.columns)
feature = data.ix[:, 'tempo':]
featureName = list(feature)
color = ['red' if l==1 else 'green' if l==2 else 'blue' if l==3 else 'orange' for l in data['label']]

for name in featureName:
    feature[name] = (feature[name]-feature[name].min())/(feature[name].max()-feature[name].min())

plt.style.use('ggplot')

array = np.array(data)

features = feature.values

pca = PCA(n_components = 2)
pca.fit(features)
out_feat = pca.transform(features)
x = [[],[],[],[]]
y = [[],[],[],[]]
labels = data.ix[:, 'class'].dropna()
for i in range(len(labels)):
    if labels[i] == "happy":
        x[0].append(out_feat[i][0])
        y[0].append(out_feat[i][1])
    if labels[i] == "sad":
        x[1].append(out_feat[i][0])
        y[1].append(out_feat[i][1])
    if labels[i] == "fear":
        x[2].append(out_feat[i][0])
        y[2].append(out_feat[i][1])
    if labels[i] == "angry":
        x[3].append(out_feat[i][0])
        y[3].append(out_feat[i][1])
plt.scatter(x[0], y[0], c='magenta')
plt.scatter(x[1], y[1], c='b')
plt.scatter(x[2], y[2], c='gray')
plt.scatter(x[3], y[3], c='r')
test_size = 0.10
random_seed = 7

train_d, test_d, train_l, test_l = train_test_split(features, labels, test_size=test_size, random_state=random_seed)

result = []
xlabel = [i for i in range(1, 11)]
for neighbors in range(1, 11):
    pca = PCA(n_components = 7)
    pca.fit(train_d)
    train_d = pca.transform(train_d)
    test_d = pca.transform(test_d)
    # kNN = KNeighborsClassifier(n_neighbors=neighbors)
    # kNN = svm.SVC()
    # kNN = svm.SVC()
    kNN = RandomForestClassifier()
    kNN.fit(train_d, train_l)
    prediction = kNN.predict(test_d)
    result.append(accuracy_score(prediction, test_l)*100)

plt.figure(figsize=(10, 10))
plt.xlabel('kNN Neighbors for k=1,2...10')
plt.ylabel('Accuracy Score')
plt.title('kNN Classifier Results')
plt.ylim(0, 100)
plt.xlim(0, xlabel[len(xlabel)-1]+1)
plt.plot(xlabel, result)
plt.savefig('1-fold 10NN Result.png')
plt.show()