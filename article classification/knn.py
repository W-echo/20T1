## Model: MNB
## Feature matrix: TFIDF
## Author: Xiaoyu Wang and Naihao Liu
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Create text
data = pd.read_csv("training.csv")

text_data = np.array(data)

# Create bag of words
#count = TfidfVectorizer()
count = TfidfVectorizer(ngram_range = (1,2))
TFIDF = count.fit_transform(text_data[:, 1])
# Create feature matrix
x = TFIDF
kf = KFold(n_splits=10)

# Create target vector
target_dict = {'ARTS CULTURE ENTERTAINMENT': 0, 'BIOGRAPHIES PERSONALITIES PEOPLE': 1, 'DEFENCE': 2,
               'DOMESTIC MARKETS': 3, 'FOREX MARKETS': 4, 'HEALTH': 5, 'MONEY MARKETS': 6,
               'SCIENCE AND TECHNOLOGY': 7, 'SHARE LISTINGS': 8, 'SPORTS': 9, 'IRRELEVANT': 10}
y = []
for i in text_data[:, -1]:
    y.append(target_dict[i])
y = np.array(y)
clf = KNeighborsClassifier(n_neighbors=6, p=2)


final_result = []
model_dic = {}
sel = SelectKBest(f_classif, k=507)
sel.fit(x, y)
x = sel.transform(x)

for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = clf.fit(x_train, y_train)
    predicted_y = model.predict(x_test)
    model_dic[accuracy_score(y_test, predicted_y)] = (x_train, y_train)

for i in model_dic.keys():
    if i == max(model_dic.keys()):
        print(f'train data precision is {i}')
        x_train, y_train = model_dic[i]
        model = clf.fit(x_train, y_train)
        break

def get_key(dict, value):
    for k, v in dict.items():
        if v == value:
            return k



test = pd.read_csv("test.csv")
test_data = np.array(test)
y_test = []
for i in test_data[:, -1]:
    y_test.append(target_dict[i])
x_test = count.transform(test_data[:, 1])
x_test = sel.transform(x_test)
y_test = np.array(y_test)

model = clf.fit(x, y)
predicted_y = model.predict(x_test)
proba = clf.predict_proba(x_test)
proba = np.array(proba)
for i in range(0,10):
    idx = (-proba[:,i]).argsort()[:10]
    print(get_key(target_dict, i),idx+9501)
# for i in range(0, len(predicted_y)):
#     if predicted_y[i]==1:
#         print(get_key(target_dict, i),predicted_y[i])
print(f'train data precision is {accuracy_score(y_test, predicted_y)}')
print(accuracy_score(y_test, predicted_y))
print(classification_report(y_test, predicted_y))
print(f'test data precision is {accuracy_score(y_test, predicted_y)}')

