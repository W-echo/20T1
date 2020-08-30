## Model: MNB
## Feature matrix: TFIDF
## Author: Xiaoyu Wang and Naihao Liu
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Create text
data = pd.read_csv("training.csv")

text_data = np.array(data)

# Create bag of words
count = TfidfVectorizer()
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
clf =  DecisionTreeClassifier(min_samples_leaf = 21)

# k_range = range(2, 25)
# k_scores = []
# for k in k_range:
#     print(k)
#     clf = DecisionTreeClassifier(min_samples_leaf = k)
#     scores = cross_val_score(clf, x, y, cv=5, scoring='accuracy')
#     k_scores.append(scores.mean())
# plt.plot(k_range, k_scores)
# plt.xlabel('The number of leaf')
# plt.ylabel('Cross-Validated Accuracy')
# plt.xticks(k_range)
# plt.show()

# clf = DecisionTreeClassifier(min_samples_leaf = 21)
# scores = cross_val_score(clf, x, y, cv=5, scoring='accuracy')
# plt.plot(scores)
# plt.title('CountVectorizer')
# plt.ylabel('Cross-Validated Accuracy')
# plt.show()


final_result = []
model_dic = {}

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


test = pd.read_csv("test.csv")
test_data = np.array(test)
y_test = []
for i in test_data[:, -1]:
    y_test.append(target_dict[i])
x_test = count.transform(test_data[:, 1])
y_test = np.array(y_test)
predicted_y = model.predict(x_test)
print(classification_report(y_test, predicted_y))
print(f'test data precision is {accuracy_score(y_test, predicted_y)}')

