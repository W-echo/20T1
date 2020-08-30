## Model: MNB
## Feature matrix: bag of word
## Author: Xiaoyu Wang and Naihao Liu
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import KFold
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest

# Create text
data = pd.read_csv("training.csv")
text_data = np.array(data)

# Create bag of words
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data[:, 1])

# Create feature matrix
x = bag_of_words
kf = KFold(n_splits=10)

# Create target vector
target_dict = {'ARTS CULTURE ENTERTAINMENT': 0, 'BIOGRAPHIES PERSONALITIES PEOPLE': 1, 'DEFENCE': 2,
               'DOMESTIC MARKETS': 3, 'FOREX MARKETS': 4, 'HEALTH': 5, 'MONEY MARKETS': 6,
               'SCIENCE AND TECHNOLOGY': 7, 'SHARE LISTINGS': 8, 'SPORTS': 9, 'IRRELEVANT': 10}
y = []
for i in text_data[:, -1]:
    y.append(target_dict[i])
y = np.array(y)
clf = MultinomialNB()

# feature number selection
sel = SelectKBest(f_classif, k=15100)
sel.fit(x, y)
x = sel.transform(x)

final_result = []
for train_index, test_index in kf.split(x):
    # print("Train:", train_index)
    # print("Validation:", len(test_index))
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = clf.fit(x_train, y_train)
    # x_test = sel.transform(x_test)
    predicted_y = model.predict(x_test)
    final_result.append(accuracy_score(y_test, predicted_y))
    # print(y_test, predicted_y)
    # print(model.predict_proba(x_test))
# print(final_result)
print(max(final_result))
# print(precision_score(y_test, predicted_y, average='macro'))
# print(recall_score(y_test, predicted_y, average='macro'))
# print(f1_score(y_test, predicted_y, average='macro'))
# print(f1_score(y_test, predicted_y, average='micro'))
# print(classification_report(y_test, predicted_y))

