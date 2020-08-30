import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.svm import SVC

target_dict = {'ARTS CULTURE ENTERTAINMENT': 0, 'BIOGRAPHIES PERSONALITIES PEOPLE': 1, 'DEFENCE': 2,
               'DOMESTIC MARKETS': 3, 'FOREX MARKETS': 4, 'HEALTH': 5, 'MONEY MARKETS': 6, 'SCIENCE AND TECHNOLOGY': 7,
               'SHARE LISTINGS': 8, 'SPORTS': 9, 'IRRELEVANT': 10}

data = pd.read_csv("training.csv")
text_data = np.array(data)

count = TfidfVectorizer()
x_train = count.fit_transform(text_data[:1000, 1])
y_train = []
for i in text_data[:1000, -1]:
    y_train.append(target_dict[i])

y_train = np.array(y_train)
clf = SVC(kernel='linear')
clf.fit(x_train, y_train)

x_test = count.transform(text_data[1000:, 1])
y_test = []
for i in text_data[1000:, -1]:
    y_test.append(target_dict[i])
y_test = np.array(y_test)

predicted_y = clf.predict(x_test)
pre = predicted_y.tolist()
print(accuracy_score(y_test, pre))