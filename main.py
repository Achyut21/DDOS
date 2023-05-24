import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


df = pd.read_csv('DDoS Dataset.csv')
#X = df.drop(columns=['Target', 'Source_IP', 'Source_PORT', 'Destination_IP', 'Destination_PORT'])
X = df.drop(columns=['Target'])
Y = df['Target']
#print(df.head())


# creating mask

mask = np.triu(np.ones_like(df.corr()))

# plotting a triangle correlation heatmap

dataplot = sns.heatmap(df.corr(), cmap="YlGnBu", annot=True, mask=mask)

# displaying heatmap
plt.show()

#Dropping

XxX = df.drop(columns=['CSD_Payload_4Gram','CSD_Payload_1Gram','CSD_Payload_2Gram','CSD_Payload_5Gram'])
mask = np.triu(np.ones_like(XxX.corr()))

# plotting a triangle correlation heatmap

dataplot = sns.heatmap(XxX.corr(), cmap="YlGnBu", annot=True, mask=mask)

# displaying heatmap
plt.show()
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////

# navie bayes
from sklearn.naive_bayes import GaussianNB

NB_model = GaussianNB()
X_train, X_test, Y_Train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=21)
NB_model.fit(X_train, Y_Train)
X_train_prediction = NB_model.predict(X_train)
train_acc = accuracy_score(X_train_prediction, Y_Train)
print("NB_Training accuracy: ", train_acc)

X_test_prediction = NB_model.predict(X_test)
test_acc = accuracy_score(X_test_prediction, Y_test)
print("NB_Testing accuracy: ", test_acc)
conf_mat = confusion_matrix(X_test_prediction, Y_test)
print(conf_mat)

#performance parameters plot .....
precision = precision_score(Y_test, X_test_prediction)
recall = recall_score(Y_test, X_test_prediction)
f1 = f1_score(Y_test, X_test_prediction)

# Print classification report
print("Classification Report:")
print(classification_report(Y_test, X_test_prediction))




# Knearest Neighbors



from sklearn.neighbors import KNeighborsClassifier

sts = MinMaxScaler()
X_trs = sts.fit_transform(X)
KNN_model = KNeighborsClassifier(n_neighbors=7)
X_train, X_test, Y_Train, Y_test = train_test_split(X_trs, Y, test_size=0.2, stratify=Y, random_state=21)
KNN_model.fit(X_train, Y_Train)
X_train_prediction = KNN_model.predict(X_train)
train_acc = accuracy_score(X_train_prediction, Y_Train)
print("\nKNN_Training accuracy: ", train_acc)

X_test_prediction = KNN_model.predict(X_test)
test_acc = accuracy_score(X_test_prediction, Y_test)
print("KNN_Testing accuracy: ", test_acc)

conf_mat = confusion_matrix(X_test_prediction, Y_test)
print(conf_mat)

#performance parameters plot .....
precision = precision_score(Y_test, X_test_prediction)
recall = recall_score(Y_test, X_test_prediction)
f1 = f1_score(Y_test, X_test_prediction)

# Print classification report
print("Classification Report for KNN:")
print(classification_report(Y_test, X_test_prediction))

# Logistic regression

from sklearn.linear_model import LogisticRegression

log_classifier = LogisticRegression(random_state=0)
X_train, X_test, Y_Train, Y_test = train_test_split(X_trs, Y, test_size=0.2, stratify=Y, random_state=22)
log_classifier.fit(X_train, Y_Train)

logreg_X_train_prediction = log_classifier.predict(X_train)
train_acc = accuracy_score(logreg_X_train_prediction, Y_Train)
print("\nLog_regression_Training accuracy: ", train_acc)

logreg_X_test_prediction = log_classifier.predict(X_test)
test_acc = accuracy_score(logreg_X_test_prediction, Y_test)
print("Log_regression_Testing accuracy: ", test_acc)

plt.figure(figsize=(13, 11))
sns.heatmap(df.corr(), annot=True)
plt.show()

conf_mat = confusion_matrix(logreg_X_test_prediction, Y_test)
print(conf_mat)
clf = LogisticRegression()
ConfusionMatrixDisplay(conf_mat).plot()


train_precision = precision_score(Y_Train, logreg_X_train_psniprediction)
train_recall = recall_score(Y_Train, logreg_X_train_prediction)
train_f1 = f1_score(Y_Train, logreg_X_train_prediction)


# Print performance metrics for training data
print("Train Precision: ", train_precision)
print("Train Recall: ", train_recall)
print("Train F1 Score: ", train_f1)






