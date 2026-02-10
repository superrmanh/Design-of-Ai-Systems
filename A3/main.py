from classifier import Classifier
from data import X_train, Y_train, X_val, Y_val, X_test, Y_test
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

k = 7
clf = Classifier(k=k)
clf.fit(X_train, Y_train)
print("Cluster -> label:", clf.label)
print("Unique predictions:", np.unique(clf.predict(X_val)))
pred = clf.predict(X_val)
acc,f1, precision,recall = clf.score(X_val, Y_val)

# Training set
train_acc, train_f1, train_precision, train_recall = clf.score(X_train, Y_train)
print("TRAIN - Accuracy:", train_acc*100, "%")
print("TRAIN - F1 Score:", train_f1)
print("TRAIN - Precision:", train_precision)
print("TRAIN - Recall:", train_recall)

print("K value:",k)
print("VAL - Accuracy:",acc*100, "%")
print("VAL - F1 Score:", f1)
print("VAL - Precision:",precision)
print("VAL - Recall:", recall)

#Test set
test_acc, test_f1, test_precision, test_recall = clf.score(X_test, Y_test)
print("TEST - Accuracy:", test_acc*100,"%")
print("TEST - F1 Score:", test_f1)
print("TEST - Precision:",test_precision)
print("TEST - Recall:", test_recall)

