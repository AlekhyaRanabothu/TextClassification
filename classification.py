from sklearn.datasets import load_svmlight_file
import numpy as np
import warnings
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.svm  import SVC
from sklearn.model_selection import cross_val_score
#Load the training data file 
feature_vectors, targets = load_svmlight_file('training_data_file')
warnings.filterwarnings('ignore')
# print(np.array(feature_vectors))
# print(targets)
#MultinomialNB
clf_MNB = MultinomialNB()
#f1_macro
scores_MNB = cross_val_score(clf_MNB, feature_vectors, targets, cv=5, scoring='f1_macro')
print("Accuracy of MNB(f1_macro): %0.2f (+/- %0.2f)" % (scores_MNB.mean(), scores_MNB.std() * 2))

#precision_macro
scores_MNB = cross_val_score(clf_MNB, feature_vectors, targets, cv=5, scoring='precision_macro')
print("Accuracy of MNB(precision_macro): %0.2f (+/- %0.2f)" % (scores_MNB.mean(), scores_MNB.std() * 2))

#recall_macro
scores_MNB = cross_val_score(clf_MNB, feature_vectors, targets, cv=5, scoring='recall_macro')
print("Accuracy of MNB(recall_macro): %0.2f (+/- %0.2f)" % (scores_MNB.mean(), scores_MNB.std() * 2))
print("\n")
#Bernoulli NB
clf_BNB=BernoulliNB()
#f1_macro
scores_BNB = cross_val_score(clf_BNB, feature_vectors, targets, cv=5, scoring='f1_macro')
print("Accuracy of BNB(f1_macro): %0.2f (+/- %0.2f)" % (scores_BNB.mean(), scores_BNB.std() * 2))

#precision_macro
scores_BNB = cross_val_score(clf_BNB, feature_vectors, targets, cv=5, scoring='precision_macro')
print("Accuracy of BNB(precision_macro): %0.2f (+/- %0.2f)" % (scores_BNB.mean(), scores_BNB.std() * 2))

#recall_macro
scores_BNB = cross_val_score(clf_BNB, feature_vectors, targets, cv=5, scoring='recall_macro')
print("Accuracy of BNB(recall_macro): %0.2f (+/- %0.2f)" % (scores_BNB.mean(), scores_BNB.std() * 2))
print("\n")
#KNN
clf_KNN = KNeighborsClassifier()
#f1_macro
scores_KNN = cross_val_score(clf_KNN, feature_vectors, targets, cv=5, scoring='f1_macro')
print("Accuracy of KNN(f1_macro): %0.2f (+/- %0.2f)" % (scores_KNN.mean(), scores_KNN.std() * 2))

#precision_macro
scores_KNN = cross_val_score(clf_KNN, feature_vectors, targets, cv=5, scoring='precision_macro')
print("Accuracy of KNN(precision_macro): %0.2f (+/- %0.2f)" % (scores_KNN.mean(), scores_KNN.std() * 2))

#recall_macro
scores_KNN = cross_val_score(clf_KNN, feature_vectors, targets, cv=5, scoring='recall_macro')
print("Accuracy of KNN(recall_macro): %0.2f (+/- %0.2f)" % (scores_KNN.mean(), scores_KNN.std() * 2))
print("\n")
#SVC

clf_SVC = SVC()
#f1_macro
scores_SVC = cross_val_score(clf_SVC, feature_vectors, targets, cv=5, scoring='f1_macro')
print("Accuracy of SVC(f1_macro): %0.2f (+/- %0.2f)" % (scores_SVC.mean(), scores_SVC.std() * 2))

#precision_macro
scores_SVC = cross_val_score(clf_SVC, feature_vectors, targets, cv=5, scoring='precision_macro')
print("Accuracy of SVC(precision_macro): %0.2f (+/- %0.2f)" % (scores_SVC.mean(), scores_SVC.std() * 2))

#recall_macro
scores_SVC = cross_val_score(clf_SVC, feature_vectors, targets, cv=5, scoring='recall_macro')
print("Accuracy of SVC(recall_macro): %0.2f (+/- %0.2f)" % (scores_SVC.mean(), scores_SVC.std() * 2))
