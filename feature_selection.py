from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.datasets import load_svmlight_file
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.svm  import SVC
import warnings
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
#Load the training data file
feature_vectors, targets = load_svmlight_file('training_data_file')
X=feature_vectors
y=targets
#list of number of samples taken each time
k_list=[]
#list of f1 scores of each k for all the four classifiers for both CHI-Squared and Mutual Information feature selection methods
f1_macro_MNB_CHI=[]
f1_macro_BNB_CHI=[]
f1_macro_KNN_CHI=[]
f1_macro_SVC_CHI=[]
f1_macro_MNB_MI=[]
f1_macro_BNB_MI=[]
f1_macro_KNN_MI=[]
f1_macro_SVC_MI=[]

for i in range(100,8000,500):
    print(i)
    k_list.append(i)
    feature_vectors_CHI = SelectKBest(chi2, k=i).fit_transform(X, y)
    feature_vectors_MI = SelectKBest(mutual_info_classif, k=i).fit_transform(X, y)

    print("CHI_SQUARED")
    #MultinomialNB
    clf_MNB = MultinomialNB()
    #f1_macro
    scores_MNB = cross_val_score(clf_MNB, feature_vectors_CHI, targets, cv=5, scoring='f1_macro')
    print("Accuracy of MNB(f1_macro): %0.2f (+/- %0.2f)" % (scores_MNB.mean(), scores_MNB.std() * 2))
    f1_macro_MNB_CHI.append(scores_MNB.mean())
    print("\n")
    #Bernoulli NB
    clf_BNB=BernoulliNB()
    #f1_macro
    scores_BNB = cross_val_score(clf_BNB, feature_vectors_CHI, targets, cv=5, scoring='f1_macro')
    print("Accuracy of BNB(f1_macro): %0.2f (+/- %0.2f)" % (scores_BNB.mean(), scores_BNB.std() * 2))
    f1_macro_BNB_CHI.append(scores_BNB.mean())
    print("\n")

    #KNN
    clf_KNN = KNeighborsClassifier()
    #f1_macro
    scores_KNN = cross_val_score(clf_KNN, feature_vectors_CHI, targets, cv=5, scoring='f1_macro')
    print("Accuracy of KNN(f1_macro): %0.2f (+/- %0.2f)" % (scores_KNN.mean(), scores_KNN.std() * 2))
    f1_macro_KNN_CHI.append(scores_KNN.mean())
    print("\n")

    #SVC

    clf_SVC = SVC()
    #f1_macro
    scores_SVC = cross_val_score(clf_SVC, feature_vectors_CHI, targets, cv=5, scoring='f1_macro')
    print("Accuracy of SVC(f1_macro): %0.2f (+/- %0.2f)" % (scores_SVC.mean(), scores_SVC.std() * 2))
    f1_macro_SVC_CHI.append(scores_SVC.mean())
    print("\n")

    print("MI")
    # MultinomialNB
    clf_MNB = MultinomialNB()
    # f1_macro
    scores_MNB = cross_val_score(clf_MNB, feature_vectors_MI, targets, cv=5, scoring='f1_macro')
    print("Accuracy of MNB(f1_macro): %0.2f (+/- %0.2f)" % (scores_MNB.mean(), scores_MNB.std() * 2))
    f1_macro_MNB_MI.append(scores_MNB.mean())
    print("\n")
    # Bernoulli NB
    clf_BNB = BernoulliNB()
    # f1_macro
    scores_BNB = cross_val_score(clf_BNB, feature_vectors_MI, targets, cv=5, scoring='f1_macro')
    print("Accuracy of BNB(f1_macro): %0.2f (+/- %0.2f)" % (scores_BNB.mean(), scores_BNB.std() * 2))
    f1_macro_BNB_MI.append(scores_BNB.mean())
    print("\n")

    # KNN
    clf_KNN = KNeighborsClassifier()
    # f1_macro
    scores_KNN = cross_val_score(clf_KNN, feature_vectors_MI, targets, cv=5, scoring='f1_macro')
    print("Accuracy of KNN(f1_macro): %0.2f (+/- %0.2f)" % (scores_KNN.mean(), scores_KNN.std() * 2))
    f1_macro_KNN_MI.append(scores_KNN.mean())
    print("\n")

    # SVC

    clf_SVC = SVC()
    # f1_macro
    scores_SVC = cross_val_score(clf_SVC, feature_vectors_MI, targets, cv=5, scoring='f1_macro')
    print("Accuracy of SVC(f1_macro): %0.2f (+/- %0.2f)" % (scores_SVC.mean(), scores_SVC.std() * 2))
    f1_macro_SVC_MI.append(scores_SVC.mean())
    print("\n")

# plotting (k,f1-macro) for CHI-squared feature selection for all the four classifiers
plt.figure(1)
plt.plot(k_list, f1_macro_MNB_CHI, label = "MNB")
plt.plot(k_list, f1_macro_BNB_CHI, label = "BNB")
plt.plot(k_list, f1_macro_KNN_CHI, label = "KNN")
plt.plot(k_list, f1_macro_SVC_CHI, label = "SVC")
plt.xlabel('k-values')
plt.ylabel('f1_macro')
plt.title('CHI-squared')
plt.legend()
plt.show()

# plotting (k,f1-macro) for MutualInformation feature selection for all the four classifiers
plt.figure(2)
plt.plot(k_list, f1_macro_MNB_MI, label = "MNB")
plt.plot(k_list, f1_macro_BNB_MI, label = "BNB")
plt.plot(k_list, f1_macro_KNN_MI, label = "KNN")
plt.plot(k_list, f1_macro_SVC_MI, label = "SVC")
plt.xlabel('k-values')
plt.ylabel('f1_macro')
plt.title('Mutual Informtion')
plt.legend()
plt.show()

