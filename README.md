# Text Classifiaction (Natural Language Processing)

-- Text Classifiaction on mini_newsgroups dataset

-- Data Cleaning

-- Feature extraction by building inverted Index using TFIDF values and encoding class labels

-- Feature Selection Techniques- Chi-Squared and MutualInformation

-- Classification using Multinomial NaiveBayes, BernoulliNaiveBayes,KNN,SVC classifiers

-- Evaluation-k-fold cross validation,f1-score,precision,recall

-- Clustering

   - KMeans and Hierarchical(Agglomerative Clustering)

   - distance metric used Euclidean

-- clustering quality is measured for a range of number of clusters using SilhoutteCoefficient and Normalized MutualInformation

****************************************************************************************
# Project flow and Additional details

•	The main goal of this project is to classify a given document. The dataset used in this project is 20 mini news group data set. The 20 news groups are divided among 6 classes. So if a document is given, it should be classified into one of the 6 classes. There are 4 steps in this project- Feature Extraction, classification, feature selection, clustering.

•	In the feature extraction, we have done data cleaning (like adding the number of lines manually which helps in reading the files, removed duplicate document IDs).

•	After data cleaning, we have built the training data file. Each row in the training file corresponds to a document in the dataset. Each row contains class-label, feature ids and feature value pairs, feature ids are the terms in the dataset and the feature values are tf-idf values. 

•	In the classification, this training  data is classified using Multinomial NaiveBayes, Bernouli NaiveBayes, KNN, SVM classifiers and accuracy is measured using k-fold cross validation and the metrics used are precision, recall, F1 score. 

•	In the Feature selection, Chi squared and mutual information methods are used to select the best k features and classification is applied for the selected features using Multinomial NaiveBayes, Bernouli NaiveBayes, KNN, SVM classifiers and the performance is measured and compared with the performance when all the features are used. 

•	In Clustering, hard clustering(K-means) and hierarchical clustering (agglomerative) are used. Silhouette score and Normalized Mutual Information Methods are used for measuring the cluster qualities of K-means and agglomerative clustering.

