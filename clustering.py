from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import load_svmlight_file
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
#Loading training data file
feature_vectors, targets = load_svmlight_file('training_data_file')
#selecting features using CHI squared method
select_features=SelectKBest(chi2, k=1000).fit_transform(feature_vectors, targets)
select_features=select_features.toarray()
#list of number of clusters
num_clusters=[]
#list of Silhoutte and Normalised Mutual Information measures for KMeans and Agglomerative clustering
Silhoutte_Kmeans=[]
NMI_Kmeans=[]
Silhoutte_Agg=[]
NMI_Agg=[]
#Calculating Silhoutte and Normalised Mutual Information measures for KMeans and Agglomerative clustering for number of clusters ranging from 2 to 25
for n in range(2,26):
    #print(n)
    num_clusters.append(n)
    kmeans_model = KMeans(n_clusters=n).fit(select_features)
    clustering_labels = kmeans_model.labels_
    Silhoutte_Kmeans.append(metrics.silhouette_score(select_features, clustering_labels, metric='euclidean'))
    NMI_Kmeans.append(metrics.normalized_mutual_info_score(targets, clustering_labels))
    average_linkage_model = AgglomerativeClustering(
        n_clusters=n, linkage='average').fit(select_features)
    clustering_labels = average_linkage_model.labels_
    Silhoutte_Agg.append(metrics.silhouette_score(select_features, clustering_labels, metric='euclidean'))
    NMI_Agg.append(metrics.normalized_mutual_info_score(targets, clustering_labels))
#plotting Silhoutte score for KMeans and Agglomerative clustering
plt.figure(1)
plt.plot(num_clusters, Silhoutte_Kmeans, label = "KMeans")
plt.plot(num_clusters, Silhoutte_Agg, label = "Agglomerative")
plt.xlabel('number of clusters')
plt.ylabel('Silhoutte score')
plt.title('Silhoutte')
plt.legend()
plt.show()

#plotting Normalized Mutual Information for KMeans and Agglomerative clustering
plt.figure(2)
plt.plot(num_clusters, NMI_Kmeans, label = "KMeans")
plt.plot(num_clusters, NMI_Agg, label = "Agglomerative")
plt.xlabel('number of clusters')
plt.ylabel('NMI score')
plt.title('Normalized Mutual Information')
plt.legend()
plt.show()

