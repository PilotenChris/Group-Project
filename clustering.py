import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv(r"cleaned_sentiment_dataset.csv")

# Verify required columns exist
assert 'Clean_Text' in df.columns, "Missing Clean_Text column - check preprocessing!"
assert 'Timestamp' in df.columns, "Missing Timestamp column - check datetime handling!"

# Convert text data into numerical features using TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf.fit_transform(df['Clean_Text'])

# Determine the optimal number of clusters using the Elbow Method
inertia = []
silhouette_scores = []
cluster_range = range(2, 11)  # Test 2 to 10 clusters

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_tfidf)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_tfidf, labels))

# Plot the Elbow Curve and Silhouette Scores
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(cluster_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')

plt.subplot(1, 2, 2)
plt.plot(cluster_range, silhouette_scores, marker='o', color='green')
plt.title('Silhouette Scores')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')

plt.tight_layout()
plt.show()

# Select the optimal number of clusters based on the highest silhouette score
best_k = cluster_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {best_k} (Silhouette Score: {max(silhouette_scores):.2f})")

# Perform K-Means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=best_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_tfidf)

# Analyze and visualize the clusters
# Get top keywords for each cluster
terms = tfidf.get_feature_names_out()
for i in range(best_k):
    print(f"\nCluster {i} Keywords:")
    cluster_terms = kmeans.cluster_centers_[i].argsort()[-10:][::-1]  # Top 10 keywords
    print([terms[idx] for idx in cluster_terms])

# Visualize clusters using PCA (2D projection)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_tfidf.toarray())
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df, palette='viridis', legend='full')
plt.title('2D Cluster Visualization (PCA Reduced)')
plt.show()

# Generate word clouds for each cluster
for cluster in range(best_k):
    cluster_text = ' '.join(df[df['Cluster'] == cluster]['Clean_Text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Cluster {cluster} Keywords')
    plt.axis('off')
    plt.show()

# Save the clustered data for further analysis
df.to_csv("clustered_data.csv", index=False)
print("\nClustered data saved to 'clustered_data.csv'")