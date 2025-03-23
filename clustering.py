import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

plt.ioff()

# Set working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

df = pd.read_csv("cleaned_sentiment_dataset.csv")

# Verify required columns
required_columns = ['Clean_Text', 'Timestamp']
missing = [col for col in required_columns if col not in df.columns]
if missing:
    print(f"Missing columns: {', '.join(missing)}")
    exit()

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf.fit_transform(df['Clean_Text'])

# Determine optimal clusters
cluster_range = range(2, 11)
inertia = []
silhouette_scores = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_tfidf)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_tfidf, labels))

# Plot elbow and silhouette scores
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(cluster_range, inertia, marker='o')
ax1.set(title='Elbow Method', xlabel='Clusters', ylabel='Inertia')
ax2.plot(cluster_range, silhouette_scores, marker='o', color='green')
ax2.set(title='Silhouette Scores', xlabel='Clusters', ylabel='Score')
plt.tight_layout()
plt.savefig('elbow_and_silhouette_scores.png')
plt.show()
plt.close()  # Explicitly close the figure

# Cluster analysis
best_k = cluster_range[np.argmax(silhouette_scores)]
print(f"Optimal clusters: {best_k} (Silhouette: {max(silhouette_scores):.2f})")

kmeans = KMeans(n_clusters=best_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_tfidf)

# PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_tfidf.toarray())
df['PC1'], df['PC2'] = X_pca[:, 0], X_pca[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df,
                palette='viridis', alpha=0.8, edgecolor='k')
plt.title('2D Cluster Visualization (PCA)')
plt.savefig('2D_Cluster_Visualization_PCA.png')
plt.show()
plt.close()  # Explicitly close the figure

# Word Clouds with controlled display
for cluster in range(best_k):
    fig = plt.figure(figsize=(10, 5))
    cluster_text = ' '.join(df[df['Cluster'] == cluster]['Clean_Text'])
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white').generate(cluster_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Cluster {cluster} Keywords', fontsize=12)
    plt.axis('off')
    plt.savefig(f'Cluster_{cluster}_Keywords.png')
    plt.show()
    plt.close(fig)  # Explicitly close each word cloud figure

# Save results
df.to_csv("clustered_data.csv", index=False)
print("Clustered data saved to clustered_data.csv")
