import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.datasets import load_breast_cancer 
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA 
from sklearn.metrics import silhouette_score, adjusted_rand_score 

df = pd.DataFrame(X, columns=feature_names) 
df['target'] = y  # Add target labels 
df.head()

3. Normalize the Data 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X)  # Standardize features 

4. Apply K-Means Clustering 
# Choose number of clusters (K=2 as we have benign/malignant labels) 
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10) 
y_pred = kmeans.fit_predict(X_scaled) 
# Add cluster labels to DataFrame 
df['cluster'] = y_pred 

5. Evaluate Clustering Performance 

ari_score = adjusted_rand_score(y, y_pred) 
silhouette_avg = silhouette_score(X_scaled, y_pred) 
print(f"Adjusted Rand Index: {ari_score:.3f}") 
print(f"Silhouette Score: {silhouette_avg:.3f}") 

6. Visualize Clusters using PCA (2D Projection) 
 
pca = PCA(n_components=2) 
X_pca = pca.fit_transform(X_scaled) 
63 
  
plt.figure(figsize=(10, 6)) 
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_pred, palette='coolwarm', alpha=0.7) 
plt.title("K-Means Clustering on Breast Cancer Dataset (PCA Projection)") 
plt.xlabel("Principal Component 1") 
plt.ylabel("Principal Component 2") 
plt.legend(title="Cluster") 
plt.show()
