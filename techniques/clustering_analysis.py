from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage

def clustering_analysis(df, scaled_data):
    """
    Perform comprehensive clustering analysis
    """
    # Determine optimal number of clusters
    inertias = []
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []
    
    K_range = range(2, 11)
    
    print("Evaluating clusters from 2 to 10...")
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_data)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_data, labels))
        davies_bouldin_scores.append(davies_bouldin_score(scaled_data, labels))
        calinski_harabasz_scores.append(calinski_harabasz_score(scaled_data, labels))
    
    # Find optimal k
    optimal_k_silhouette = K_range[np.argmax(silhouette_scores)]
    optimal_k_db = K_range[np.argmin(davies_bouldin_scores)]
    
    print(f"\nOptimal k (Silhouette Score): {optimal_k_silhouette}")
    print(f"Optimal k (Davies-Bouldin Index): {optimal_k_db}")
    
    # Use consensus optimal k
    optimal_k = optimal_k_silhouette
    
    return {
        'K_range': K_range,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'davies_bouldin_scores': davies_bouldin_scores,
        'calinski_harabasz_scores': calinski_harabasz_scores,
        'optimal_k': optimal_k
    }

# Prepare data
numeric_cols = df.select_dtypes(include=[np.number]).columns
data = df[numeric_cols].dropna()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Execute clustering analysis
cluster_results = clustering_analysis(df, scaled_data)

# Visualize cluster evaluation metrics
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Elbow Method
axes[0, 0].plot(cluster_results['K_range'], cluster_results['inertias'], 'bo-', linewidth=2)
axes[0, 0].set_xlabel('Number of Clusters (k)')
axes[0, 0].set_ylabel('Inertia')
axes[0, 0].set_title('Elbow Method', fontweight='bold')
axes[0, 0].grid(alpha=0.3)

# Silhouette Score
axes[0, 1].plot(cluster_results['K_range'], cluster_results['silhouette_scores'], 'go-', linewidth=2)
axes[0, 1].axvline(x=cluster_results['optimal_k'], color='r', linestyle='--', 
                   label=f"Optimal k={cluster_results['optimal_k']}")
axes[0, 1].set_xlabel('Number of Clusters (k)')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].set_title('Silhouette Score Analysis', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Davies-Bouldin Index (lower is better)
axes[1, 0].plot(cluster_results['K_range'], cluster_results['davies_bouldin_scores'], 'mo-', linewidth=2)
axes[1, 0].set_xlabel('Number of Clusters (k)')
axes[1, 0].set_ylabel('Davies-Bouldin Index')
axes[1, 0].set_title('Davies-Bouldin Index (Lower is Better)', fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Calinski-Harabasz Score (higher is better)
axes[1, 1].plot(cluster_results['K_range'], cluster_results['calinski_harabasz_scores'], 'co-', linewidth=2)
axes[1, 1].set_xlabel('Number of Clusters (k)')
axes[1, 1].set_ylabel('Calinski-Harabasz Score')
axes[1, 1].set_title('Calinski-Harabasz Score (Higher is Better)', fontweight='bold')
axes[1, 1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('cluster_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

# Apply K-Means with optimal k
optimal_k = cluster_results['optimal_k']
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(scaled_data)
print(f"\n✓ K-Means clustering complete! Using k={optimal_k}")
print(f"Silhouette Score: {silhouette_score(scaled_data, df['KMeans_Cluster']):.4f}")

# Hierarchical Clustering
print("\nPerforming Hierarchical Clustering...")
linkage_matrix = linkage(scaled_data, method='ward')

# Create dendrogram
fig, ax = plt.subplots(figsize=(15, 7))
dendrogram(linkage_matrix, ax=ax, leaf_font_size=8)
ax.set_title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Distance')
ax.axhline(y=np.sort(linkage_matrix[:, 2])[-optimal_k], color='r', 
           linestyle='--', label=f'Cut for {optimal_k} clusters')
ax.legend()
plt.tight_layout()
plt.savefig('dendrogram.png', dpi=300, bbox_inches='tight')
plt.show()

# Apply hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
df['Hierarchical_Cluster'] = hierarchical.fit_predict(scaled_data)
print(f"Silhouette Score: {silhouette_score(scaled_data, df['Hierarchical_Cluster']):.4f}")
print("\n✓ Clustering analysis complete!")