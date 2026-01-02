from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def dimensionality_reduction_analysis(df):
    """
    Apply multiple dimensionality reduction techniques
    """
    # Prepare data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    data = df[numeric_cols].dropna()
    
    # Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 1. PCA Analysis
    print("Performing PCA...")
    pca_full = PCA()
    pca_full.fit(scaled_data)
    
    # Calculate cumulative explained variance
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Find number of components for 95% variance
    n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
    
    print(f"Number of components for 95% variance: {n_components_95}")
    print(f"Total features: {len(numeric_cols)}")
    print(f"Reduction ratio: {(1 - n_components_95/len(numeric_cols))*100:.1f}%")
    
    # Apply PCA with optimal components
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    
    # 2. t-SNE Analysis
    print("\nPerforming t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_data = tsne.fit_transform(scaled_data)
    
    # 3. UMAP Analysis
    print("Performing UMAP...")
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_data = umap_reducer.fit_transform(scaled_data)
    
    return {
        'pca': pca,
        'pca_data': pca_data,
        'pca_full': pca_full,
        'tsne_data': tsne_data,
        'umap_data': umap_data,
        'scaled_data': scaled_data,
        'cumsum_var': cumsum_var
    }

# Execute dimensionality reduction
dim_red_results = dimensionality_reduction_analysis(df)

# Visualize results
fig = plt.figure(figsize=(18, 12))

# 1. PCA Explained Variance
ax1 = plt.subplot(2, 3, 1)
n_components = len(dim_red_results['cumsum_var'])
ax1.plot(range(1, n_components + 1), dim_red_results['cumsum_var'], 'bo-', linewidth=2)
ax1.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
ax1.set_xlabel('Number of Components')
ax1.set_ylabel('Cumulative Explained Variance')
ax1.set_title('PCA: Cumulative Variance Explained', fontweight='bold')
ax1.grid(alpha=0.3)
ax1.legend()

# 2. Individual Variance per Component
ax2 = plt.subplot(2, 3, 2)
ax2.bar(range(1, min(11, n_components + 1)), 
        dim_red_results['pca_full'].explained_variance_ratio_[:10], 
        color='steelblue', edgecolor='black')
ax2.set_xlabel('Principal Component')
ax2.set_ylabel('Explained Variance Ratio')
ax2.set_title('PCA: Individual Variance Per Component', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 3. PCA 2D Projection
ax3 = plt.subplot(2, 3, 3)
scatter = ax3.scatter(dim_red_results['pca_data'][:, 0], 
                      dim_red_results['pca_data'][:, 1], 
                      c=df['Target'].values, cmap='viridis', alpha=0.6, s=50)
ax3.set_xlabel(f"PC1 ({dim_red_results['pca'].explained_variance_ratio_:.2%})")
ax3.set_ylabel(f"PC2 ({dim_red_results['pca'].explained_variance_ratio_:.2%})")
ax3.set_title('PCA: 2D Projection', fontweight='bold')
plt.colorbar(scatter, ax=ax3, label='Target')
ax3.grid(alpha=0.3)

# 4. t-SNE Projection
ax4 = plt.subplot(2, 3, 4)
scatter = ax4.scatter(dim_red_results['tsne_data'][:, 0], 
                      dim_red_results['tsne_data'][:, 1], 
                      c=df['Target'].values, cmap='viridis', alpha=0.6, s=50)
ax4.set_xlabel('t-SNE Dimension 1')
ax4.set_ylabel('t-SNE Dimension 2')
ax4.set_title('t-SNE: Non-linear Dimensionality Reduction', fontweight='bold')
plt.colorbar(scatter, ax=ax4, label='Target')
ax4.grid(alpha=0.3)

# 5. UMAP Projection
ax5 = plt.subplot(2, 3, 5)
scatter = ax5.scatter(dim_red_results['umap_data'][:, 0], 
                      dim_red_results['umap_data'][:, 1], 
                      c=df['Target'].values, cmap='viridis', alpha=0.6, s=50)
ax5.set_xlabel('UMAP Dimension 1')
ax5.set_ylabel('UMAP Dimension 2')
ax5.set_title('UMAP: Fast Non-linear Reduction', fontweight='bold')
plt.colorbar(scatter, ax=ax5, label='Target')
ax5.grid(alpha=0.3)

# 6. Comparison (PCA vs UMAP)
ax6 = plt.subplot(2, 3, 6)
ax6.scatter(dim_red_results['pca_data'][:, 0], dim_red_results['pca_data'][:, 1], 
            alpha=0.5, s=30, label='PCA', color='blue')
ax6.scatter(dim_red_results['umap_data'][:, 0], dim_red_results['umap_data'][:, 1], 
            alpha=0.5, s=30, label='UMAP', color='red')
ax6.set_xlabel('Dimension 1')
ax6.set_ylabel('Dimension 2')
ax6.set_title('Method Comparison: PCA vs UMAP', fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('dimensionality_reduction.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ Dimensionality reduction analysis complete!")