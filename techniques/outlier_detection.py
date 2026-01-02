from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

def detect_outliers_multiple_methods(df, numeric_cols=None):
    """
    Detect outliers using 5 different methods and compare results
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Prepare data
    data = df[numeric_cols].copy()
    data_clean = data.dropna()
    
    # Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_clean)
    
    # Method 1: Z-Score (Statistical)
    z_scores = np.abs(stats.zscore(scaled_data))
    outliers_zscore = (z_scores > 3).any(axis=1)
    
    # Method 2: IQR (Statistical)
    Q1 = data_clean.quantile(0.25)
    Q3 = data_clean.quantile(0.75)
    IQR = Q3 - Q1
    outliers_iqr = ((data_clean < (Q1 - 1.5 * IQR)) | 
                    (data_clean > (Q3 + 1.5 * IQR))).any(axis=1)
    
    # Method 3: Isolation Forest (Model-based)
    iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
    outliers_if = iso_forest.fit_predict(scaled_data) == -1
    
    # Method 4: Local Outlier Factor (Density-based)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    outliers_lof = lof.fit_predict(scaled_data) == -1
    
    # Method 5: Mahalanobis Distance
    try:
        ee = EllipticEnvelope(contamination=0.05, random_state=42)
        outliers_ee = ee.fit_predict(scaled_data) == -1
    except:
        outliers_ee = np.zeros(len(scaled_data), dtype=bool)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Z-Score': outliers_zscore,
        'IQR': outliers_iqr,
        'Isolation Forest': outliers_if,
        'LOF': outliers_lof,
        'Elliptic Envelope': outliers_ee
    }, index=data_clean.index)
    
    # Count outliers detected by each method
    print("Outliers Detected by Each Method:")
    print(results.sum())
    print(f"\nTotal samples: {len(data_clean)}")
    
    # Find consensus outliers (detected by at least 3 methods)
    consensus_outliers = (results.sum(axis=1) >= 3)
    print(f"\nConsensus outliers (≥3 methods): {consensus_outliers.sum()}")
    
    return results, consensus_outliers

# Execute outlier detection
outlier_results, consensus = detect_outliers_multiple_methods(df)

# Visualize outlier detection results
numeric_cols = df.select_dtypes(include=[np.number]).columns[:2]
X = df[numeric_cols].dropna().values

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
methods = ['Z-Score', 'IQR', 'Isolation Forest', 'LOF', 'Elliptic Envelope', 'Consensus']

for idx, method in enumerate(methods):
    ax = axes[idx]
    
    if method == 'Consensus':
        colors = np.where(consensus.values, 'red', 'blue')
        title_text = f'{method}\n(≥3 methods)'
        outlier_count = consensus.sum()
    else:
        colors = np.where(outlier_results[method].values, 'red', 'blue')
        outlier_count = outlier_results[method].sum()
        title_text = f'{method}\n({outlier_count} outliers)'
    
    ax.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6, s=50)
    ax.set_xlabel(numeric_cols)
    ax.set_ylabel(numeric_cols)
    ax.set_title(title_text)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outlier_detection_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ Outlier detection comparison complete!")