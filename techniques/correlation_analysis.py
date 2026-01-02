from scipy.stats import pearsonr, spearmanr, kendalltau
from statsmodels.stats.outliers_influence import variance_inflation_factor

def advanced_correlation_analysis(df):
    """
    Perform comprehensive correlation analysis with multiple methods
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Method 1: Pearson Correlation (Linear relationships)
    pearson_corr = numeric_df.corr(method='pearson')
    
    # Method 2: Spearman Correlation (Rank-based, robust to outliers)
    spearman_corr = numeric_df.corr(method='spearman')
    
    # Method 3: Kendall Correlation (Ordinal data)
    kendall_corr = numeric_df.corr(method='kendall')
    
    return pearson_corr, spearman_corr, kendall_corr

# Execute correlation analysis
pearson_corr, spearman_corr, kendall_corr = advanced_correlation_analysis(df)

# Visualize correlation comparisons
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Pearson
sns.heatmap(pearson_corr, annot=False, cmap='coolwarm', center=0, 
            ax=axes, cbar_kws={'label': 'Correlation'})
axes.set_title('Pearson Correlation\n(Linear relationships)', fontsize=12, fontweight='bold')

# Spearman
sns.heatmap(spearman_corr, annot=False, cmap='coolwarm', center=0, 
            ax=axes, cbar_kws={'label': 'Correlation'})
axes.set_title('Spearman Correlation\n(Rank-based, robust)', fontsize=12, fontweight='bold')

# Kendall
sns.heatmap(kendall_corr, annot=False, cmap='coolwarm', center=0, 
            ax=axes, cbar_kws={'label': 'Correlation'})
axes.set_title('Kendall Correlation\n(Ordinal data)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Find top correlated pairs
print("\nTop 10 Correlated Pairs (Pearson):")
print("="*60)

# Get correlation pairs
corr_pairs = []
for i in range(len(pearson_corr.columns)):
    for j in range(i+1, len(pearson_corr.columns)):
        corr_pairs.append({
            'Var1': pearson_corr.columns[i],
            'Var2': pearson_corr.columns[j],
            'Pearson': pearson_corr.iloc[i, j],
            'Spearman': spearman_corr.iloc[i, j],
            'Kendall': kendall_corr.iloc[i, j]
        })
corr_df = pd.DataFrame(corr_pairs).sort_values('Pearson', key=abs, ascending=False)
print(corr_df.head(10).to_string())

# Multicollinearity Detection using VIF
print("\n\nVariance Inflation Factor (VIF) Analysis:")
print("="*60)
def calculate_vif(df):
    """
    Calculate VIF for all features to detect multicollinearity
    """
    numeric_df = df.select_dtypes(include=[np.number]).drop(
        columns=['Target'], errors='ignore')
    
    vif_data = pd.DataFrame()
    vif_data['Feature'] = numeric_df.columns
    vif_data['VIF'] = [variance_inflation_factor(numeric_df.values, i) 
                       for i in range(numeric_df.shape)]
    
    return vif_data.sort_values('VIF', ascending=False)

vif_results = calculate_vif(df)
print(vif_results.to_string())

# Visualize VIF
plt.figure(figsize=(12, 6))
bars = plt.barh(vif_results['Feature'], vif_results['VIF'], color='steelblue')

# Add threshold lines
plt.axvline(x=5, color='orange', linestyle='--', linewidth=2, label='VIF = 5 (Investigate)')
plt.axvline(x=10, color='red', linestyle='--', linewidth=2, label='VIF = 10 (Problem)')

# Color code bars based on VIF value
for i, (idx, row) in enumerate(vif_results.iterrows()):
    if row['VIF'] > 10:
        bars[i].set_color('red')
    elif row['VIF'] > 5:
        bars[i].set_color('orange')
    else:
        bars[i].set_color('green')

plt.xlabel('Variance Inflation Factor', fontsize=12)
plt.title('Multicollinearity Detection: VIF Analysis', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('vif_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ Correlation and multicollinearity analysis complete!")