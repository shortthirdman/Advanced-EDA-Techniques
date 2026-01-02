import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load your dataset
df = pd.read_csv('sample_eda_data.csv')
# Function for advanced univariate analysis
def advanced_univariate_analysis(df):
    """
    Perform comprehensive univariate analysis on numerical columns
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Create a summary statistics dataframe
    summary_stats = pd.DataFrame()
    
    for col in numeric_cols:
        summary_stats[col] = {
            'Count': df[col].count(),
            'Mean': df[col].mean(),
            'Median': df[col].median(),
            'Std Dev': df[col].std(),
            'Min': df[col].min(),
            'Q1': df[col].quantile(0.25),
            'Q3': df[col].quantile(0.75),
            'Max': df[col].max(),
            'IQR': df[col].quantile(0.75) - df[col].quantile(0.25),
            'Skewness': df[col].skew(),
            'Kurtosis': df[col].kurtosis(),
            'Variance': df[col].var(),
            'Range': df[col].max() - df[col].min(),
            'CV': df[col].std() / df[col].mean() if df[col].mean() != 0 else np.nan
        }
    
    return summary_stats.T

# Execute analysis
summary = advanced_univariate_analysis(df)
print("Advanced Univariate Statistics:")
print(summary.head(10))

# Test for normality
print("\n" + "="*70)
print("NORMALITY TESTS")
print("="*70)

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols[:5]:  # Test first 5 columns
    # Shapiro-Wilk test
    stat_sw, p_value_sw = stats.shapiro(df[col].dropna())
    
    # Anderson-Darling test
    result_ad = stats.anderson(df[col].dropna())
    
    # D'Agostino-Pearson test
    stat_dp, p_value_dp = stats.normaltest(df[col].dropna())
    
    print(f"\n{col}:")
    print(f"  Shapiro-Wilk: statistic={stat_sw:.4f}, p-value={p_value_sw:.4f}")
    print(f"  Anderson-Darling: statistic={result_ad.statistic:.4f}")
    print(f"  D'Agostino-Pearson: statistic={stat_dp:.4f}, p-value={p_value_dp:.4f}")
    
    # Interpretation
    if p_value_sw > 0.05:
        print(f"  ✓ {col} appears normally distributed (p > 0.05)")
    else:
        print(f"  ✗ {col} is NOT normally distributed (p < 0.05)")

# Visualize distributions with advanced plots
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()
for idx, col in enumerate(numeric_cols[:6]):
    # Histogram with KDE
    axes[idx].hist(df[col].dropna(), bins=30, density=True, alpha=0.7, 
                   color='skyblue', edgecolor='black', label='Histogram')
    
    # Overlay KDE
    df[col].dropna().plot(kind='kde', ax=axes[idx], color='red', 
                          linewidth=2, label='KDE', secondary_y=False)
    
    # Add statistics to plot
    skewness = df[col].skew()
    kurtosis = df[col].kurtosis()
    axes[idx].set_title(f'{col}\nSkewness: {skewness:.3f}, Kurtosis: {kurtosis:.3f}')
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Density')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('univariate_distributions.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ Univariate analysis complete! Distribution plots saved.")