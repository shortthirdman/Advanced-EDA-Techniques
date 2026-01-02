from scipy.stats import f_oneway, kruskal, chi2_contingency, ttest_ind, mannwhitneyu

def statistical_testing_framework(df):
    """
    Perform comprehensive statistical testing
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    print("="*70)
    print("STATISTICAL TESTING FRAMEWORK")
    print("="*70)
    
    # 1. Categorical vs Numerical
    print("\n1. CATEGORICAL vs NUMERICAL ANALYSIS")
    print("-"*70)
    
    for cat_col in categorical_cols[:1]:  # Test first categorical column
        for num_col in numeric_cols[:3]:  # Test first 3 numerical columns
            if df[cat_col].nunique() <= 10:  # Only if reasonable number of categories
                groups = [group[num_col].dropna().values 
                         for name, group in df.groupby(cat_col)]
                
                # Check normality of each group
                all_normal = all(stats.shapiro(group) > 0.05 for group in groups)
                
                if all_normal:
                    # Use ANOVA (parametric)
                    f_stat, p_value = f_oneway(*groups)
                    test_name = "ANOVA (F-test)"
                else:
                    # Use Kruskal-Wallis (non-parametric)
                    stat, p_value = kruskal(*groups)
                    test_name = "Kruskal-Wallis"
                
                significance = "***" if p_value < 0.001 else ("**" if p_value < 0.01 
                               else ("*" if p_value < 0.05 else "ns"))
                
                print(f"\n{cat_col} vs {num_col}:")
                print(f"  Test: {test_name}")
                print(f"  p-value: {p_value:.6f} {significance}")
                print(f"  Interpretation: {'Significant' if p_value < 0.05 else 'Not significant'}")
    
    # 2. Categorical vs Categorical
    print("\n\n2. CATEGORICAL vs CATEGORICAL ANALYSIS")
    print("-"*70)
    
    categorical_cols_list = list(categorical_cols)
    if len(categorical_cols_list) >= 2:
        col1, col2 = categorical_cols_list, categorical_cols_list
        
        contingency_table = pd.crosstab(df[col1], df[col2])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        print(f"\n{col1} vs {col2}:")
        print(f"  Chi-square statistic: {chi2:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Degrees of freedom: {dof}")
        print(f"  Interpretation: {'Dependent' if p_value < 0.05 else 'Independent'}")
        
        # Display contingency table
        print(f"\n  Contingency Table:\n{contingency_table}")
    
    # 3. Numerical vs Numerical
    print("\n\n3. NUMERICAL vs NUMERICAL CORRELATION TESTING")
    print("-"*70)
    
    for i, col1 in enumerate(numeric_cols[:3]):
        for col2 in numeric_cols[i+1:4]:
            r_pearson, p_pearson = pearsonr(df[col1].dropna(), df[col2].dropna())
            r_spearman, p_spearman = spearmanr(df[col1].dropna(), df[col2].dropna())
            
            print(f"\n{col1} vs {col2}:")
            print(f"  Pearson r: {r_pearson:.4f} (p={p_pearson:.6f})")
            print(f"  Spearman ρ: {r_spearman:.4f} (p={p_spearman:.6f})")

# Execute statistical testing
statistical_testing_framework(df)
print("\n✓ Statistical testing complete!")