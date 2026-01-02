class AdvancedEDAFramework:
    """
    Comprehensive EDA framework automating all analyses
    """
    
    def __init__(self, df, target_col=None):
        self.df = df.copy()
        self.target_col = target_col
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.categorical_cols = df.select_dtypes(include=['object']).columns
        self.results = {}
        
    def data_quality_report(self):
        """Generate data quality summary"""
        print("\n" + "="*70)
        print("DATA QUALITY REPORT")
        print("="*70)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Total Cells: {self.df.shape * self.df.shape}")
        
        # Missing values
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        print(f"\nMissing Values:")
        for col, count in missing[missing > 0].items():
            print(f"  {col}: {count} ({missing_pct[col]:.2f}%)")
        
        # Duplicates
        duplicates = self.df.duplicated().sum()
        print(f"\nDuplicate Rows: {duplicates} ({(duplicates/len(self.df))*100:.2f}%)")
        
        # Data types
        print(f"\nData Types:")
        print(self.df.dtypes.value_counts())
        
        self.results['data_quality'] = {
            'missing': missing,
            'duplicates': duplicates
        }
    
    def univariate_analysis(self):
        """Perform univariate analysis"""
        print("\n" + "="*70)
        print("UNIVARIATE ANALYSIS")
        print("="*70)
        
        summary_stats = pd.DataFrame()
        
        for col in self.numeric_cols:
            summary_stats[col] = {
                'Mean': self.df[col].mean(),
                'Median': self.df[col].median(),
                'Std': self.df[col].std(),
                'Min': self.df[col].min(),
                'Max': self.df[col].max(),
                'Skewness': self.df[col].skew(),
                'Kurtosis': self.df[col].kurtosis(),
                'Missing %': (self.df[col].isnull().sum() / len(self.df)) * 100
            }
        
        print("\nNumerical Features Summary:")
        print(summary_stats.T.round(4))
        
        self.results['univariate'] = summary_stats
    
    def multivariate_analysis(self):
        """Perform multivariate analysis"""
        print("\n" + "="*70)
        print("MULTIVARIATE ANALYSIS - CORRELATIONS")
        print("="*70)
        
        numeric_df = self.df[self.numeric_cols]
        pearson_corr = numeric_df.corr(method='pearson')
        
        # Get top correlations
        corr_pairs = []
        for i in range(len(pearson_corr.columns)):
            for j in range(i+1, len(pearson_corr.columns)):
                corr_pairs.append({
                    'Var1': pearson_corr.columns[i],
                    'Var2': pearson_corr.columns[j],
                    'Correlation': pearson_corr.iloc[i, j]
                })
        
        top_corr = pd.DataFrame(corr_pairs).sort_values('Correlation', 
                                                         key=abs, ascending=False).head(10)
        
        print("\nTop 10 Correlations:")
        print(top_corr.to_string(index=False))
        
        self.results['multivariate'] = top_corr
    
    def generate_full_report(self):
        """Generate complete EDA report"""
        print("\n" + "#"*70)
        print("# ADVANCED EDA FRAMEWORK - COMPLETE ANALYSIS")
        print("#"*70)
        
        self.data_quality_report()
        self.univariate_analysis()
        self.multivariate_analysis()
        
        print("\n" + "#"*70)
        print("# ANALYSIS COMPLETE")
        print("#"*70)

# Usage
eda_framework = AdvancedEDAFramework(df, target_col='Target')
eda_framework.generate_full_report()
print("\n✓ Advanced EDA framework execution complete!")