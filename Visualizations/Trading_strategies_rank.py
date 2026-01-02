# Read the Excel file
file_path = '/kaggle/input/excel-files/Trdaing strategies ablation.xlsx'
df = pd.read_excel(file_path)

# Select the columns you want to test
columns_to_test = ['TPL+MSFE', 'SGA', 'NPV', 'NPMM', 'OTL', 'CTL', 'UP/DOWN', 
                   'Momentum', 'Mean Reversion', 'Breakout', 'Buy & Hold']

# Create a subset with only the columns to test
data = df[columns_to_test].dropna()

# Initialize matrix for p-values and median differences
n_cols = len(columns_to_test)
p_value_matrix = np.ones((n_cols, n_cols))
median_diff_matrix = np.zeros((n_cols, n_cols))

# Perform pairwise Wilcoxon signed-rank tests
for i, col1 in enumerate(columns_to_test):
    for j, col2 in enumerate(columns_to_test):
        if i < j:
            statistic, p_value = wilcoxon(data[col1], data[col2])
            p_value_matrix[i, j] = p_value
            p_value_matrix[j, i] = p_value
            
            # Calculate median difference to determine direction
            median_diff = np.median(data[col1] - data[col2])
            median_diff_matrix[i, j] = median_diff
            median_diff_matrix[j, i] = -median_diff

# **FIXED PART: Calculate ranking with NaN checks**
# Count how many strategies each strategy significantly outperforms
win_count = np.zeros(n_cols)

for i in range(n_cols):
    for j in range(n_cols):
        if i < j:  # Upper triangle only
            # **FIX: Check for NaN and valid values before comparison**
            p_val = p_value_matrix[i, j]
            median_diff = median_diff_matrix[i, j]
            
            # Only proceed if values are valid (not NaN and not inf)
            if not (np.isnan(p_val) or np.isnan(median_diff) or 
                    np.isinf(p_val) or np.isinf(median_diff)):
                
                if p_val < 0.05:  # Significant difference
                    if median_diff > 0:  # Strategy i is better
                        win_count[i] += 1
                    else:  # Strategy j is better
                        win_count[j] += 1

# Create ranking dataframe
ranking_df = pd.DataFrame({
    'Strategy': columns_to_test,
    'Wins': win_count.astype(int),
    'Rank': win_count.argsort()[::-1].argsort() + 1
}).sort_values('Wins', ascending=False)

# Create a bar chart for the ranking
plt.figure(figsize=(12, 6))
colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(ranking_df)))
bars = plt.bar(range(len(ranking_df)), ranking_df['Wins'], color=colors)
plt.xticks(range(len(ranking_df)), ranking_df['Strategy'], rotation=45, ha='right', fontsize=15)
plt.ylabel('Number of Strategies Significantly \n Outperformed', fontsize=15)
plt.grid(axis='y', alpha=0.5)

# Add value labels on bars
for i, (bar, wins) in enumerate(zip(bars, ranking_df['Wins'])):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{int(wins)}', ha='center', va='bottom', fontsize=15)

plt.tight_layout()
plt.savefig('strategy_ranking.png', dpi=300, bbox_inches='tight')
plt.show()

# Print ranking table
print("\n" + "="*60)
print("STRATEGY RANKING (Based on Upper Diagonal)")
print("="*60)
print(ranking_df.to_string(index=False))
print("\nNote: 'Wins' = number of strategies significantly outperformed (p < 0.05)")
