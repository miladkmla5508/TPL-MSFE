from scipy.stats import wilcoxon

# Read the Excel file
file_path = '/kaggle/input/excel-files/Trdaing strategies ablation.xlsx'
df = pd.read_excel(file_path)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
})

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

# Create combined matrix for visualization
combined_matrix = np.zeros((n_cols, n_cols))
for i in range(n_cols):
    for j in range(n_cols):
        if i > j:  # Lower triangle: use p-values
            combined_matrix[i, j] = p_value_matrix[i, j]
        elif i < j:  # Upper triangle: use direction (if significant)
            if p_value_matrix[i, j] < 0.05:
                combined_matrix[i, j] = np.sign(median_diff_matrix[i, j])
            else:
                combined_matrix[i, j] = 0
        else:  # Diagonal
            combined_matrix[i, j] = np.nan

# **FIXED PART: Create combined annotations with NaN checks**
annot_array = np.empty_like(combined_matrix, dtype=object)
for i in range(n_cols):
    for j in range(n_cols):
        if i > j:  # Lower triangle: p-values
            val = p_value_matrix[i, j]
            if val < 0.0001:
                annot_array[i, j] = f'{val:.0e}'
            else:
                annot_array[i, j] = f'{val:.4f}'
        elif i < j:  # Upper triangle: direction
            # **FIX 1: Check for NaN before comparison**
            if np.isnan(combined_matrix[i, j]):
                annot_array[i, j] = '-'
            elif combined_matrix[i, j] > 0:
                annot_array[i, j] = f'{columns_to_test[i]}'
            elif combined_matrix[i, j] < 0:
                annot_array[i, j] = f'{columns_to_test[j]}'
            else:
                annot_array[i, j] = 'ns'
        else:  # Diagonal
            annot_array[i, j] = '-'

# Create the combined heatmap
plt.figure(figsize=(16, 14))

from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.patches as mpatches

combined_df = pd.DataFrame(combined_matrix, index=columns_to_test, columns=columns_to_test)

# Create masks for upper and lower triangles
mask_lower = np.tril(np.ones_like(combined_matrix, dtype=bool), k=-1) == False
mask_upper = np.triu(np.ones_like(combined_matrix, dtype=bool), k=1) == False

# Plot lower triangle (p-values) with RdYlGn_r colormap
ax = sns.heatmap(combined_df, annot=annot_array, fmt='', cmap='RdYlGn_r',
                 mask=mask_lower, vmin=0, vmax=0.05, 
                 cbar_kws={'label': 'P-value'},
                 linewidths=0.5, linecolor='gray', 
                 annot_kws={'fontsize': 15, 'rotation': 45})

# Plot upper triangle (direction) with RdBu_r colormap
sns.heatmap(combined_df, annot=annot_array, fmt='', cmap='RdBu_r',
            mask=mask_upper, center=0, vmin=-1, vmax=1,
            cbar_kws={'label': 'Direction (Upper Triangle)'},
            linewidths=0.5, linecolor='gray', 
            annot_kws={'fontsize': 13, 'rotation': -45},
            cbar=False)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('wilcoxon_combined_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# **FIX 2: Summary statistics with NaN handling**
print("\nSummary:")
print(f"Total comparisons: {int(n_cols * (n_cols - 1) / 2)}")

# Use np.nansum to ignore NaN values when counting
lower_triangle_pvalues = np.tril(p_value_matrix, k=-1)
# Filter out zeros (diagonal) and count significant p-values
sig_count = np.sum((lower_triangle_pvalues > 0) & (lower_triangle_pvalues < 0.05))

print(f"Significant differences (p < 0.05): {sig_count}")
print(f"Non-significant differences: {int(n_cols * (n_cols - 1) / 2) - sig_count}")
