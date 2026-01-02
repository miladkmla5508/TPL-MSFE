import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data from Excel file
excel = pd.read_excel('/kaggle/input/excel-files/Models computational metrics ablation.xlsx')  # Or use pd.read_csv() if you convert to CSV

# Create figure with subplots - 2x2 layout
fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=100)
#fig.suptitle('Model Performance Metrics Comparison', fontsize=24, fontweight='bold', y=0.98)

# Flatten axes for easier iteration
axes = axes.flatten()

# Plot 1: Accuracy
ax1 = axes[0]
sns.boxplot(
    data=excel,
    x='Model',
    y='accuracy',
    width=0.5,
    palette='tab10',
    ax=ax1
)
ax1.set_title('Accuracy', fontsize=20, fontweight='bold', pad=10)
ax1.set_xlabel('', fontsize=16)
ax1.set_ylabel('Accuracy', fontsize=16)
ax1.tick_params(axis='x', rotation=60, labelsize=16)
ax1.tick_params(axis='y', labelsize=16)
ax1.grid(True, alpha=0.3, linestyle='--')

# Plot 2: MCC
ax2 = axes[1]
sns.boxplot(
    data=excel,
    x='Model',
    y='MCC',
    width=0.5,
    palette='tab10',
    ax=ax2
)
ax2.set_title("Matthew's Correlation Coefficient (MCC)", fontsize=20, fontweight='bold', pad=10)
ax2.set_xlabel('', fontsize=16)
ax2.set_ylabel('MCC', fontsize=16)
ax2.tick_params(axis='x', rotation=60, labelsize=16)
ax2.tick_params(axis='y', labelsize=16)
ax2.grid(True, alpha=0.3, linestyle='--')

# Plot 3: Precision (BUY vs SELL)
ax3 = axes[2]
precision_data = pd.DataFrame({
    'Model': excel['Model'].tolist() * 2,
    'Precision': excel['precision buy'].tolist() + excel['precision sell'].tolist(),
    'Label': ['BUY'] * len(excel) + ['SELL'] * len(excel)
})

sns.boxplot(
    data=precision_data,
    x='Model',
    y='Precision',
    hue='Label',
    width=0.6,
    palette=['#3498db', '#e74c3c'],
    ax=ax3
)
ax3.set_title('Precision: BUY vs SELL', fontsize=20, fontweight='bold', pad=10)
ax3.set_xlabel('', fontsize=16)
ax3.set_ylabel('Precision', fontsize=16)
ax3.tick_params(axis='x', rotation=60, labelsize=16)
ax3.tick_params(axis='y', labelsize=16)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(title='', fontsize=14, loc='best')

# Plot 4: Recall (BUY vs SELL)
ax4 = axes[3]
recall_data = pd.DataFrame({
    'Model': excel['Model'].tolist() * 2,
    'Recall': excel['recall buy'].tolist() + excel['recall sell'].tolist(),
    'Label': ['BUY'] * len(excel) + ['SELL'] * len(excel)
})

sns.boxplot(
    data=recall_data,
    x='Model',
    y='Recall',
    hue='Label',
    width=0.6,
    palette=['#3498db', '#e74c3c'],
    ax=ax4
)
ax4.set_title('Recall: BUY vs SELL', fontsize=20, fontweight='bold', pad=10)
ax4.set_xlabel('', fontsize=16)
ax4.set_ylabel('Recall', fontsize=16)
ax4.tick_params(axis='x', rotation=60, labelsize=16)
ax4.tick_params(axis='y', labelsize=16)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.legend(title='', fontsize=14, loc='best')

# Adjust layout
plt.tight_layout()
plt.show()
