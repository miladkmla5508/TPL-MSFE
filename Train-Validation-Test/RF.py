import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             classification_report, roc_auc_score)
from itertools import product
from operator import itemgetter

# ============================================================================
# SUPPRESS WARNINGS
# ============================================================================
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=Warning, module='scipy')
warnings.filterwarnings('ignore', category=Warning, module='ReliefF')
warnings.filterwarnings('ignore', category=Warning, module='mlxtend')
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# START TIMER
# ============================================================================
start = time.time()

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
initial_window = 1260  # Initial training window size (5 years of trading days)
test_window = 252      # Test window size (1 year of trading days)
num_windows = 7        # Number of sliding windows
num_features = 50      # Number of features to select
topk = 200             # Initial feature selection threshold
riskfree = 0.02        # Risk-free rate for performance metrics

# Walk-Forward Cross-Validation Parameters
cv_window = 252        # Size of each CV fold (1 year)

# Hyperparameter Grid for Random Forest
param_grid = {
    'n_estimators': [20, 40, 60, 80, 100],  # Number of trees
    'criterion': ['gini', 'entropy'],  # Splitting criteria
    'max_depth': [2, 3, 4, 5],  # Tree depth
    'min_samples_split': [1, 2, 3, 4],  # Min samples to split
    'min_samples_leaf': [1, 2, 3, 4],  # Min samples in leaf
    'max_features': ['sqrt'],  # Features per split
    'bootstrap': [True, False],  # Bootstrap sampling
    'oob_score': [True],  # Out-of-bag score
    'max_samples': [0.6, 0.7, 0.8],  # Bootstrap sample size
    #'class_weight': [None, 'balanced', 'balanced_subsample'],  # Class weights
    'random_state': [0],  # Random seed
    #'min_impurity_decrease': [0.0, 0.001, 0.01],  # Impurity decrease threshold
    #'ccp_alpha': [0.0, 0.001, 0.01],  # Complexity parameter for pruning
    #'max_leaf_nodes': [None, 10, 20, 50, 100]  # Maximum leaf nodes
}

# Trading parameters
cb = 0.0075    # Buy commission
cs = 0.0075    # Sell commission

# Initialize scaler (optional for RF, but kept for consistency)
scaler = StandardScaler()

# ============================================================================
# EXTRACT FEATURE NAMES
# ============================================================================
list_features = list(df.loc[:, 'SMA_6':'UI_20'].columns)
print(f"Total features available: {len(list_features)}")

# ============================================================================
# INITIALIZE STORAGE VARIABLES
# ============================================================================
yptt = []              # Store predictions for each window
modelval_RF = []       # Store trading system values over time
y_test_window_RF = []  # Store all test predictions concatenated
feature_importance_history = []  # Store feature importance across windows
oob_scores_history = []         # Store out-of-bag scores
best_params_history = []  # Store best parameters from CV for each window
cv_results_history = []  # Store CV results for each window
cv_folds_history = []    # Track number of CV folds used per window

# ============================================================================
# SLIDING WINDOW TRAINING AND PREDICTION
# ============================================================================
print("\n" + "="*80)
print("STARTING SLIDING WINDOW ANALYSIS WITH RANDOM FOREST AND DYNAMIC CROSS-VALIDATION")
print("="*80 + "\n")

for window_idx in range(num_windows):
    print(f"\n{'='*60}")
    print(f"PROCESSING WINDOW {window_idx + 1}/{num_windows}")
    print(f"{'='*60}\n")
    
    # Calculate window boundaries
    offset = test_window * window_idx
    train_end = initial_window + offset
    test_start = train_end
    test_end = test_start + test_window
    
    # ========================================================================
    # DATA PREPARATION
    # ========================================================================
    X_train_raw = df.drop('LABEL', axis=1)[:train_end]
    X_test_raw = df.drop('LABEL', axis=1)[test_start:test_end]
    y_train = df['LABEL'][:train_end]
    y_test = df['LABEL'][test_start:test_end]
    
    # Scale data (optional for RF)
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
    
    # ========================================================================
    # FEATURE SELECTION - STEP 1: UNIVARIATE METHODS
    # ========================================================================
    print("\n--- Feature Selection Step 1: Univariate Methods ---")
    
    x_main = X_train.copy()
    x_main_raw = X_train_raw.values
    
    # ANOVA F-test
    select_k_best_anova = SelectKBest(f_classif, k=topk)
    select_k_best_anova.fit(x_main, y_train)
    selected_indices_anova = select_k_best_anova.get_support(indices=True)
    selected_features_anova = [list_features[i] for i in selected_indices_anova if i < len(list_features)]
    print(f"ANOVA selected {len(selected_features_anova)} features")
    
    # Mutual Information
    select_k_best_mi = SelectKBest(mutual_info_classif, k=topk)
    select_k_best_mi.fit(x_main, y_train)
    selected_indices_mi = select_k_best_mi.get_support(indices=True)
    selected_features_mi = [list_features[i] for i in selected_indices_mi if i < len(list_features)]
    print(f"Mutual Info selected {len(selected_features_mi)} features")
    
    # Chi-Square test
    if np.any(x_main_raw < 0):
        minmax_scaler = MinMaxScaler()
        x_main_chi2 = minmax_scaler.fit_transform(x_main_raw)
    else:
        x_main_chi2 = x_main_raw
    
    select_k_best_chi2 = SelectKBest(chi2, k=topk)
    select_k_best_chi2.fit(x_main_chi2, y_train)
    selected_indices_chi2 = select_k_best_chi2.get_support(indices=True)
    selected_features_chi2 = [list_features[i] for i in selected_indices_chi2 if i < len(list_features)]
    print(f"Chi-Square selected {len(selected_features_chi2)} features")
    
    # ========================================================================
    # FEATURE SELECTION - STEP 2: INTERSECTION
    # ========================================================================
    print("\n--- Feature Selection Step 2: Finding Common Features ---")
    
    common = list(set(set(selected_features_anova).intersection(selected_features_mi)).intersection(selected_features_chi2))
    print(f"Common features found: {len(common)}")
    
    if len(common) < num_features:
        print(f"WARNING: Only {len(common)} common features found, needed {num_features}")
        common_union = list(set(set(selected_features_anova).union(selected_features_mi)).union(selected_features_chi2))
        print(f"Using union instead: {len(common_union)} features")
        common = common_union[:num_features]
    
    feat_idx = sorted([list_features.index(f) for f in common])
    X_train_common = X_train[:, feat_idx]
    X_test_common = X_test[:, feat_idx]
    
    # ========================================================================
    # FEATURE SELECTION - STEP 3: RELIEFF
    # ========================================================================
    print("\n--- Feature Selection Step 3: ReliefF ---")
    
    try:
        from sklearn_relief import ReliefF
        
        fs = ReliefF(n_features=num_features)
        fs.fit(X_train_common, y_train.values)
        
        if hasattr(fs, 'feature_importances_'):
            feature_scores = fs.feature_importances_
            top_indices = np.argsort(feature_scores)[-num_features:][::-1]
            selected_feature_names_relief = [common[i] for i in top_indices]
        else:
            X_train_df = pd.DataFrame(X_train_common, columns=common).reset_index(drop=True)
            X_train_transformed = fs.transform(X_train_common)
            X_transformed_df = pd.DataFrame(X_train_transformed).reset_index(drop=True)
            
            selected_feature_names_relief = []
            for col_idx in range(X_transformed_df.shape[1]):
                transformed_col = X_transformed_df.iloc[:, col_idx].values
                for original_col in X_train_df.columns:
                    original_values = X_train_df[original_col].values
                    if np.allclose(transformed_col, original_values, rtol=1e-5, atol=1e-8):
                        selected_feature_names_relief.append(original_col)
                        break
        
        print(f"ReliefF selected {len(selected_feature_names_relief)} features")
        
    except Exception as e:
        print(f"ReliefF error: {str(e)}, using top features from common")
        selected_feature_names_relief = common[:num_features]
    
    # ========================================================================
    # FEATURE SELECTION - STEP 4: SFFS
    # ========================================================================
    print("\n--- Feature Selection Step 4: SFFS ---")
    
    try:
        from mlxtend.feature_selection import SequentialFeatureSelector as SFS
        
        X_sffs = df[selected_feature_names_relief][:train_end].values
        y_sffs = y_train.values if hasattr(y_train, 'values') else y_train
        
        # Use default RF for SFFS
        sfs = SFS(
            RandomForestClassifier(n_estimators=50, max_depth=3, random_state=0),
            k_features=(1, 20),
            forward=True,
            floating=True,
            verbose=0,
            scoring='accuracy',
            cv=None,
            n_jobs=-1
        )
        
        sfs.fit(X_sffs, y_sffs)
        final_features = [selected_feature_names_relief[i] for i in sfs.k_feature_idx_]
        print(f"SFFS selected {len(final_features)} features")
        
    except ImportError:
        print("mlxtend not installed, using top 20 from ReliefF")
        final_features = selected_feature_names_relief[:20]
    
    # ========================================================================
    # CREATE FINAL DATASET WITH SELECTED FEATURES
    # ========================================================================
    df_novel = pd.concat([df[final_features], df['LABEL']], axis=1)
    
    X_train_final = df_novel.drop('LABEL', axis=1)[:train_end]
    X_test_final = df_novel.drop('LABEL', axis=1)[test_start:test_end]
    y_train_final = df_novel['LABEL'][:train_end]
    y_test_final = df_novel['LABEL'][test_start:test_end]
    
    # ========================================================================
    # DYNAMIC CROSS-VALIDATION SETUP
    # ========================================================================
    print("\n" + "="*60)
    print(f"DYNAMIC CROSS-VALIDATION - WINDOW {window_idx + 1}")
    print("="*60 + "\n")
    
    # Calculate maximum number of CV folds based on available training data
    max_possible_folds = len(X_train_final) // cv_window
    cv_folds = max_possible_folds
    
    if cv_folds < 2:
        print(f"WARNING: Not enough data for CV. Available: {len(X_train_final)} days, needed: {cv_window*2} days")
        cv_folds = 2  # Minimum 2 folds for validation
    
    print(f"Using {cv_folds} CV folds for this window")
    print(f"Training data size: {len(X_train_final)} days ({len(X_train_final)/252:.1f} years)")
    print(f"CV fold size: {cv_window} days (1 year each)")
    
    # Store CV folds info
    cv_folds_history.append(cv_folds)
    
    # ========================================================================
    # RANDOM FOREST PARAMETER OPTIMIZATION
    # ========================================================================
    print("\n" + "="*60)
    print(f"RANDOM FOREST PARAMETER OPTIMIZATION - WINDOW {window_idx + 1}")
    print("="*60 + "\n")
    
    # Generate valid parameter combinations
    param_combinations = []
    keys = list(param_grid.keys())
    for values in product(*[param_grid[k] for k in keys]):
        param_dict = dict(zip(keys, values))
        
        # Filter invalid parameter combinations
        # min_samples_split should be >= 2 * min_samples_leaf for valid trees
        min_samples_split = param_dict['min_samples_split']
        min_samples_leaf = param_dict['min_samples_leaf']
        if min_samples_split < 2 * min_samples_leaf:
            continue
        
        # oob_score only works with bootstrap=True
        if param_dict['oob_score'] and not param_dict['bootstrap']:
            continue
        
        # max_samples only works with bootstrap=True
        if param_dict['max_samples'] is not None and not param_dict['bootstrap']:
            continue
        
        param_combinations.append(param_dict)
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    # Storage for CV results
    cv_results = []
    
    # Test each parameter combination
    for param_idx, params in enumerate(param_combinations):
        fold_accuracies = []
        
        # Perform walk-forward CV
        for fold in range(1, cv_folds):
            train_cv_end = cv_window * fold
            val_cv_start = train_cv_end
            val_cv_end = val_cv_start + cv_window
            
            if val_cv_end > len(X_train_final):
                continue
                
            # Extract CV data
            X_cv_train = X_train_final.iloc[:train_cv_end]
            y_cv_train = y_train_final.iloc[:train_cv_end]
            X_cv_val = X_train_final.iloc[val_cv_start:val_cv_end]
            y_cv_val = y_train_final.iloc[val_cv_start:val_cv_end]
            
            # Scale CV data
            cv_scaler = StandardScaler()
            X_cv_train_scaled = cv_scaler.fit_transform(X_cv_train)
            X_cv_val_scaled = cv_scaler.transform(X_cv_val)
            
            try:
                model = RandomForestClassifier(**params)
                model.fit(X_cv_train_scaled, y_cv_train)
                y_cv_pred = model.predict(X_cv_val_scaled)
                fold_acc = accuracy_score(y_cv_val, y_cv_pred)
                fold_accuracies.append(fold_acc)
                
            except Exception as e:
                print(f"  Warning: RF failed for params {params} on fold {fold}: {str(e)}")
                fold_accuracies.append(0.0)
        
        # Only consider combinations that were tested on all folds
        if len(fold_accuracies) > 0:
            mean_acc = np.mean(fold_accuracies)
            std_acc = np.std(fold_accuracies)
            
            # Store results
            cv_results.append({
                'params': params,
                'fold_accuracies': fold_accuracies,
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'num_folds': len(fold_accuracies)
            })
        
        # Progress update - RF is slower than DT, so update less frequently
        if (param_idx + 1) % 5 == 0:
            print(f"  Tested {param_idx + 1}/{len(param_combinations)} combinations...")
    
    # Find best parameters
    if cv_results:
        best_result = max(cv_results, key=lambda x: x['mean_accuracy'])
        best_params = best_result['params']
        best_mean_acc = best_result['mean_accuracy']
        best_fold_accs = best_result['fold_accuracies']
        
        print(f"\n{'='*60}")
        print(f"BEST RANDOM FOREST PARAMETERS FOUND:")
        print(f"{'='*60}")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"\nCross-Validation Accuracies by Fold:")
        for fold_idx, acc in enumerate(best_fold_accs):
            print(f"  Fold {fold_idx + 1}: {acc:.4f}")
        print(f"Mean CV Accuracy: {best_mean_acc:.4f} (+/- {best_result['std_accuracy']:.4f})")
        print(f"Number of CV folds used: {best_result['num_folds']}")
        print(f"{'='*60}\n")
    else:
        print("ERROR: No valid CV results. Using safe default parameters.")
        best_params = {
            'n_estimators': 100,
            'criterion': 'gini',
            'max_depth': 4,
            'min_samples_split': 10,
            'min_samples_leaf': 1,
            'max_features': 20,
            'bootstrap': True,
            'oob_score': False,
            'max_samples': None,
            'class_weight': None,
            'random_state': 0,
            'min_impurity_decrease': 0.0,
            'ccp_alpha': 0.0,
            'max_leaf_nodes': None
        }
        best_mean_acc = 0.0
        best_fold_accs = []
    
    # Store CV results for this window
    best_params_history.append(best_params)
    cv_results_history.append(cv_results)
    
    # ========================================================================
    # VISUALIZE CV RESULTS
    # ========================================================================
    if best_fold_accs:
        fig, ax = plt.subplots(figsize=(7, 5))
        fold_indices = np.arange(1, len(best_fold_accs) + 1)
        
        ax.scatter(fold_indices, best_fold_accs, s=50, c='green', edgecolors='black', zorder=3)
        ax.plot(fold_indices, best_fold_accs, 'o-', linewidth=1.5, markersize=0, color='gray', alpha=0.5)
        ax.axhline(y=best_mean_acc, color='blue', linestyle='--', alpha=0.7, linewidth=1.5, 
                   label=f'Mean Accuracy: {best_mean_acc:.4f}')
        
        ax.set_xlabel('CV Fold', fontsize=15)
        ax.set_ylabel('Accuracy', fontsize=15)
        ax.set_title(f'Window {window_idx + 1}: RF CV Results ({len(best_fold_accs)} folds)', fontsize=18)
        ax.set_xticks(fold_indices)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        plt.tight_layout()
        plt.show()  
    
    # ========================================================================
    # TRAIN FINAL RANDOM FOREST WITH OPTIMIZED PARAMETERS
    # ========================================================================
    print("\n--- Training Final Random Forest with Optimized Parameters ---")
    
    # Scale the full training data
    final_scaler = StandardScaler()
    X_train_final_scaled = final_scaler.fit_transform(X_train_final)
    X_test_final_scaled = final_scaler.transform(X_test_final)
    
    # Train Random Forest with best parameters
    try:
        model_rf = RandomForestClassifier(**best_params)
        model_rf.fit(X_train_final_scaled, y_train_final)
        print(f"Random Forest trained with {best_params['n_estimators']} trees")
        
    except Exception as e:
        print(f"Warning: RF failed to train with best params: {str(e)}")
        print("Using fallback parameters...")
        
        # Fallback to simple RF
        fallback_params = {
            'n_estimators': 100,
            'criterion': 'gini',
            'max_depth': 4,
            'min_samples_split': 10,
            'min_samples_leaf': 1,
            'max_features': 20,
            'bootstrap': True,
            'oob_score': False,
            'max_samples': None,
            'class_weight': None,
            'random_state': 0,
            'min_impurity_decrease': 0.0,
            'ccp_alpha': 0.0,
            'max_leaf_nodes': None,
            'n_jobs': -1,
            'verbose': 0
        }
        
        model_rf = RandomForestClassifier(**fallback_params)
        model_rf.fit(X_train_final_scaled, y_train_final)
    
    # Make predictions
    y_pred_test = model_rf.predict(X_test_final_scaled)
    y_pred_train = model_rf.predict(X_train_final_scaled)
    y_pred_prob_test = model_rf.predict_proba(X_test_final_scaled)[:, 1]
    
    # Evaluate model
    test_accuracy = accuracy_score(y_test_final, y_pred_test)
    train_accuracy = accuracy_score(y_train_final, y_pred_train)
    
    try:
        auc_roc = roc_auc_score(y_test_final, y_pred_prob_test)
        print(f"AUC-ROC Score: {auc_roc:.4f}")
    except:
        auc_roc = None
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test_final, y_pred_test)}")
    
    # Store feature importance
    if hasattr(model_rf, 'feature_importances_') and len(model_rf.feature_importances_) > 0:
        feature_importance = dict(zip(final_features, model_rf.feature_importances_))
        feature_importance_history.append({
            'features': final_features,
            'importances': model_rf.feature_importances_,
            'best_params': best_params,
            'window_idx': window_idx
        })
    
    # Store OOB score if enabled
    if best_params.get('oob_score', False) and hasattr(model_rf, 'oob_score_'):
        oob_scores_history.append({
            'window': window_idx,
            'oob_score': model_rf.oob_score_,
            'best_params': best_params
        })
        print(f"Out-of-Bag Score: {model_rf.oob_score_:.4f}")
    
    # Store predictions
    yptt.append(y_pred_test)
    
    # Display top 10 features
    print("\nTop 10 most important features:")
    if hasattr(model_rf, 'feature_importances_') and len(model_rf.feature_importances_) > 0:
        feature_importance_list = list(zip(final_features, model_rf.feature_importances_))
        feature_importance_list.sort(key=lambda x: x[1], reverse=True)
        for feature, importance in feature_importance_list[:10]:
            print(f"  {feature}: {importance:.4f}")

# ============================================================================
# CONSOLIDATE PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("CONSOLIDATING PREDICTIONS")
print("="*80 + "\n")

for i in range(num_windows):
    y_test_window_RF.extend(yptt[i])

print(f"Total predictions: {len(y_test_window_RF)}")

# ============================================================================
# CV FOLDS ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("CROSS-VALIDATION FOLDS ANALYSIS")
print("="*80 + "\n")

if cv_folds_history:
    fig, ax = plt.subplots(figsize=(10, 6))
    windows = np.arange(1, num_windows + 1)
    
    ax.bar(windows, cv_folds_history, color='darkred', edgecolor='black', alpha=0.7)
    ax.plot(windows, cv_folds_history, 'o-', color='red', linewidth=2, markersize=8)
    
    ax.set_xlabel('Window', fontsize=12)
    ax.set_ylabel('Number of CV Folds', fontsize=12)
    ax.set_title('Number of Cross-Validation Folds per Window (Random Forest)', fontsize=14)
    ax.set_xticks(windows)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(cv_folds_history):
        ax.text(i + 1, v + 0.1, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\nCV Folds Summary for Random Forest:")
    for window_idx, num_folds in enumerate(cv_folds_history):
        training_years = (initial_window + test_window * window_idx) / 252
        print(f"Window {window_idx + 1}: {training_years:.1f} years training data → {num_folds} CV folds")

# ============================================================================
# OVERALL MODEL PERFORMANCE
# ============================================================================
print("\n" + "="*80)
print("OVERALL MODEL PERFORMANCE WITH PARAMETER OPTIMIZATION AND DYNAMIC CV")
print("="*80 + "\n")

y_true = df['LABEL'][initial_window:]
conf_matrix = confusion_matrix(y_true, y_test_window_RF)
acc = accuracy_score(y_true, y_test_window_RF)
precision = precision_score(y_true, y_test_window_RF)
recall = recall_score(y_true, y_test_window_RF)
f1 = f1_score(y_true, y_test_window_RF)
mcc = matthews_corrcoef(y_true, y_test_window_RF)

print(f"Confusion Matrix:\n{conf_matrix}")
print(f"\nAccuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")
print(f"\n{classification_report(y_true, y_test_window_RF, digits=4)}")

# ============================================================================
# VISUALIZATION: CONFUSION MATRIX
# ============================================================================
group_names = ['True Sell', 'False Buy', 'False Sell', 'True Buy']
group_counts = [f"{value:0.0f}" for value in conf_matrix.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
labels = np.asarray(labels).reshape(2, 2)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=labels, linewidths=1, fmt='', 
            cmap='Reds', annot_kws={"size": 14})
ax.set_xlabel('Predicted Labels', fontsize=15)
ax.set_ylabel('Actual Labels', fontsize=15)
ax.set_title('Random Forest - Confusion Matrix (With Dynamic CV)', fontsize=16)
plt.tight_layout()
plt.show()

# ============================================================================
# RANDOM FOREST PARAMETER OPTIMIZATION SUMMARY WITH DYNAMIC CV
# ============================================================================
print("\n" + "="*80)
print("RANDOM FOREST PARAMETER OPTIMIZATION SUMMARY WITH DYNAMIC CV")
print("="*80 + "\n")

print("Best Parameters Selected for Each Window:")
print("-" * 60)

n_estimators_stats = {}
criterion_stats = {}
max_depth_stats = {}
bootstrap_stats = {}
max_features_stats = {}
oob_score_stats = {}
class_weight_stats = {}

for window_idx, params in enumerate(best_params_history):
    print(f"\nWindow {window_idx + 1}:")
    print(f"  n_estimators: {params.get('n_estimators', 'N/A')}")
    print(f"  criterion: {params.get('criterion', 'N/A')}")
    print(f"  max_depth: {params.get('max_depth', 'N/A')}")
    print(f"  bootstrap: {params.get('bootstrap', 'N/A')}")
    print(f"  max_features: {params.get('max_features', 'N/A')}")
    print(f"  oob_score: {params.get('oob_score', 'N/A')}")
    print(f"  class_weight: {params.get('class_weight', 'N/A')}")
    print(f"  CV folds used: {cv_folds_history[window_idx]}")
    
    # Collect statistics
    n_estimators = params.get('n_estimators', 'N/A')
    criterion = params.get('criterion', 'N/A')
    max_depth = params.get('max_depth', 'N/A')
    bootstrap = params.get('bootstrap', 'N/A')
    max_features = params.get('max_features', 'N/A')
    oob_score = params.get('oob_score', 'N/A')
    class_weight = params.get('class_weight', 'N/A')
    
    n_estimators_stats[n_estimators] = n_estimators_stats.get(n_estimators, 0) + 1
    criterion_stats[criterion] = criterion_stats.get(criterion, 0) + 1
    max_depth_stats[max_depth] = max_depth_stats.get(max_depth, 0) + 1
    bootstrap_stats[bootstrap] = bootstrap_stats.get(bootstrap, 0) + 1
    max_features_stats[max_features] = max_features_stats.get(max_features, 0) + 1
    oob_score_stats[oob_score] = oob_score_stats.get(oob_score, 0) + 1
    class_weight_stats[class_weight] = class_weight_stats.get(class_weight, 0) + 1

# Print statistics
print("\n" + "="*60)
print("RANDOM FOREST PARAMETER SELECTION STATISTICS WITH DYNAMIC CV:")
print("="*60)

print("\nNumber of Trees Selection:")
for n_est, count in sorted(n_estimators_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  n_estimators={n_est}: {count} windows ({percentage:.1f}%)")

print("\nSplitting Criterion Selection:")
for criterion, count in sorted(criterion_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  {criterion}: {count} windows ({percentage:.1f}%)")

print("\nMaximum Depth Selection:")
for depth, count in sorted(max_depth_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    depth_str = 'None (unlimited)' if depth is None else str(depth)
    print(f"  {depth_str}: {count} windows ({percentage:.1f}%)")

print("\nBootstrap Sampling Selection:")
for bootstrap, count in sorted(bootstrap_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    bootstrap_str = 'Enabled' if bootstrap else 'Disabled'
    print(f"  {bootstrap_str}: {count} windows ({percentage:.1f}%)")

print("\nMaximum Features per Split Selection:")
for features, count in sorted(max_features_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    features_str = 'None (all)' if features is None else str(features)
    print(f"  {features_str}: {count} windows ({percentage:.1f}%)")

print("\nOut-of-Bag Score Selection:")
for oob, count in sorted(oob_score_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    oob_str = 'Enabled' if oob else 'Disabled'
    print(f"  {oob_str}: {count} windows ({percentage:.1f}%)")

print("\nClass Weight Selection:")
for weight, count in sorted(class_weight_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    weight_str = str(weight) if weight is not None else 'None'
    print(f"  {weight_str}: {count} windows ({percentage:.1f}%)")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Number of trees
n_ests, n_est_counts = zip(*sorted(n_estimators_stats.items(), key=lambda x: x[1], reverse=True))
axes[0, 0].bar([str(n) for n in n_ests], n_est_counts, color='crimson')
axes[0, 0].set_xlabel('Number of Trees', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Number of Trees Selection', fontsize=14)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Criterion selection
criteria, criterion_counts = zip(*sorted(criterion_stats.items(), key=lambda x: x[1], reverse=True))
axes[0, 1].bar(criteria, criterion_counts, color='firebrick')
axes[0, 1].set_xlabel('Splitting Criterion', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('Splitting Criterion Selection', fontsize=14)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Max depth selection
depths, depth_counts = zip(*sorted(max_depth_stats.items(), key=lambda x: x[1], reverse=True))
depth_labels = ['None' if d is None else str(d) for d in depths]
axes[0, 2].bar(depth_labels, depth_counts, color='darkred')
axes[0, 2].set_xlabel('Maximum Depth', fontsize=12)
axes[0, 2].set_ylabel('Frequency', fontsize=12)
axes[0, 2].set_title('Maximum Depth Selection', fontsize=14)
axes[0, 2].grid(True, alpha=0.3, axis='y')

# Bootstrap selection
bootstraps, bootstrap_counts = zip(*sorted(bootstrap_stats.items(), key=lambda x: x[1], reverse=True))
bootstrap_labels = ['Enabled' if b else 'Disabled' for b in bootstraps]
axes[1, 0].bar(bootstrap_labels, bootstrap_counts, color='indianred')
axes[1, 0].set_xlabel('Bootstrap Sampling', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Bootstrap Sampling Selection', fontsize=14)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Max features selection
features, feature_counts = zip(*sorted(max_features_stats.items(), key=lambda x: x[1], reverse=True))
feature_labels = ['None' if f is None else str(f) for f in features]
axes[1, 1].bar(feature_labels, feature_counts, color='lightcoral')
axes[1, 1].set_xlabel('Max Features per Split', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('Maximum Features per Split', fontsize=14)
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3, axis='y')

# CV Folds Evolution
windows = np.arange(1, num_windows + 1)
axes[1, 2].plot(windows, cv_folds_history, 'o-', linewidth=2, markersize=8, color='darkred')
axes[1, 2].set_xlabel('Window', fontsize=12)
axes[1, 2].set_ylabel('Number of CV Folds', fontsize=12)
axes[1, 2].set_title('CV Folds Evolution (Random Forest)', fontsize=14)
axes[1, 2].set_xticks(windows)
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS WITH OPTIMIZATION AND DYNAMIC CV")
print("="*80 + "\n")

if feature_importance_history:
    aggregated_importance = {}
    feature_occurrence = {}
    
    for window_data in feature_importance_history:
        features = window_data['features']
        importances = window_data['importances']
        
        for feature, importance in zip(features, importances):
            if feature not in aggregated_importance:
                aggregated_importance[feature] = []
                feature_occurrence[feature] = 0
            aggregated_importance[feature].append(importance)
            feature_occurrence[feature] += 1
    
    avg_importance = {feature: np.mean(importances) for feature, importances in aggregated_importance.items()}
    top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:15]
    
    print("Top 15 Most Important Features (Average across all windows):")
    for feature, importance in top_features:
        occurrence_pct = (feature_occurrence[feature] / len(feature_importance_history)) * 100
        print(f"  {feature}: {importance:.4f} (in {occurrence_pct:.1f}% of windows)")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    features, importances = zip(*top_features)
    y_pos = np.arange(len(features))
    
    ax1.barh(y_pos, importances, color='darkred')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(features)
    ax1.invert_yaxis()
    ax1.set_xlabel('Average Feature Importance', fontsize=12)
    ax1.set_title('Top 15 Feature Importance - Random Forest', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    occurrence_counts = [feature_occurrence[f] for f in features]
    ax2.barh(y_pos, occurrence_counts, color='lightcoral')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(features)
    ax2.invert_yaxis()
    ax2.set_xlabel('Number of Windows Selected', fontsize=12)
    ax2.set_title('Feature Selection Frequency', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# OUT-OF-BAG SCORES ANALYSIS WITH DYNAMIC CV
# ============================================================================
print("\n" + "="*80)
print("OUT-OF-BAG SCORES ANALYSIS WITH DYNAMIC CV")
print("="*80 + "\n")

if oob_scores_history:
    print("Out-of-Bag Scores by Window:")
    for oob_data in oob_scores_history:
        print(f"  Window {oob_data['window'] + 1}: {oob_data['oob_score']:.4f}")
    
    oob_values = [oob_data['oob_score'] for oob_data in oob_scores_history]
    print(f"\nAverage OOB Score: {np.mean(oob_values):.4f}")
    print(f"Min OOB Score: {np.min(oob_values):.4f}")
    print(f"Max OOB Score: {np.max(oob_values):.4f}")
    
    # Plot OOB scores trend
    plt.figure(figsize=(10, 6))
    windows = np.arange(1, len(oob_scores_history) + 1)
    oob_values = [oob_data['oob_score'] for oob_data in oob_scores_history]
    
    plt.plot(windows, oob_values, 'o-', linewidth=2, markersize=8, color='darkred')
    plt.axhline(y=np.mean(oob_values), color='r', linestyle='--', alpha=0.7, 
                label=f'Average: {np.mean(oob_values):.4f}')
    plt.xlabel('Window Number', fontsize=12)
    plt.ylabel('Out-of-Bag Score', fontsize=12)
    plt.title('Out-of-Bag Scores Trend Across Windows (Dynamic CV)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(windows)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Out-of-Bag scores not enabled or not available")

# ============================================================================
# TRADING SIMULATION
# ============================================================================
print("\n" + "="*80)
print("TRADING SIMULATION WITH OPTIMIZED RANDOM FOREST AND DYNAMIC CV")
print("="*80 + "\n")

initial_capital = 1000
sm1 = initial_capital
f = 0
ns = 0
nn = test_window
n = initial_window

open_prices = df['Open'].values
close_prices = df['Close'].values

modelval_RF = []

for j in range(num_windows):
    for i in range(len(yptt[j])):
        price_idx = j * nn + i + n
        open_price = open_prices[price_idx]
        close_price = close_prices[price_idx]
        signal = yptt[j][i]
        
        if signal == 1 and f == 0:
            ns = (sm1 / (1 + cb)) / open_price
            sm1 = ns * close_price
            modelval_RF.append(sm1)
            f = 1
        elif signal == 1 and f == 1:
            sm1 = ns * close_price
            modelval_RF.append(sm1)
        elif signal == 0 and f == 0:
            price_change_pct = (close_price - open_price) / open_price
            sm1 = sm1 * (1 - price_change_pct)
            modelval_RF.append(sm1)
        elif signal == 0 and f == 1:
            price_change_pct = (close_price - open_price) / open_price
            sm1 = ns * open_price * (1 - cs) * (1 - price_change_pct)
            modelval_RF.append(sm1)
            f = 0
            ns = 0

total_return = ((modelval_RF[-1] - initial_capital) / initial_capital) * 100

print(f"{'='*60}")
print(f"TRADING RESULTS WITH OPTIMIZED RANDOM FOREST AND DYNAMIC CV")
print(f"{'='*60}")
print(f"Initial Investment:     ${initial_capital:.2f}")
print(f"Final Investment Value: ${modelval_RF[-1]:.2f}")
print(f"Total Return:           {total_return:.2f}%")
print(f"Absolute Profit/Loss:   ${modelval_RF[-1] - initial_capital:.2f}")
print(f"{'='*60}\n")

plt.figure(figsize=(12, 6))
plt.plot(modelval_RF, label='Investment Value', linewidth=2, color='darkred')
plt.axhline(y=initial_capital, color='r', linestyle='--', label='Initial Investment')
plt.xlabel('Trading Day')
plt.ylabel('Investment Value ($)')
plt.title('Trading System Performance with Optimized Random Forest (Dynamic CV)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================
print("\n" + "="*80)
print("PERFORMANCE METRICS WITH DYNAMIC CV")
print("="*80 + "\n")

def max_drawdown(series, window=252):
    roll_max = series.rolling(window, min_periods=1).max()
    daily_drawdown = series / roll_max - 1
    max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()
    return max_daily_drawdown

df_modelval_RF = pd.DataFrame(modelval_RF)
num_years = num_windows
aar = (((modelval_RF[-1] / modelval_RF[0]) ** (1 / num_years)) - 1) * 100
print(f"Average Annual Return (AAR): {aar:.2f}%")

mdd_series = max_drawdown(df_modelval_RF)
mdd_value = mdd_series.min()
if isinstance(mdd_value, pd.Series):
    mdd_value = mdd_value.iloc[0]
print(f"Maximum Drawdown (MDD): {mdd_value:.4f}")

modelval_RF_list = modelval_RF.copy()
modelval_RF_list.insert(0, initial_capital)

port_value = pd.DataFrame(modelval_RF_list, columns=['Value'])
ret_data = port_value.pct_change()

annal_return_RF = []
for year_idx in range(num_years):
    start_idx = year_idx * test_window
    end_idx = start_idx + test_window
    start_val = port_value.iloc[start_idx, 0]
    end_val = port_value.iloc[end_idx, 0]
    annual_ret = (end_val - start_val) / start_val
    annal_return_RF.append(annual_ret)

annal_return_RF_df = pd.DataFrame(annal_return_RF, columns=['Return'])

dffff = pd.DataFrame(np.array(ret_data), columns=['Returns'])
dffff['downside_returns'] = 0.0
dffff_clean = dffff.dropna()
dffff_clean.loc[dffff_clean['Returns'] < 0, 'downside_returns'] = abs(dffff_clean['Returns'])

negative_returns = dffff_clean[dffff_clean['Returns'] < 0]['Returns']
std_neg = negative_returns.std() if len(negative_returns) > 0 else 0

annual_downside_std = dffff_clean['downside_returns'].std() * np.sqrt(252)
print(f"Annualized Downside Std: {annual_downside_std:.4f}")

if std_neg > 0:
    sortino = ((((modelval_RF_list[-1] / modelval_RF_list[0]) ** (1 / num_years)) - 1) - riskfree) / (std_neg * np.sqrt(252))
    print(f"Sortino Ratio: {sortino:.4f}")

annual_mean_return = (((modelval_RF_list[-1] / modelval_RF_list[0]) ** (1 / num_years)) - 1)
annual_return_std = annal_return_RF_df['Return'].std()

sharp_ratio = ((annual_mean_return - riskfree) / annual_return_std) if annual_return_std > 0 else 0
calmar_ratio = ((annual_mean_return - riskfree) / (-mdd_value)) if mdd_value != 0 else 0
sortino_ratio = ((annual_mean_return - riskfree) / annual_downside_std) if annual_downside_std > 0 else 0

print(f"\nSharpe Ratio: {sharp_ratio:.4f}")
print(f"Calmar Ratio: {calmar_ratio:.4f}")
print(f"Sortino Ratio: {sortino_ratio:.4f}")

# ============================================================================
# EXECUTION TIME
# ============================================================================
end = time.time()
execution_time = (end - start) / 60
print("\n" + "="*80)
print(f"Total Execution Time: {execution_time:.2f} minutes")
print("="*80)
