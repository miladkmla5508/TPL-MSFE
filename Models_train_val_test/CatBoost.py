import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import product

# Try to import catboost, handle if not installed
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    print("Warning: CatBoost not installed. Installing via pip...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             classification_report, roc_auc_score)

# Suppress convergence warnings
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

param_grid = {
    'iterations': [20, 40, 60],                   # Number of boosting iterations (trees)
    'depth': [2, 3, 4],                           # Depth of the trees
    'learning_rate': [0.01, 0.1, 1],              # Step size shrinkage
    'l2_leaf_reg':  [0.01, 0.1, 1],               # L2 regularization term
    'border_count': [32, 64, 128],                # Number of splits for numerical features
    'random_strength': [0.5, 1, 2],               # Randomness in scoring
    #'bagging_temperature': [0.5, 1, 2],          # Controls randomness of bagging
    'random_state': [0],                          # Random seed for reproducibility
    'loss_function': ['Logloss'],                 # Loss function for binary classification
    'verbose': [False],                           # Disable training logs
    'task_type': ['CPU'],                         # Use CPU for computation
    #'thread_count': [-1],                        # Use all available cores
    #'bootstrap_type': ['Bayesian', 'Bernoulli'], # Bootstrap method
    #'leaf_estimation_iterations': [1],           # Number of gradient steps
    #'eval_metric': ['Accuracy'],                 # Metric for evaluation
    #'early_stopping_rounds': [20],               # Early stopping rounds
    #'use_best_model': [False]                    # Use best model according to eval metric
}
# Trading parameters
cb = 0.0075    # Buy commission
cs = 0.0075    # Sell commission

# Initialize scaler
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
modelval_CAT = []       # Store trading system values over time
y_test_window_CAT = []  # Store all test predictions concatenated
feature_importance_history = []  # Store feature importance across windows
learning_curves_history = []    # Store learning curves across windows
best_params_history = []  # Store best parameters from CV for each window
cv_results_history = []  # Store CV results for each window
cv_folds_history = []    # Track number of CV folds used per window

# ============================================================================
# SLIDING WINDOW TRAINING AND PREDICTION
# ============================================================================
print("\n" + "="*80)
print("STARTING SLIDING WINDOW ANALYSIS WITH CATBOOST AND DYNAMIC CROSS-VALIDATION")
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
    
    # Scale data (CatBoost handles scaling internally but we keep for consistency)
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
        
        # Use default CatBoost for SFFS
        sfs = SFS(
            CatBoostClassifier(iterations=100, depth=4, learning_rate=0.05, random_state=0, verbose=False),
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
    # CATBOOST PARAMETER OPTIMIZATION WITH DYNAMIC CV
    # ========================================================================
    print("\n" + "="*60)
    print(f"CATBOOST PARAMETER OPTIMIZATION - WINDOW {window_idx + 1}")
    print("="*60 + "\n")
    
    # Generate all parameter combinations
    param_combinations = []
    keys = list(param_grid.keys())
    for values in product(*[param_grid[k] for k in keys]):
        param_dict = dict(zip(keys, values))
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
                model = CatBoostClassifier(**params)
                model.fit(X_cv_train_scaled, y_cv_train, 
                         eval_set=(X_cv_val_scaled, y_cv_val),
                         verbose=False)
                y_cv_pred = model.predict(X_cv_val_scaled)
                fold_acc = accuracy_score(y_cv_val, y_cv_pred)
                fold_accuracies.append(fold_acc)
                
            except Exception as e:
                print(f"  Warning: CatBoost failed for params {params} on fold {fold}: {str(e)}")
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
        
        # Progress update
        if (param_idx + 1) % 10 == 0:
            print(f"  Tested {param_idx + 1}/{len(param_combinations)} combinations...")
    
    # Find best parameters
    if cv_results:
        best_result = max(cv_results, key=lambda x: x['mean_accuracy'])
        best_params = best_result['params']
        best_mean_acc = best_result['mean_accuracy']
        best_fold_accs = best_result['fold_accuracies']
        
        print(f"\n{'='*60}")
        print(f"BEST CATBOOST PARAMETERS FOUND:")
        print(f"{'='*60}")
        for key, value in best_params.items():
            if key not in ['verbose']:  # Skip verbose in display
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
            'iterations': 100,
            'depth': 4,
            'learning_rate': 0.05,
            'l2_leaf_reg': 3,
            'border_count': 254,
            'random_strength': 1,
            'random_state': 0,
            'loss_function': 'Logloss',
            'verbose': False,
            'task_type': 'CPU',
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
        ax.set_title(f'Window {window_idx + 1}: CatBoost CV Results ({len(best_fold_accs)} folds)', fontsize=18)
        ax.set_xticks(fold_indices)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        plt.tight_layout()
        plt.show()
    
    # ========================================================================
    # TRAIN FINAL CATBOOST WITH OPTIMIZED PARAMETERS
    # ========================================================================
    print("\n--- Training Final CatBoost with Optimized Parameters ---")
    
    # Scale the full training data
    final_scaler = StandardScaler()
    X_train_final_scaled = final_scaler.fit_transform(X_train_final)
    X_test_final_scaled = final_scaler.transform(X_test_final)
    
    # Train CatBoost with best parameters
    try:
        model_cat = CatBoostClassifier(**best_params)
        
        # Train model with validation set for early stopping
        model_cat.fit(X_train_final_scaled, y_train_final, 
                      eval_set=(X_test_final_scaled, y_test_final),
                      verbose=False)
        
        print(f"CatBoost trained with {best_params['iterations']} iterations")
        print(f"Tree depth: {best_params['depth']}")
        print(f"Learning rate: {best_params['learning_rate']}")
        print(f"L2 regularization: {best_params.get('l2_leaf_reg', 'N/A')}")
        
    except Exception as e:
        print(f"Warning: CatBoost failed to train with best params: {str(e)}")
        print("Using fallback parameters...")
        
        # Fallback to simple CatBoost
        fallback_params = {
            'iterations': 100,
            'depth': 4,
            'learning_rate': 0.05,
            'l2_leaf_reg': 3,
            'border_count': 254,
            'random_strength': 1,
            'random_state': 0,
            'loss_function': 'Logloss',
            'verbose': False,
            'task_type': 'CPU',
        }
        
        model_cat = CatBoostClassifier(**fallback_params)
        model_cat.fit(X_train_final_scaled, y_train_final, verbose=False)
    
    # Make predictions
    y_pred_test = model_cat.predict(X_test_final_scaled)
    y_pred_train = model_cat.predict(X_train_final_scaled)
    y_pred_prob_test = model_cat.predict_proba(X_test_final_scaled)[:, 1]
    
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
    if hasattr(model_cat, 'get_feature_importance'):
        feature_importance = model_cat.get_feature_importance()
        if feature_importance is not None and len(feature_importance) > 0:
            feature_importance_history.append({
                'features': final_features,
                'importances': feature_importance,
                'best_params': best_params,
                'window_idx': window_idx
            })
    
    # Store learning curve data
    if hasattr(model_cat, 'get_evals_result'):
        evals_result = model_cat.get_evals_result()
        if evals_result:
            learning_curves_history.append({
                'window': window_idx,
                'evals_result': evals_result
            })
    
    # Store predictions
    yptt.append(y_pred_test)
    
    # Display top 10 features
    print("\nTop 10 most important features:")
    if hasattr(model_cat, 'get_feature_importance'):
        feature_importance = model_cat.get_feature_importance()
        if feature_importance is not None and len(feature_importance) > 0:
            feature_importance_list = list(zip(final_features, feature_importance))
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
    y_test_window_CAT.extend(yptt[i])

print(f"Total predictions: {len(y_test_window_CAT)}")

# ============================================================================
# CV FOLDS ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("CROSS-VALIDATION FOLDS ANALYSIS")
print("="*80 + "\n")

if cv_folds_history:
    fig, ax = plt.subplots(figsize=(10, 6))
    windows = np.arange(1, num_windows + 1)
    
    ax.bar(windows, cv_folds_history, color='teal', edgecolor='black', alpha=0.7)
    ax.plot(windows, cv_folds_history, 'o-', color='darkcyan', linewidth=2, markersize=8)
    
    ax.set_xlabel('Window', fontsize=12)
    ax.set_ylabel('Number of CV Folds', fontsize=12)
    ax.set_title('Number of Cross-Validation Folds per Window (CatBoost)', fontsize=14)
    ax.set_xticks(windows)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(cv_folds_history):
        ax.text(i + 1, v + 0.1, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\nCV Folds Summary for CatBoost:")
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
conf_matrix = confusion_matrix(y_true, y_test_window_CAT)
acc = accuracy_score(y_true, y_test_window_CAT)
precision = precision_score(y_true, y_test_window_CAT)
recall = recall_score(y_true, y_test_window_CAT)
f1 = f1_score(y_true, y_test_window_CAT)
mcc = matthews_corrcoef(y_true, y_test_window_CAT)

print(f"Confusion Matrix:\n{conf_matrix}")
print(f"\nAccuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")
print(f"\n{classification_report(y_true, y_test_window_CAT, digits=4)}")

# ============================================================================
# VISUALIZATION: CONFUSION MATRIX
# ============================================================================
group_names = ['True Sell', 'False Buy', 'False Sell', 'True Buy']
group_counts = [f"{value:0.0f}" for value in conf_matrix.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
labels = np.asarray(labels).reshape(2, 2)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=labels, linewidths=1, fmt='', 
            cmap='viridis', annot_kws={"size": 14})
ax.set_xlabel('Predicted Labels', fontsize=15)
ax.set_ylabel('Actual Labels', fontsize=15)
ax.set_title('CatBoost - Confusion Matrix (With Dynamic CV)', fontsize=16)
plt.tight_layout()
plt.show()

# ============================================================================
# CATBOOST PARAMETER OPTIMIZATION SUMMARY WITH DYNAMIC CV
# ============================================================================
print("\n" + "="*80)
print("CATBOOST PARAMETER OPTIMIZATION SUMMARY WITH DYNAMIC CV")
print("="*80 + "\n")

print("Best Parameters Selected for Each Window:")
print("-" * 60)

iterations_stats = {}
depth_stats = {}
learning_rate_stats = {}
l2_leaf_reg_stats = {}
border_count_stats = {}
random_strength_stats = {}

for window_idx, params in enumerate(best_params_history):
    print(f"\nWindow {window_idx + 1}:")
    print(f"  iterations: {params.get('iterations', 'N/A')}")
    print(f"  depth: {params.get('depth', 'N/A')}")
    print(f"  learning_rate: {params.get('learning_rate', 'N/A')}")
    print(f"  l2_leaf_reg: {params.get('l2_leaf_reg', 'N/A')}")
    print(f"  border_count: {params.get('border_count', 'N/A')}")
    print(f"  random_strength: {params.get('random_strength', 'N/A')}")
    print(f"  CV folds used: {cv_folds_history[window_idx]}")
    
    # Collect statistics
    iterations = params.get('iterations', 'N/A')
    depth = params.get('depth', 'N/A')
    learning_rate = params.get('learning_rate', 'N/A')
    l2_leaf_reg = params.get('l2_leaf_reg', 'N/A')
    border_count = params.get('border_count', 'N/A')
    random_strength = params.get('random_strength', 'N/A')
    
    iterations_stats[iterations] = iterations_stats.get(iterations, 0) + 1
    depth_stats[depth] = depth_stats.get(depth, 0) + 1
    learning_rate_stats[learning_rate] = learning_rate_stats.get(learning_rate, 0) + 1
    l2_leaf_reg_stats[l2_leaf_reg] = l2_leaf_reg_stats.get(l2_leaf_reg, 0) + 1
    border_count_stats[border_count] = border_count_stats.get(border_count, 0) + 1
    random_strength_stats[random_strength] = random_strength_stats.get(random_strength, 0) + 1

# Print statistics
print("\n" + "="*60)
print("CATBOOST PARAMETER SELECTION STATISTICS WITH DYNAMIC CV:")
print("="*60)

print("\nNumber of Iterations Selection:")
for iters, count in sorted(iterations_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  iterations={iters}: {count} windows ({percentage:.1f}%)")

print("\nTree Depth Selection:")
for depth, count in sorted(depth_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  depth={depth}: {count} windows ({percentage:.1f}%)")

print("\nLearning Rate Selection:")
for lr, count in sorted(learning_rate_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  learning_rate={lr}: {count} windows ({percentage:.1f}%)")

print("\nL2 Regularization Selection:")
for reg, count in sorted(l2_leaf_reg_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  l2_leaf_reg={reg}: {count} windows ({percentage:.1f}%)")

print("\nBorder Count Selection:")
for border, count in sorted(border_count_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  border_count={border}: {count} windows ({percentage:.1f}%)")

print("\nRandom Strength Selection:")
for strength, count in sorted(random_strength_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  random_strength={strength}: {count} windows ({percentage:.1f}%)")

# Visualization
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Number of iterations selection
iters, iter_counts = zip(*sorted(iterations_stats.items(), key=lambda x: x[1], reverse=True))
axes[0, 0].bar([str(i) for i in iters], iter_counts, color='teal')
axes[0, 0].set_xlabel('Number of Iterations', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Number of Iterations Selection', fontsize=14)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Tree depth selection
depths, depth_counts = zip(*sorted(depth_stats.items(), key=lambda x: x[1], reverse=True))
axes[0, 1].bar([str(d) for d in depths], depth_counts, color='darkcyan')
axes[0, 1].set_xlabel('Tree Depth', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('Tree Depth Selection', fontsize=14)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Learning rate selection
lrs, lr_counts = zip(*sorted(learning_rate_stats.items(), key=lambda x: x[1], reverse=True))
axes[1, 0].bar([str(lr) for lr in lrs], lr_counts, color='cadetblue')
axes[1, 0].set_xlabel('Learning Rate', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Learning Rate Selection', fontsize=14)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# L2 regularization selection
regs, reg_counts = zip(*sorted(l2_leaf_reg_stats.items(), key=lambda x: x[1], reverse=True))
axes[1, 1].bar([str(reg) for reg in regs], reg_counts, color='lightseagreen')
axes[1, 1].set_xlabel('L2 Leaf Regularization', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('L2 Regularization Selection', fontsize=14)
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Border count selection
borders, border_counts = zip(*sorted(border_count_stats.items(), key=lambda x: x[1], reverse=True))
axes[2, 0].bar([str(b) for b in borders], border_counts, color='mediumturquoise')
axes[2, 0].set_xlabel('Border Count', fontsize=12)
axes[2, 0].set_ylabel('Frequency', fontsize=12)
axes[2, 0].set_title('Border Count Selection', fontsize=14)
axes[2, 0].grid(True, alpha=0.3, axis='y')

# CV Folds Evolution
windows = np.arange(1, num_windows + 1)
axes[2, 1].plot(windows, cv_folds_history, 'o-', linewidth=2, markersize=8, color='darkgreen')
axes[2, 1].set_xlabel('Window', fontsize=12)
axes[2, 1].set_ylabel('Number of CV Folds', fontsize=12)
axes[2, 1].set_title('CV Folds Evolution (CatBoost)', fontsize=14)
axes[2, 1].set_xticks(windows)
axes[2, 1].grid(True, alpha=0.3)

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
    
    ax1.barh(y_pos, importances, color='teal')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(features)
    ax1.invert_yaxis()
    ax1.set_xlabel('Average Feature Importance', fontsize=12)
    ax1.set_title('Top 15 Feature Importance - CatBoost', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    occurrence_counts = [feature_occurrence[f] for f in features]
    ax2.barh(y_pos, occurrence_counts, color='darkcyan')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(features)
    ax2.invert_yaxis()
    ax2.set_xlabel('Number of Windows Selected', fontsize=12)
    ax2.set_title('Feature Selection Frequency', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# TRADING SIMULATION
# ============================================================================
print("\n" + "="*80)
print("TRADING SIMULATION WITH OPTIMIZED CATBOOST AND DYNAMIC CV")
print("="*80 + "\n")

initial_capital = 1000
sm1 = initial_capital
f = 0
ns = 0
nn = test_window
n = initial_window

open_prices = df['Open'].values
close_prices = df['Close'].values

modelval_CAT = []

for j in range(num_windows):
    for i in range(len(yptt[j])):
        price_idx = j * nn + i + n
        open_price = open_prices[price_idx]
        close_price = close_prices[price_idx]
        signal = yptt[j][i]
        
        if signal == 1 and f == 0:
            ns = (sm1 / (1 + cb)) / open_price
            sm1 = ns * close_price
            modelval_CAT.append(sm1)
            f = 1
        elif signal == 1 and f == 1:
            sm1 = ns * close_price
            modelval_CAT.append(sm1)
        elif signal == 0 and f == 0:
            price_change_pct = (close_price - open_price) / open_price
            sm1 = sm1 * (1 - price_change_pct)
            modelval_CAT.append(sm1)
        elif signal == 0 and f == 1:
            price_change_pct = (close_price - open_price) / open_price
            sm1 = ns * open_price * (1 - cs) * (1 - price_change_pct)
            modelval_CAT.append(sm1)
            f = 0
            ns = 0

total_return = ((modelval_CAT[-1] - initial_capital) / initial_capital) * 100

print(f"{'='*60}")
print(f"TRADING RESULTS WITH OPTIMIZED CATBOOST AND DYNAMIC CV")
print(f"{'='*60}")
print(f"Initial Investment:     ${initial_capital:.2f}")
print(f"Final Investment Value: ${modelval_CAT[-1]:.2f}")
print(f"Total Return:           {total_return:.2f}%")
print(f"Absolute Profit/Loss:   ${modelval_CAT[-1] - initial_capital:.2f}")
print(f"{'='*60}\n")

plt.figure(figsize=(12, 6))
plt.plot(modelval_CAT, label='Investment Value', linewidth=2, color='darkcyan')
plt.axhline(y=initial_capital, color='r', linestyle='--', label='Initial Investment')
plt.xlabel('Trading Day')
plt.ylabel('Investment Value ($)')
plt.title('Trading System Performance with Optimized CatBoost (Dynamic CV)')
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

df_modelval_CAT = pd.DataFrame(modelval_CAT)
num_years = num_windows
aar = (((modelval_CAT[-1] / modelval_CAT[0]) ** (1 / num_years)) - 1) * 100
print(f"Average Annual Return (AAR): {aar:.2f}%")

mdd_series = max_drawdown(df_modelval_CAT)
mdd_value = mdd_series.min()
if isinstance(mdd_value, pd.Series):
    mdd_value = mdd_value.iloc[0]
print(f"Maximum Drawdown (MDD): {mdd_value:.4f}")

modelval_CAT_list = modelval_CAT.copy()
modelval_CAT_list.insert(0, initial_capital)

port_value = pd.DataFrame(modelval_CAT_list, columns=['Value'])
ret_data = port_value.pct_change()

annal_return_CAT = []
for year_idx in range(num_years):
    start_idx = year_idx * test_window
    end_idx = start_idx + test_window
    start_val = port_value.iloc[start_idx, 0]
    end_val = port_value.iloc[end_idx, 0]
    annual_ret = (end_val - start_val) / start_val
    annal_return_CAT.append(annual_ret)

annal_return_CAT_df = pd.DataFrame(annal_return_CAT, columns=['Return'])

dffff = pd.DataFrame(np.array(ret_data), columns=['Returns'])
dffff['downside_returns'] = 0.0
dffff_clean = dffff.dropna()
dffff_clean.loc[dffff_clean['Returns'] < 0, 'downside_returns'] = abs(dffff_clean['Returns'])

negative_returns = dffff_clean[dffff_clean['Returns'] < 0]['Returns']
std_neg = negative_returns.std() if len(negative_returns) > 0 else 0

annual_mean_return = (((modelval_CAT_list[-1] / modelval_CAT_list[0]) ** (1 / num_years)) - 1)
annual_return_std = annal_return_CAT_df['Return'].std()


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
