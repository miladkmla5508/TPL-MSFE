import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import product

# Try to import lightgbm, handle if not installed
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("Warning: LightGBM not installed. Installing via pip...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True

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
warnings.filterwarnings('ignore', category=Warning, module='lgb')
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

# LightGBM Hyperparameter Grid for Tuning
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1, 1],      # Step size shrinkage
    'max_depth': [2, 3, 4, 5],                   # Maximum tree depth
    'num_leaves': [2, 4, 6, 8],                  # Maximum number of leaves
    #'min_child_samples': [10, 20, 40, 80],      # Minimum data in leaf
    #'min_child_weight': [1e-5, 1e-3, 1e-1],     # Minimum sum of instance weight
    'subsample': [0.2, 0.4, 0.6],                # Subsample ratio of training data
    'colsample_bytree': [0.2, 0.4, 0.6],         # Subsample ratio of columns
    'reg_alpha': [0.01, 0.1, 1.0],               # L1 regularization
    'reg_lambda': [0.01, 0.1, 1.0,],             # L2 regularization
    'n_estimators': [20, 40, 60],                # Number of boosting iterations
    'random_state': [0],                         # Random seed
    'objective': ['binary'],                     # Learning task type
    'metric': ['binary_logloss'],                # Evaluation metric
    #'boosting_type': ['gbdt', 'dart', 'goss'],  # Boosting algorithms
    #'min_split_gain': [0.0, 0.1, 0.5],          # Minimum loss reduction
    #'subsample_freq': [1],                      # Frequency for bagging
    #'verbose': [-1],                            # Disable verbose output
    #'n_jobs': [-1]                              # Use all cores
}

# Trading parameters
cb = 0.0075    # Buy commission
cs = 0.0075    # Sell commission

# Initialize scaler (LightGBM doesn't require scaling but we keep for consistency)
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
modelval_LGB = []       # Store trading system values over time
y_test_window_LGB = []  # Store all test predictions concatenated
feature_importance_history = []  # Store feature importance across windows
training_history = []           # Store training history across windows
best_params_history = []  # Store best parameters from CV for each window
cv_results_history = []  # Store CV results for each window
cv_folds_history = []    # Track number of CV folds used per window

# ============================================================================
# SLIDING WINDOW TRAINING AND PREDICTION
# ============================================================================
print("\n" + "="*80)
print("STARTING SLIDING WINDOW ANALYSIS WITH LIGHTGBM AND DYNAMIC CROSS-VALIDATION")
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
    
    # Scale data (LightGBM doesn't require scaling but we keep for consistency)
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
        
        # Use default LightGBM for SFFS
        sfs = SFS(
            lgb.LGBMClassifier(learning_rate=0.05, max_depth=4, n_estimators=50, random_state=0, verbose=-1),
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
    # LIGHTGBM PARAMETER OPTIMIZATION WITH DYNAMIC CV
    # ========================================================================
    print("\n" + "="*60)
    print(f"LIGHTGBM PARAMETER OPTIMIZATION - WINDOW {window_idx + 1}")
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
                # Create LightGBM datasets
                train_data = lgb.Dataset(X_cv_train_scaled, label=y_cv_train)
                val_data = lgb.Dataset(X_cv_val_scaled, label=y_cv_val, reference=train_data)
                
                # Remove verbose from params for training
                train_params = params.copy()
                train_params['verbose'] = -1
                
                # Train model
                model = lgb.train(
                    params=train_params,
                    train_set=train_data,
                    valid_sets=[val_data],
                    num_boost_round=train_params.get('n_estimators', 100),
                    callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)]
                )
                
                # Make predictions
                y_cv_pred = model.predict(X_cv_val_scaled)
                y_cv_pred_binary = (y_cv_pred > 0.5).astype(int)
                fold_acc = accuracy_score(y_cv_val, y_cv_pred_binary)
                fold_accuracies.append(fold_acc)
                
            except Exception as e:
                print(f"  Warning: LightGBM failed for params {params} on fold {fold}: {str(e)}")
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
        print(f"BEST LIGHTGBM PARAMETERS FOUND:")
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
            'learning_rate': 0.05,
            'max_depth': 4,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'n_estimators': 100,
            'random_state': 0,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbose': -1,
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
        ax.set_title(f'Window {window_idx + 1}: LightGBM CV Results ({len(best_fold_accs)} folds)', fontsize=18)
        ax.set_xticks(fold_indices)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        plt.tight_layout()
        plt.show()
    
    
    # ========================================================================
    # TRAIN FINAL LIGHTGBM WITH OPTIMIZED PARAMETERS
    # ========================================================================
    print("\n--- Training Final LightGBM with Optimized Parameters ---")
    
    # Scale the full training data
    final_scaler = StandardScaler()
    X_train_final_scaled = final_scaler.fit_transform(X_train_final)
    X_test_final_scaled = final_scaler.transform(X_test_final)
    
    # Train LightGBM with best parameters
    try:
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train_final_scaled, label=y_train_final)
        test_data = lgb.Dataset(X_test_final_scaled, label=y_test_final, reference=train_data)
        
        # Remove verbose from params for training
        train_params = best_params.copy()
        train_params['verbose'] = -1
        
        # Train model
        model_lgb = lgb.train(
            params=train_params,
            train_set=train_data,
            valid_sets=[test_data],
            num_boost_round=train_params.get('n_estimators', 100),
            callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)]
        )
        
        print(f"LightGBM trained with {best_params.get('n_estimators', 100)} iterations")
        print(f"Learning rate: {best_params.get('learning_rate', 'N/A')}")
        print(f"Max depth: {best_params.get('max_depth', 'N/A')}")
        print(f"Number of leaves: {best_params.get('num_leaves', 'N/A')}")
        
    except Exception as e:
        print(f"Warning: LightGBM failed to train with best params: {str(e)}")
        print("Using fallback parameters...")
        
        # Fallback to simple LightGBM
        fallback_params = {
            'learning_rate': 0.05,
            'max_depth': 4,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'n_estimators': 100,
            'random_state': 0,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbose': -1,
        }
        
        train_data = lgb.Dataset(X_train_final_scaled, label=y_train_final)
        test_data = lgb.Dataset(X_test_final_scaled, label=y_test_final, reference=train_data)
        
        model_lgb = lgb.train(
            params=fallback_params,
            train_set=train_data,
            valid_sets=[test_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)]
        )
    
    # Make predictions
    y_pred_test = model_lgb.predict(X_test_final_scaled)
    y_pred_train = model_lgb.predict(X_train_final_scaled)
    
    # Convert probabilities to binary predictions
    y_pred_test_binary = (y_pred_test > 0.5).astype(int)
    y_pred_train_binary = (y_pred_train > 0.5).astype(int)
    
    # Evaluate model
    test_accuracy = accuracy_score(y_test_final, y_pred_test_binary)
    train_accuracy = accuracy_score(y_train_final, y_pred_train_binary)
    
    try:
        auc_roc = roc_auc_score(y_test_final, y_pred_test)
        print(f"AUC-ROC Score: {auc_roc:.4f}")
    except:
        auc_roc = None
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test_final, y_pred_test_binary)}")
    
    # Store feature importance
    feature_importance = model_lgb.feature_importance(importance_type='gain')
    if feature_importance is not None and len(feature_importance) > 0:
        feature_importance_history.append({
            'features': final_features,
            'importances': feature_importance,
            'best_params': best_params,
            'window_idx': window_idx
        })
    
    # Store training history
    eval_results = model_lgb.evals_result_
    if eval_results:
        training_history.append({
            'window': window_idx,
            'eval_results': eval_results
        })
    
    # Store predictions
    yptt.append(y_pred_test_binary)
    
    # Display top 10 features
    print("\nTop 10 most important features:")
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
    y_test_window_LGB.extend(yptt[i])

print(f"Total predictions: {len(y_test_window_LGB)}")

# ============================================================================
# CV FOLDS ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("CROSS-VALIDATION FOLDS ANALYSIS")
print("="*80 + "\n")

if cv_folds_history:
    fig, ax = plt.subplots(figsize=(10, 6))
    windows = np.arange(1, num_windows + 1)
    
    ax.bar(windows, cv_folds_history, color='limegreen', edgecolor='black', alpha=0.7)
    ax.plot(windows, cv_folds_history, 'o-', color='forestgreen', linewidth=2, markersize=8)
    
    ax.set_xlabel('Window', fontsize=12)
    ax.set_ylabel('Number of CV Folds', fontsize=12)
    ax.set_title('Number of Cross-Validation Folds per Window (LightGBM)', fontsize=14)
    ax.set_xticks(windows)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(cv_folds_history):
        ax.text(i + 1, v + 0.1, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\nCV Folds Summary for LightGBM:")
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
conf_matrix = confusion_matrix(y_true, y_test_window_LGB)
acc = accuracy_score(y_true, y_test_window_LGB)
precision = precision_score(y_true, y_test_window_LGB)
recall = recall_score(y_true, y_test_window_LGB)
f1 = f1_score(y_true, y_test_window_LGB)
mcc = matthews_corrcoef(y_true, y_test_window_LGB)

print(f"Confusion Matrix:\n{conf_matrix}")
print(f"\nAccuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")
print(f"\n{classification_report(y_true, y_test_window_LGB, digits=4)}")

# ============================================================================
# VISUALIZATION: CONFUSION MATRIX
# ============================================================================
group_names = ['True Sell', 'False Buy', 'False Sell', 'True Buy']
group_counts = [f"{value:0.0f}" for value in conf_matrix.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
labels = np.asarray(labels).reshape(2, 2)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=labels, linewidths=1, fmt='', 
            cmap='YlGnBu', annot_kws={"size": 14})
ax.set_xlabel('Predicted Labels', fontsize=15)
ax.set_ylabel('Actual Labels', fontsize=15)
ax.set_title('LightGBM - Confusion Matrix (With Dynamic CV)', fontsize=16)
plt.tight_layout()
plt.show()

# ============================================================================
# LIGHTGBM PARAMETER OPTIMIZATION SUMMARY WITH DYNAMIC CV
# ============================================================================
print("\n" + "="*80)
print("LIGHTGBM PARAMETER OPTIMIZATION SUMMARY WITH DYNAMIC CV")
print("="*80 + "\n")

print("Best Parameters Selected for Each Window:")
print("-" * 60)

learning_rate_stats = {}
max_depth_stats = {}
num_leaves_stats = {}
n_estimators_stats = {}
reg_alpha_stats = {}
reg_lambda_stats = {}
subsample_stats = {}
colsample_bytree_stats = {}

for window_idx, params in enumerate(best_params_history):
    print(f"\nWindow {window_idx + 1}:")
    print(f"  learning_rate: {params.get('learning_rate', 'N/A')}")
    print(f"  max_depth: {params.get('max_depth', 'N/A')}")
    print(f"  num_leaves: {params.get('num_leaves', 'N/A')}")
    print(f"  n_estimators: {params.get('n_estimators', 'N/A')}")
    print(f"  reg_alpha: {params.get('reg_alpha', 'N/A')}")
    print(f"  reg_lambda: {params.get('reg_lambda', 'N/A')}")
    print(f"  subsample: {params.get('subsample', 'N/A')}")
    print(f"  colsample_bytree: {params.get('colsample_bytree', 'N/A')}")
    print(f"  CV folds used: {cv_folds_history[window_idx]}")
    
    # Collect statistics
    learning_rate = params.get('learning_rate', 'N/A')
    max_depth = params.get('max_depth', 'N/A')
    num_leaves = params.get('num_leaves', 'N/A')
    n_estimators = params.get('n_estimators', 'N/A')
    reg_alpha = params.get('reg_alpha', 'N/A')
    reg_lambda = params.get('reg_lambda', 'N/A')
    subsample = params.get('subsample', 'N/A')
    colsample_bytree = params.get('colsample_bytree', 'N/A')
    
    learning_rate_stats[learning_rate] = learning_rate_stats.get(learning_rate, 0) + 1
    max_depth_stats[max_depth] = max_depth_stats.get(max_depth, 0) + 1
    num_leaves_stats[num_leaves] = num_leaves_stats.get(num_leaves, 0) + 1
    n_estimators_stats[n_estimators] = n_estimators_stats.get(n_estimators, 0) + 1
    reg_alpha_stats[reg_alpha] = reg_alpha_stats.get(reg_alpha, 0) + 1
    reg_lambda_stats[reg_lambda] = reg_lambda_stats.get(reg_lambda, 0) + 1
    subsample_stats[subsample] = subsample_stats.get(subsample, 0) + 1
    colsample_bytree_stats[colsample_bytree] = colsample_bytree_stats.get(colsample_bytree, 0) + 1

# Print statistics
print("\n" + "="*60)
print("LIGHTGBM PARAMETER SELECTION STATISTICS WITH DYNAMIC CV:")
print("="*60)

print("\nLearning Rate Selection:")
for lr, count in sorted(learning_rate_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  learning_rate={lr}: {count} windows ({percentage:.1f}%)")

print("\nMax Depth Selection:")
for depth, count in sorted(max_depth_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  max_depth={depth}: {count} windows ({percentage:.1f}%)")

print("\nNumber of Leaves Selection:")
for leaves, count in sorted(num_leaves_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  num_leaves={leaves}: {count} windows ({percentage:.1f}%)")

print("\nNumber of Estimators Selection:")
for n_est, count in sorted(n_estimators_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  n_estimators={n_est}: {count} windows ({percentage:.1f}%)")

print("\nL1 Regularization (Alpha) Selection:")
for alpha, count in sorted(reg_alpha_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  reg_alpha={alpha}: {count} windows ({percentage:.1f}%)")

print("\nL2 Regularization (Lambda) Selection:")
for lambda_val, count in sorted(reg_lambda_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  reg_lambda={lambda_val}: {count} windows ({percentage:.1f}%)")

print("\nSubsample Ratio Selection:")
for subsample_val, count in sorted(subsample_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  subsample={subsample_val}: {count} windows ({percentage:.1f}%)")

print("\nColumn Sample Ratio Selection:")
for colsample, count in sorted(colsample_bytree_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  colsample_bytree={colsample}: {count} windows ({percentage:.1f}%)")

# Visualization
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# Learning rate selection
lrs, lr_counts = zip(*sorted(learning_rate_stats.items(), key=lambda x: x[1], reverse=True))
axes[0, 0].bar([str(lr) for lr in lrs], lr_counts, color='limegreen')
axes[0, 0].set_xlabel('Learning Rate', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Learning Rate Selection', fontsize=14)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Max depth selection
depths, depth_counts = zip(*sorted(max_depth_stats.items(), key=lambda x: x[1], reverse=True))
axes[0, 1].bar([str(d) for d in depths], depth_counts, color='forestgreen')
axes[0, 1].set_xlabel('Max Depth', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('Max Depth Selection', fontsize=14)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Number of leaves selection
leaves, leaf_counts = zip(*sorted(num_leaves_stats.items(), key=lambda x: x[1], reverse=True))
axes[0, 2].bar([str(l) for l in leaves], leaf_counts, color='green')
axes[0, 2].set_xlabel('Number of Leaves', fontsize=12)
axes[0, 2].set_ylabel('Frequency', fontsize=12)
axes[0, 2].set_title('Number of Leaves Selection', fontsize=14)
axes[0, 2].grid(True, alpha=0.3, axis='y')

# Number of estimators selection
n_ests, n_est_counts = zip(*sorted(n_estimators_stats.items(), key=lambda x: x[1], reverse=True))
axes[1, 0].bar([str(n) for n in n_ests], n_est_counts, color='mediumseagreen')
axes[1, 0].set_xlabel('Number of Estimators', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Number of Estimators Selection', fontsize=14)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# L1 regularization selection
alphas, alpha_counts = zip(*sorted(reg_alpha_stats.items(), key=lambda x: x[1], reverse=True))
axes[1, 1].bar([str(a) for a in alphas], alpha_counts, color='lightgreen')
axes[1, 1].set_xlabel('L1 Regularization (Alpha)', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('L1 Regularization Selection', fontsize=14)
axes[1, 1].grid(True, alpha=0.3, axis='y')

# L2 regularization selection
lambdas, lambda_counts = zip(*sorted(reg_lambda_stats.items(), key=lambda x: x[1], reverse=True))
axes[1, 2].bar([str(l) for l in lambdas], lambda_counts, color='palegreen')
axes[1, 2].set_xlabel('L2 Regularization (Lambda)', fontsize=12)
axes[1, 2].set_ylabel('Frequency', fontsize=12)
axes[1, 2].set_title('L2 Regularization Selection', fontsize=14)
axes[1, 2].grid(True, alpha=0.3, axis='y')

# Subsample selection
subsamples, subsample_counts = zip(*sorted(subsample_stats.items(), key=lambda x: x[1], reverse=True))
axes[2, 0].bar([str(s) for s in subsamples], subsample_counts, color='darkseagreen')
axes[2, 0].set_xlabel('Subsample Ratio', fontsize=12)
axes[2, 0].set_ylabel('Frequency', fontsize=12)
axes[2, 0].set_title('Subsample Selection', fontsize=14)
axes[2, 0].grid(True, alpha=0.3, axis='y')

# Column sample selection
colsamples, colsample_counts = zip(*sorted(colsample_bytree_stats.items(), key=lambda x: x[1], reverse=True))
axes[2, 1].bar([str(c) for c in colsamples], colsample_counts, color='honeydew')
axes[2, 1].set_xlabel('Column Sample Ratio', fontsize=12)
axes[2, 1].set_ylabel('Frequency', fontsize=12)
axes[2, 1].set_title('Column Sample Selection', fontsize=14)
axes[2, 1].grid(True, alpha=0.3, axis='y')

# CV Folds Evolution
windows = np.arange(1, num_windows + 1)
axes[2, 2].plot(windows, cv_folds_history, 'o-', linewidth=2, markersize=8, color='darkgreen')
axes[2, 2].set_xlabel('Window', fontsize=12)
axes[2, 2].set_ylabel('Number of CV Folds', fontsize=12)
axes[2, 2].set_title('CV Folds Evolution (LightGBM)', fontsize=14)
axes[2, 2].set_xticks(windows)
axes[2, 2].grid(True, alpha=0.3)

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
    
    ax1.barh(y_pos, importances, color='limegreen')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(features)
    ax1.invert_yaxis()
    ax1.set_xlabel('Average Feature Importance (Gain)', fontsize=12)
    ax1.set_title('Top 15 Feature Importance - LightGBM', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    occurrence_counts = [feature_occurrence[f] for f in features]
    ax2.barh(y_pos, occurrence_counts, color='forestgreen')
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
print("TRADING SIMULATION WITH OPTIMIZED LIGHTGBM AND DYNAMIC CV")
print("="*80 + "\n")

initial_capital = 1000
sm1 = initial_capital
f = 0
ns = 0
nn = test_window
n = initial_window

open_prices = df['Open'].values
close_prices = df['Close'].values

modelval_LGB = []

for j in range(num_windows):
    for i in range(len(yptt[j])):
        price_idx = j * nn + i + n
        open_price = open_prices[price_idx]
        close_price = close_prices[price_idx]
        signal = yptt[j][i]
        
        if signal == 1 and f == 0:
            ns = (sm1 / (1 + cb)) / open_price
            sm1 = ns * close_price
            modelval_LGB.append(sm1)
            f = 1
        elif signal == 1 and f == 1:
            sm1 = ns * close_price
            modelval_LGB.append(sm1)
        elif signal == 0 and f == 0:
            price_change_pct = (close_price - open_price) / open_price
            sm1 = sm1 * (1 - price_change_pct)
            modelval_LGB.append(sm1)
        elif signal == 0 and f == 1:
            price_change_pct = (close_price - open_price) / open_price
            sm1 = ns * open_price * (1 - cs) * (1 - price_change_pct)
            modelval_LGB.append(sm1)
            f = 0
            ns = 0

total_return = ((modelval_LGB[-1] - initial_capital) / initial_capital) * 100

print(f"{'='*60}")
print(f"TRADING RESULTS WITH OPTIMIZED LIGHTGBM AND DYNAMIC CV")
print(f"{'='*60}")
print(f"Initial Investment:     ${initial_capital:.2f}")
print(f"Final Investment Value: ${modelval_LGB[-1]:.2f}")
print(f"Total Return:           {total_return:.2f}%")
print(f"Absolute Profit/Loss:   ${modelval_LGB[-1] - initial_capital:.2f}")
print(f"{'='*60}\n")

plt.figure(figsize=(12, 6))
plt.plot(modelval_LGB, label='Investment Value', linewidth=2, color='forestgreen')
plt.axhline(y=initial_capital, color='r', linestyle='--', label='Initial Investment')
plt.xlabel('Trading Day')
plt.ylabel('Investment Value ($)')
plt.title('Trading System Performance with Optimized LightGBM (Dynamic CV)')
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

df_modelval_LGB = pd.DataFrame(modelval_LGB)
num_years = num_windows
aar = (((modelval_LGB[-1] / modelval_LGB[0]) ** (1 / num_years)) - 1) * 100
print(f"Average Annual Return (AAR): {aar:.2f}%")

mdd_series = max_drawdown(df_modelval_LGB)
mdd_value = mdd_series.min()
if isinstance(mdd_value, pd.Series):
    mdd_value = mdd_value.iloc[0]
print(f"Maximum Drawdown (MDD): {mdd_value:.4f}")

modelval_LGB_list = modelval_LGB.copy()
modelval_LGB_list.insert(0, initial_capital)

port_value = pd.DataFrame(modelval_LGB_list, columns=['Value'])
ret_data = port_value.pct_change()

annal_return_LGB = []
for year_idx in range(num_years):
    start_idx = year_idx * test_window
    end_idx = start_idx + test_window
    start_val = port_value.iloc[start_idx, 0]
    end_val = port_value.iloc[end_idx, 0]
    annual_ret = (end_val - start_val) / start_val
    annal_return_LGB.append(annual_ret)

annal_return_LGB_df = pd.DataFrame(annal_return_LGB, columns=['Return'])

dffff = pd.DataFrame(np.array(ret_data), columns=['Returns'])
dffff['downside_returns'] = 0.0
dffff_clean = dffff.dropna()
dffff_clean.loc[dffff_clean['Returns'] < 0, 'downside_returns'] = abs(dffff_clean['Returns'])

negative_returns = dffff_clean[dffff_clean['Returns'] < 0]['Returns']
std_neg = negative_returns.std() if len(negative_returns) > 0 else 0

annual_downside_std = dffff_clean['downside_returns'].std() * np.sqrt(252)
print(f"Annualized Downside Std: {annual_downside_std:.4f}")

if std_neg > 0:
    sortino = ((((modelval_LGB_list[-1] / modelval_LGB_list[0]) ** (1 / num_years)) - 1) - riskfree) / (std_neg * np.sqrt(252))
    print(f"Sortino Ratio: {sortino:.4f}")

annual_mean_return = (((modelval_LGB_list[-1] / modelval_LGB_list[0]) ** (1 / num_years)) - 1)
annual_return_std = annal_return_LGB_df['Return'].std()

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
