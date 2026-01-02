import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import shap
import xgboost as xgb
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2, RFECV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             classification_report, roc_auc_score, roc_curve, 
                             precision_recall_curve, average_precision_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from operator import itemgetter
from matplotlib.patches import Patch
import json
import warnings
from itertools import product

# Suppress convergence warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=Warning, module='scipy')
warnings.filterwarnings('ignore', category=Warning, module='ReliefF')
warnings.filterwarnings('ignore', category=Warning, module='mlxtend')

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

# XGBoost Hyperparameter Grid for Tuning
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1, 1],     # Step size shrinkage
    'max_depth': [2, 3, 4, 5],                  # Maximum tree depth
    'n_estimators': [100, 200, 300],        # Number of boosting rounds
    'gamma': [0.01, 0.1, 1],                     # Minimum loss reduction
    #'subsample': [0.6, 0.8, 1.0],                # Subsample ratio of training instances
    'colsample_bytree': [0.6, 0.8, 1.0],         # Subsample ratio of columns
    #'min_child_weight': [1, 3, 5],               # Minimum sum of instance weight needed
    #'reg_alpha': [0.001, 0.01, 0.1],                # L1 regularization term
    'reg_lambda': [0.001, 0.01, 0.1],               # L2 regularization term
    'objective': ['binary:logistic'],            # Learning task
    'eval_metric': ['logloss'],                  # Evaluation metric for validation data
    'random_state': [0],                         # Random seed
    #'n_jobs': [-1],                              # Number of parallel threads
    #'tree_method': ['hist'],                     # Tree construction algorithm
    #'grow_policy': ['depthwise', 'lossguide'],   # Tree growth policy
    #'max_bin': [256],                            # Number of bins for histogram
    #'enable_categorical': [False]                # Disable categorical features
}

# Trading parameters
cb = 0.0075  # Buy commission
cs = 0.0075  # Sell commission

# Initialize scaler
scaler = StandardScaler()

# ============================================================================
# EXTRACT FEATURE NAMES
# ============================================================================
list_features = list(df.loc[:, 'SMA_6':'UI_20'].columns)
print(f"Total features available: {len(list_features)}")

# ============================================================================
# INITIALIZE STORAGE VARIABLES AND DICTIONARIES
# ============================================================================
yptt = []              # Store predictions for each window
modelval_XGB = []      # Store trading system values over time
y_test_window_XGB = [] # Store all test predictions concatenated

# For SHAP analysis
window_shap_values = {}
window_feature_importance = {}
selected_features_per_window = {}
shap_force_plots_data = {}

# For metrics and SFS
sfs_metrics = {}
fprs = []
tprs = []
roc_aucs = []
precisions = []
recalls = []
pr_aucs = []
window_performance = {}

# For hyperparameter tuning
best_params_history = []  # Store best parameters from CV for each window
cv_results_history = []   # Store CV results for each window
cv_accuracy_history = []  # Store CV accuracy for visualization
cv_folds_history = []     # Store CV folds for each window

# ============================================================================
# SLIDING WINDOW TRAINING AND PREDICTION
# ============================================================================
print("\n" + "="*80)
print("STARTING SLIDING WINDOW ANALYSIS WITH XGBOOST HYPERPARAMETER TUNING")
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
    # Split features and labels
    X_train_raw = df.drop('LABEL', axis=1)[:train_end]
    X_test_raw = df.drop('LABEL', axis=1)[test_start:test_end]
    y_train = df['LABEL'][:train_end]
    y_test = df['LABEL'][test_start:test_end]
    
    # Scale data
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
    print(f"Class distribution - Test: {y_test.value_counts().to_dict()}")
    
    # Multi-stage Self-adaptive Feature Engineering (MSFE)
    # ========================================================================
    # FEATURE SELECTION - STEP 1: Filtering METHODS
    # ========================================================================
    print("\n--- Feature Selection Step 1: Univariate Methods ---")
    
    # Use scaled data for ANOVA and MI
    x_main = X_train.copy()
    # Use raw data for Chi-Square (needs non-negative values)
    x_main_raw = X_train_raw.values
    
    # ANOVA F-test
    print("Running ANOVA F-test...")
    select_k_best_anova = SelectKBest(f_classif, k=topk)
    select_k_best_anova.fit(x_main, y_train)
    selected_indices_anova = select_k_best_anova.get_support(indices=True)
    selected_features_anova = [list_features[i] for i in selected_indices_anova if i < len(list_features)]
    print(f"ANOVA selected features: {len(selected_features_anova)}")
    
    # Mutual Information
    print("Running Mutual Information...")
    select_k_best_mi = SelectKBest(mutual_info_classif, k=topk)
    select_k_best_mi.fit(x_main, y_train)
    selected_indices_mi = select_k_best_mi.get_support(indices=True)
    selected_features_mi = [list_features[i] for i in selected_indices_mi if i < len(list_features)]
    print(f"Mutual Info selected features: {len(selected_features_mi)}")
    
    # Chi-Square test (needs non-negative values - use raw data)
    print("Running Chi-Square test...")
    
    # Check if raw data has negative values
    if np.any(x_main_raw < 0):
        print("WARNING: Data contains negative values, using MinMax scaling for Chi-Square")
        minmax_scaler = MinMaxScaler()
        x_main_chi2 = minmax_scaler.fit_transform(x_main_raw)
    else:
        x_main_chi2 = x_main_raw
    
    select_k_best_chi2 = SelectKBest(chi2, k=topk)
    select_k_best_chi2.fit(x_main_chi2, y_train)
    selected_indices_chi2 = select_k_best_chi2.get_support(indices=True)
    selected_features_chi2 = [list_features[i] for i in selected_indices_chi2 if i < len(list_features)]
    print(f"Chi-Square selected features: {len(selected_features_chi2)}")
    
    # ========================================================================
    # FEATURE SELECTION - STEP 2: INTERSECTION
    # ========================================================================
    print("\n--- Feature Selection Step 2: Finding Common Features ---")
    
    # Find intersection of all three methods
    common = list(set(set(selected_features_anova).intersection(selected_features_mi)).intersection(selected_features_chi2))
    print(f"Common features found: {len(common)}")
    
    if len(common) < num_features:
        print(f"WARNING: Only {len(common)} common features found, needed {num_features}")
        print("Consider increasing 'topk' parameter")
        common = common[:min(len(common), num_features)]
    
    # Get feature indices
    feat_idx = sorted([list_features.index(f) for f in common])
    print(f"Feature indices: {feat_idx[:10]}..." if len(feat_idx) > 10 else f"Feature indices: {feat_idx}")
    
    # Extract common features from scaled data
    X_train_common = X_train[:, feat_idx]
    X_test_common = X_test[:, feat_idx]
    
    # ========================================================================
    # FEATURE SELECTION - STEP 3: RELIEFF (Filtering METHODS)
    # ========================================================================
    print("\n--- Feature Selection Step 3: ReliefF ---")
    
    try:
        from sklearn_relief import ReliefF
        
        # Apply ReliefF
        fs = ReliefF(n_features=num_features)
        fs.fit(X_train_common, y_train.values)  # Convert to numpy array
        
        # Transform and identify selected features
        X_train_transformed = fs.transform(X_train_common)
        
        # Get the feature indices that ReliefF selected
        if hasattr(fs, 'feature_importances_'):
            # Get indices of top features based on importance scores
            feature_scores = fs.feature_importances_
            top_indices = np.argsort(feature_scores)[-num_features:][::-1]
            selected_feature_names_relief = [common[i] for i in top_indices]
        else:
            # Alternative method: match transformed columns to original
            X_train_df = pd.DataFrame(X_train_common, columns=common).reset_index(drop=True)
            X_transformed_df = pd.DataFrame(X_train_transformed).reset_index(drop=True)
            
            selected_feature_names_relief = []
            for col_idx in range(X_transformed_df.shape[1]):
                transformed_col = X_transformed_df.iloc[:, col_idx].values
                
                # Find matching column in original data
                for original_col in X_train_df.columns:
                    original_values = X_train_df[original_col].values
                    if np.allclose(transformed_col, original_values, rtol=1e-5, atol=1e-8):
                        selected_feature_names_relief.append(original_col)
                        break
        
        print(f"ReliefF selected {len(selected_feature_names_relief)} features")
        
    except ImportError:
        print("sklearn_relief not installed, skipping ReliefF step")
        selected_feature_names_relief = common[:num_features]
    except Exception as e:
        print(f"ReliefF encountered an error: {str(e)}")
        print("Using fallback: selecting top features from common features")
        selected_feature_names_relief = common[:num_features]
    
    # ========================================================================
    # FEATURE SELECTION - STEP 4: SFFS (Sequential Forward Floating Selection) (Wrapper METHODS)
    # ========================================================================
    print("\n--- Feature Selection Step 4: SFFS ---")

    try:
        from mlxtend.feature_selection import SequentialFeatureSelector as SFS
        
        # Prepare data for SFFS
        feature_indices = [list_features.index(f) for f in selected_feature_names_relief]
        X_sffs = df.iloc[:train_end, feature_indices].values
        y_sffs = y_train.values
        
        sfs = SFS(
            xgb.XGBClassifier(learning_rate=0.05, max_depth=3, n_estimators=200, random_state=0),
            k_features=(1, 20),
            forward=True,
            floating=True,
            verbose=0,
            scoring='accuracy',
            cv=False,
            n_jobs=-1
        )
        
        sfs.fit(X_sffs, y_sffs)
        final_features = [selected_feature_names_relief[i] for i in sfs.k_feature_idx_]
        
        # Store SFS metrics
        sfs_metrics[f'Window {window_idx + 1}'] = sfs.get_metric_dict()
        print(f"SFFS selected {len(final_features)} features")
        
    except ImportError:
        print("mlxtend not installed, using top 10 from ReliefF")
        final_features = selected_feature_names_relief[:10]
        sfs_metrics[f'Window {window_idx + 1}'] = {}
    except Exception as e:
        print(f"SFFS error: {str(e)}")
        final_features = selected_feature_names_relief[:10]
        sfs_metrics[f'Window {window_idx + 1}'] = {}
    
    # Store selected features
    selected_features_per_window[f'Window {window_idx + 1}'] = final_features
    
    # ========================================================================
    # CREATE FINAL DATASET
    # ========================================================================
    df_novel = pd.concat([df[final_features], df['LABEL']], axis=1)
    
    X_train_final = df_novel.drop('LABEL', axis=1)[:train_end]
    X_test_final = df_novel.drop('LABEL', axis=1)[test_start:test_end]
    
    # Re-scale the selected features
    scaler_final = StandardScaler()
    X_train_final_scaled = scaler_final.fit_transform(X_train_final)
    X_test_final_scaled = scaler_final.transform(X_test_final)
    
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
    # XGBOOST HYPERPARAMETER TUNING WITH WALK-FORWARD CV
    # ========================================================================
    print("\n" + "="*60)
    print(f"XGBOOST HYPERPARAMETER TUNING - WINDOW {window_idx + 1}")
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
                # Create XGBoost with current parameters
                model = xgb.XGBClassifier(**params)
                model.fit(X_cv_train_scaled, y_cv_train, verbose=False)
                y_cv_pred = model.predict(X_cv_val_scaled)
                fold_acc = accuracy_score(y_cv_val, y_cv_pred)
                fold_accuracies.append(fold_acc)
                
            except Exception as e:
                print(f"  Warning: XGBoost failed for params {params} on fold {fold}: {str(e)}")
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
        if (param_idx + 1) % 10 == 0 or (param_idx + 1) == len(param_combinations):
            print(f"  Tested {param_idx + 1}/{len(param_combinations)} combinations...")
    
    # Find best parameters
    if cv_results:
        best_result = max(cv_results, key=lambda x: x['mean_accuracy'])
        best_params = best_result['params']
        best_mean_acc = best_result['mean_accuracy']
        best_fold_accs = best_result['fold_accuracies']
        
        print(f"\n{'='*60}")
        print(f"BEST XGBOOST PARAMETERS FOUND:")
        print(f"{'='*60}")
        for key, value in best_params.items():
            if key not in ['n_jobs']:  # Skip n_jobs in display
                print(f"  {key}: {value}")
        print(f"\nCross-Validation Accuracies by Fold:")
        for fold_idx, acc in enumerate(best_fold_accs):
            print(f"  Fold {fold_idx + 1}: {acc:.4f}")
        print(f"Mean CV Accuracy: {best_mean_acc:.4f} (+/- {best_result['std_accuracy']:.4f})")
        print(f"Number of CV folds used: {best_result['num_folds']}")
        print(f"{'='*60}\n")
    else:
        print("ERROR: No valid CV results. Using default parameters.")
        best_params = {
            'learning_rate': 0.05,
            'max_depth': 3,
            'n_estimators': 200,
            'gamma': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 0,
            'n_jobs': -1,
            'tree_method': 'hist',
            'grow_policy': 'depthwise',
            'max_bin': 256,
            'enable_categorical': False
        }
        best_mean_acc = 0.0
        best_fold_accs = []
    
    # Store CV results for this window
    best_params_history.append(best_params)
    cv_results_history.append(cv_results)
    cv_accuracy_history.append(best_fold_accs)
    
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
        ax.set_title(f'Window {window_idx + 1}: XGBoost CV Results ({len(best_fold_accs)} folds)', fontsize=18)
        ax.set_xticks(fold_indices)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        plt.tight_layout()
        #plt.savefig(f'/kaggle/working/walkforward_CV{window_idx + 1}.svg', 
                  #bbox_inches='tight', dpi=150)
        plt.show()  
    
    
    # ========================================================================
    # TRAIN FINAL XGBOOST WITH OPTIMIZED PARAMETERS
    # ========================================================================
    print("\n--- Training Final XGBoost with Optimized Parameters ---")
    
    # Scale the full training data
    final_scaler = StandardScaler()
    X_train_final_scaled = final_scaler.fit_transform(X_train_final)
    X_test_final_scaled = final_scaler.transform(X_test_final)
    
    # Train XGBoost with best parameters
    try:
        model_xgb = xgb.XGBClassifier(**best_params)
        model_xgb.fit(
            X_train_final_scaled, 
            y_train_final,
            eval_set=[(X_train_final_scaled, y_train_final), (X_test_final_scaled, y_test_final)],
            verbose=False
        )
        
        print(f"XGBoost trained with {best_params['n_estimators']} estimators")
        print(f"Learning rate: {best_params['learning_rate']}")
        print(f"Max depth: {best_params['max_depth']}")
        print(f"Gamma: {best_params['gamma']}")
        print(f"Subsample: {best_params['subsample']}")
        print(f"Colsample: {best_params['colsample_bytree']}")
        
    except Exception as e:
        print(f"Warning: XGBoost failed to train with best params: {str(e)}")
        print("Using fallback parameters...")
        
        # Fallback to default XGBoost
        fallback_params = {
            'learning_rate': 0.05,
            'max_depth': 3,
            'n_estimators': 200,
            'gamma': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 0,
            'n_jobs': -1
        }
        
        model_xgb = xgb.XGBClassifier(**fallback_params)
        model_xgb.fit(
            X_train_final_scaled, 
            y_train_final,
            eval_set=[(X_train_final_scaled, y_train_final), (X_test_final_scaled, y_test_final)],
            verbose=False
        )
    
    # ========================================================================
    # SHAP ANALYSIS (PART 1 - FOR EACH WINDOW)
    # ========================================================================
    print("\n--- SHAP Analysis for Current Window ---")
    
    try:
        # Create SHAP explainer
        explainer = shap.Explainer(model_xgb, X_train_final_scaled)
        
        # Calculate SHAP values for test set
        shap_values = explainer.shap_values(X_test_final_scaled)
        
        # Store SHAP values for aggregation
        window_shap_values[f'Window {window_idx + 1}'] = shap_values
        
        # Calculate mean absolute SHAP importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance_dict = dict(zip(final_features, mean_abs_shap))
        window_feature_importance[f'Window {window_idx + 1}'] = feature_importance_dict
        
        # SHAP Summary Plot for this window
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X_test_final_scaled,
            feature_names=final_features,
            plot_type="dot",
            show=False,
            max_display=20
        )
        plt.title(f'Window {window_idx + 1} - SHAP Feature Importance', 
                 fontsize=20, fontweight='bold', pad=40)
        plt.tight_layout()
        #plt.savefig(f'/content/sample_data/shap_summary_window_{window_idx + 1}.svg', 
                   #bbox_inches='tight', dpi=150)
        plt.show()
        
        # SHAP Force Plot for a specific instance
        plt.figure(figsize=(15, 4))
        shap.force_plot(
            explainer.expected_value, 
            shap_values[0, :], 
            X_test_final_scaled[0, :],
            feature_names=final_features,
            matplotlib=True,
            text_rotation=15,
            show=False
        )
        plt.title(f'Window {window_idx + 1} - SHAP Force Plot', 
                 fontsize=20, fontweight='bold', pad=80)
        plt.tight_layout()
        #plt.savefig(f'/content/sample_data/shap_force_window_{window_idx + 1}.svg', 
                   #bbox_inches='tight', dpi=150)
        plt.show()
        
        # Store force plot data
        shap_force_plots_data[f'Window {window_idx + 1}'] = {
            'expected_value': explainer.expected_value,
            'shap_values_sample': shap_values[0, :],
            'feature_values': X_test_final_scaled[0, :]
        }
        
    except Exception as e:
        print(f"SHAP analysis failed: {str(e)}")

    
    # ========================================================================
    # PREDICTIONS AND EVALUATION
    # ========================================================================
    y_pred_test = model_xgb.predict(X_test_final_scaled)
    y_pred_train = model_xgb.predict(X_train_final_scaled)
    y_pred_prob_test = model_xgb.predict_proba(X_test_final_scaled)[:, 1]
    
    # Store predictions
    yptt.append(y_pred_test)
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test_final, y_pred_test)
    train_accuracy = accuracy_score(y_train_final, y_pred_train)
    test_precision = precision_score(y_test_final, y_pred_test, zero_division=0)
    test_recall = recall_score(y_test_final, y_pred_test, zero_division=0)
    test_f1 = f1_score(y_test_final, y_pred_test, zero_division=0)
    
    # ROC and Precision-Recall curves
    fpr, tpr, _ = roc_curve(y_test_final, y_pred_prob_test)
    roc_auc = roc_auc_score(y_test_final, y_pred_prob_test)
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_test_final, y_pred_prob_test)
    pr_auc = average_precision_score(y_test_final, y_pred_prob_test)
    
    # Store metrics
    fprs.append(fpr)
    tprs.append(tpr)
    roc_aucs.append(roc_auc)
    precisions.append(precision_curve)
    recalls.append(recall_curve)
    pr_aucs.append(pr_auc)
    
    # Store window performance
    window_performance[f'Window {window_idx + 1}'] = {
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'selected_features': final_features,
        'num_features': len(final_features),
        'best_params': {k: v for k, v in best_params.items() if k not in ['n_jobs']}
    }
    
    print(f"\nWindow {window_idx + 1} Performance:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test_final, y_pred_test)}")
    
    # XGBoost feature importance
    xgb_importance = model_xgb.feature_importances_
    sorted_idx = np.argsort(xgb_importance)[::-1]
    
    print(f"\nTop 10 Features (XGBoost Importance):")
    for i in range(min(10, len(final_features))):
        print(f"  {i+1}. {final_features[sorted_idx[i]]}: {xgb_importance[sorted_idx[i]]:.4f}")
        
# ============================================================================
# CONCATENATE ALL TEST PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("CONSOLIDATING PREDICTIONS")
print("="*80 + "\n")

for i in range(num_windows):
    y_test_window_XGB.extend(yptt[i])

print(f"Total predictions: {len(y_test_window_XGB)}")

# ============================================================================
# OVERALL MODEL PERFORMANCE
# ============================================================================
print("\n" + "="*80)
print("OVERALL MODEL PERFORMANCE - XGBOOST WITH TUNING")
print("="*80 + "\n")

y_true = df['LABEL'][initial_window:]
conf_matrix = confusion_matrix(y_true, y_test_window_XGB)
acc = accuracy_score(y_true, y_test_window_XGB)
precision = precision_score(y_true, y_test_window_XGB, zero_division=0)
recall = recall_score(y_true, y_test_window_XGB, zero_division=0)
f1 = f1_score(y_true, y_test_window_XGB, zero_division=0)
mcc = matthews_corrcoef(y_true, y_test_window_XGB)

print(f"Confusion Matrix:\n{conf_matrix}")
print(f"\nAccuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")
print(f"\n{classification_report(y_true, y_test_window_XGB, digits=4)}")

# ============================================================================
# VISUALIZATION 1: CONFUSION MATRIX
# ============================================================================
group_names = ['True Sell', 'False Buy', 'False Sell', 'True Buy']
group_counts = [f"{value:0.0f}" for value in conf_matrix.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
labels = np.asarray(labels).reshape(2, 2)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=labels, linewidths=1, fmt='', 
            cmap='Greens', annot_kws={"size": 14})
ax.set_xlabel('Predicted Labels', fontsize=15)
ax.set_ylabel('Actual Labels', fontsize=15)
ax.set_title('XGBoost - Confusion Matrix (With Dynamic CV)', fontsize=16)
plt.tight_layout()
plt.show()

# ============================================================================
# VISUALIZATION 2: DYNAMIC CV FOLDS ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("DYNAMIC CROSS-VALIDATION FOLDS ANALYSIS")
print("="*80 + "\n")

fig, ax = plt.subplots(figsize=(10, 6))
windows = np.arange(1, num_windows + 1)

ax.bar(windows, cv_folds_history, color='darkgreen', edgecolor='black', alpha=0.7)
ax.plot(windows, cv_folds_history, 'o-', color='forestgreen', linewidth=2, markersize=8)

ax.set_xlabel('Window', fontsize=12)
ax.set_ylabel('Number of CV Folds', fontsize=12)
ax.set_title('Number of Cross-Validation Folds per Window (XGBoost)', fontsize=14)
ax.set_xticks(windows)
ax.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(cv_folds_history):
    ax.text(i + 1, v + 0.1, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

print("\nCV Folds Summary for XGBoost:")
for window_idx, num_folds in enumerate(cv_folds_history):
    training_years = (initial_window + test_window * window_idx) / 252
    print(f"Window {window_idx + 1}: {training_years:.1f} years training data → {num_folds} CV folds")

# ============================================================================
# VISUALIZATION 3: HYPERPARAMETER TUNING SUMMARY
# ============================================================================
print("\n" + "="*80)
print("XGBOOST HYPERPARAMETER TUNING SUMMARY")
print("="*80 + "\n")

print("Best Parameters Selected for Each Window:")
print("-" * 60)

learning_rate_stats = {}
max_depth_stats = {}
n_estimators_stats = {}
gamma_stats = {}
subsample_stats = {}
colsample_stats = {}
min_child_weight_stats = {}
reg_alpha_stats = {}
reg_lambda_stats = {}

for window_idx, params in enumerate(best_params_history):
    print(f"\nWindow {window_idx + 1}:")
    print(f"  learning_rate: {params.get('learning_rate', 'N/A')}")
    print(f"  max_depth: {params.get('max_depth', 'N/A')}")
    print(f"  n_estimators: {params.get('n_estimators', 'N/A')}")
    print(f"  gamma: {params.get('gamma', 'N/A')}")
    print(f"  subsample: {params.get('subsample', 'N/A')}")
    print(f"  colsample_bytree: {params.get('colsample_bytree', 'N/A')}")
    print(f"  min_child_weight: {params.get('min_child_weight', 'N/A')}")
    print(f"  reg_alpha: {params.get('reg_alpha', 'N/A')}")
    print(f"  reg_lambda: {params.get('reg_lambda', 'N/A')}")
    
    # Collect statistics
    learning_rate = params.get('learning_rate', 'N/A')
    max_depth = params.get('max_depth', 'N/A')
    n_estimators = params.get('n_estimators', 'N/A')
    gamma = params.get('gamma', 'N/A')
    subsample = params.get('subsample', 'N/A')
    colsample = params.get('colsample_bytree', 'N/A')
    min_child_weight = params.get('min_child_weight', 'N/A')
    reg_alpha = params.get('reg_alpha', 'N/A')
    reg_lambda = params.get('reg_lambda', 'N/A')
    
    learning_rate_stats[learning_rate] = learning_rate_stats.get(learning_rate, 0) + 1
    max_depth_stats[max_depth] = max_depth_stats.get(max_depth, 0) + 1
    n_estimators_stats[n_estimators] = n_estimators_stats.get(n_estimators, 0) + 1
    gamma_stats[gamma] = gamma_stats.get(gamma, 0) + 1
    subsample_stats[subsample] = subsample_stats.get(subsample, 0) + 1
    colsample_stats[colsample] = colsample_stats.get(colsample, 0) + 1
    min_child_weight_stats[min_child_weight] = min_child_weight_stats.get(min_child_weight, 0) + 1
    reg_alpha_stats[reg_alpha] = reg_alpha_stats.get(reg_alpha, 0) + 1
    reg_lambda_stats[reg_lambda] = reg_lambda_stats.get(reg_lambda, 0) + 1

# Print statistics
print("\n" + "="*60)
print("XGBOOST PARAMETER SELECTION STATISTICS:")
print("="*60)

print("\nLearning Rate Selection:")
for lr, count in sorted(learning_rate_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  learning_rate={lr}: {count} windows ({percentage:.1f}%)")

print("\nMax Depth Selection:")
for depth, count in sorted(max_depth_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  max_depth={depth}: {count} windows ({percentage:.1f}%)")

print("\nNumber of Estimators Selection:")
for n_est, count in sorted(n_estimators_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  n_estimators={n_est}: {count} windows ({percentage:.1f}%)")

print("\nGamma Selection:")
for gamma, count in sorted(gamma_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  gamma={gamma}: {count} windows ({percentage:.1f}%)")

print("\nSubsample Selection:")
for subsample, count in sorted(subsample_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  subsample={subsample}: {count} windows ({percentage:.1f}%)")

print("\nColumn Sample Selection:")
for colsample, count in sorted(colsample_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  colsample_bytree={colsample}: {count} windows ({percentage:.1f}%)")

print("\nL1 Regularization (Alpha) Selection:")
for alpha, count in sorted(reg_alpha_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  reg_alpha={alpha}: {count} windows ({percentage:.1f}%)")

print("\nL2 Regularization (Lambda) Selection:")
for lambda_val, count in sorted(reg_lambda_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  reg_lambda={lambda_val}: {count} windows ({percentage:.1f}%)")

# Visualization of parameter selection
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

# Learning rate
lrs, lr_counts = zip(*sorted(learning_rate_stats.items(), key=lambda x: x[1], reverse=True))
axes[0, 0].bar([str(lr) for lr in lrs], lr_counts, color='darkgreen')
axes[0, 0].set_xlabel('Learning Rate', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Learning Rate Selection', fontsize=14)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Max depth
depths, depth_counts = zip(*sorted(max_depth_stats.items(), key=lambda x: x[1], reverse=True))
axes[0, 1].bar([str(d) for d in depths], depth_counts, color='forestgreen')
axes[0, 1].set_xlabel('Max Depth', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('Max Depth Selection', fontsize=14)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Number of estimators
n_ests, n_est_counts = zip(*sorted(n_estimators_stats.items(), key=lambda x: x[1], reverse=True))
axes[0, 2].bar([str(n) for n in n_ests], n_est_counts, color='limegreen')
axes[0, 2].set_xlabel('Number of Estimators', fontsize=12)
axes[0, 2].set_ylabel('Frequency', fontsize=12)
axes[0, 2].set_title('Number of Estimators Selection', fontsize=14)
axes[0, 2].grid(True, alpha=0.3, axis='y')

# Gamma
gammas, gamma_counts = zip(*sorted(gamma_stats.items(), key=lambda x: x[1], reverse=True))
axes[1, 0].bar([str(g) for g in gammas], gamma_counts, color='mediumseagreen')
axes[1, 0].set_xlabel('Gamma', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Gamma Selection', fontsize=14)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Subsample
subsamples, subsample_counts = zip(*sorted(subsample_stats.items(), key=lambda x: x[1], reverse=True))
axes[1, 1].bar([str(s) for s in subsamples], subsample_counts, color='seagreen')
axes[1, 1].set_xlabel('Subsample', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('Subsample Selection', fontsize=14)
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Column sample
colsamples, colsample_counts = zip(*sorted(colsample_stats.items(), key=lambda x: x[1], reverse=True))
axes[1, 2].bar([str(c) for c in colsamples], colsample_counts, color='springgreen')
axes[1, 2].set_xlabel('Column Sample', fontsize=12)
axes[1, 2].set_ylabel('Frequency', fontsize=12)
axes[1, 2].set_title('Column Sample Selection', fontsize=14)
axes[1, 2].grid(True, alpha=0.3, axis='y')

# L1 regularization (alpha)
alphas, alpha_counts = zip(*sorted(reg_alpha_stats.items(), key=lambda x: x[1], reverse=True))
axes[2, 0].bar([str(a) for a in alphas], alpha_counts, color='lightgreen')
axes[2, 0].set_xlabel('L1 Regularization (Alpha)', fontsize=12)
axes[2, 0].set_ylabel('Frequency', fontsize=12)
axes[2, 0].set_title('L1 Regularization Selection', fontsize=14)
axes[2, 0].grid(True, alpha=0.3, axis='y')

# L2 regularization (lambda)
lambdas, lambda_counts = zip(*sorted(reg_lambda_stats.items(), key=lambda x: x[1], reverse=True))
axes[2, 1].bar([str(l) for l in lambdas], lambda_counts, color='palegreen')
axes[2, 1].set_xlabel('L2 Regularization (Lambda)', fontsize=12)
axes[2, 1].set_ylabel('Frequency', fontsize=12)
axes[2, 1].set_title('L2 Regularization Selection', fontsize=14)
axes[2, 1].grid(True, alpha=0.3, axis='y')

# Min child weight
child_weights, weight_counts = zip(*sorted(min_child_weight_stats.items(), key=lambda x: x[1], reverse=True))
axes[2, 2].bar([str(w) for w in child_weights], weight_counts, color='honeydew')
axes[2, 2].set_xlabel('Min Child Weight', fontsize=12)
axes[2, 2].set_ylabel('Frequency', fontsize=12)
axes[2, 2].set_title('Min Child Weight Selection', fontsize=14)
axes[2, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================================================
# VISUALIZATION 4: CV ACCURACY TREND VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("CROSS-VALIDATION ACCURACY TREND")
print("="*80 + "\n")

if cv_accuracy_history:
    # Calculate average CV accuracy for each window
    window_cv_means = []
    window_cv_stds = []
    
    for window_accs in cv_accuracy_history:
        if window_accs:  # Check if not empty
            window_cv_means.append(np.mean(window_accs))
            window_cv_stds.append(np.std(window_accs))
    
    if window_cv_means:
        fig, ax = plt.subplots(figsize=(10, 6))
        windows = np.arange(1, len(window_cv_means) + 1)
        
        # Plot with error bars
        ax.errorbar(windows, window_cv_means, yerr=window_cv_stds, 
                   fmt='o-', linewidth=2, markersize=8, capsize=5,
                   color='darkgreen', ecolor='limegreen', elinewidth=2)
        
        ax.axhline(y=np.mean(window_cv_means), color='r', linestyle='--', alpha=0.7,
                   label=f'Overall Mean: {np.mean(window_cv_means):.4f}')
        
        ax.set_xlabel('Window Number', fontsize=12)
        ax.set_ylabel('CV Accuracy', fontsize=12)
        ax.set_title('XGBoost CV Accuracy Trend Across Windows', fontsize=14)
        ax.set_xticks(windows)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()
        
        print(f"Average CV accuracy across windows: {np.mean(window_cv_means):.4f}")
        print(f"CV accuracy range: [{np.min(window_cv_means):.4f}, {np.max(window_cv_means):.4f}]")

# ============================================================================
# VISUALIZATION 5: AGGREGATED VISUALIZATIONS (4-SUBPLOT FIGURE)
# ============================================================================
print("\n" + "="*80)
print("4-SUBPLOT COMPREHENSIVE ANALYSIS")
print("="*80 + "\n")

# Set font parameters
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 15,
    'axes.titlesize': 15,
    'axes.labelsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15
})

# Define a list of colors and markers for different sliding windows
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan']
markers = ['o', 's', 'D', '^', 'v', 'p', '*']

# Create a figure with 4 subplots
fig, axes = plt.subplots(4, 1, figsize=(8, 25))

# ========================================================================
# SUBPLOT 1: SFS Results (Training Performance)
# ========================================================================
if sfs_metrics:
    for idx, (slide, metrics) in enumerate(sfs_metrics.items()):
        if metrics:
            x_values = []
            y_values = []
            for k, v in metrics.items():
                if isinstance(k, tuple) and len(k) == 1:
                    x_values.append(k[0])
                elif isinstance(k, int):
                    x_values.append(k)
                else:
                    continue
                y_values.append(v.get('avg_score', 0))
            
            if x_values and y_values:
                axes[0].plot(
                    x_values,
                    y_values,
                    color=colors[idx % len(colors)],
                    marker=markers[idx % len(markers)],
                    markevery=10,
                    label=slide,
                    lw=2,
                    markersize=8
                )
axes[0].set_title('Training Performance', fontsize=20, fontweight='bold')
axes[0].set_xlabel('Number of Features Selected', fontsize=20, fontweight='bold')
axes[0].set_ylabel('Accuracy', fontsize=20, fontweight='bold')
axes[0].legend(fontsize=14, loc='best')
axes[0].grid(True, linestyle='--', alpha=0.7)
axes[0].tick_params(axis='both', labelsize=16)

# ========================================================================
# SUBPLOT 2: ROC Curves
# ========================================================================
for i in range(num_windows):
    axes[1].plot(fprs[i], tprs[i], color=colors[i % len(colors)],
                 markevery=10, label=f'Window {i+1} (AUC = {roc_aucs[i]:.2f})', lw=2)

# Add the diagonal line (random chance)
axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random Chance', alpha=0.7)
axes[1].set_title('Testing ROC Curves', fontsize=22, fontweight='bold', pad=20)
axes[1].set_xlabel('False Positive Rate', fontsize=20, fontweight='bold')
axes[1].set_ylabel('True Positive Rate', fontsize=20, fontweight='bold')
axes[1].legend(loc='lower right', fontsize=14)
axes[1].grid(True, linestyle='--', alpha=0.7)
axes[1].tick_params(axis='both', labelsize=16)

# ========================================================================
# SUBPLOT 3: Precision-Recall Curves
# ========================================================================
for i in range(num_windows):
    axes[2].plot(recalls[i], precisions[i], color=colors[i % len(colors)],
                 markevery=10, label=f'Window {i+1} (AUC = {pr_aucs[i]:.2f})', lw=2)
axes[2].set_title('Testing Precision-Recall Curves', fontsize=22, fontweight='bold', pad=20)
axes[2].set_xlabel('Recall', fontsize=20, fontweight='bold')
axes[2].set_ylabel('Precision', fontsize=20, fontweight='bold')
axes[2].legend(loc='lower left', fontsize=14)
axes[2].grid(True, linestyle='--', alpha=0.7)
axes[2].tick_params(axis='both', labelsize=16)

# ========================================================================
# SUBPLOT 4: Train vs Test Accuracy Bar Plot (Overfitting Checking)
# ========================================================================
axes[3].set_title('Overfitting Checking', fontsize=22, fontweight='bold', pad=20)

# Get train and test accuracies
train_accuracies = [window_performance[f'Window {i+1}']['train_accuracy'] for i in range(num_windows)]
test_accuracies = [window_performance[f'Window {i+1}']['test_accuracy'] for i in range(num_windows)]

x_pos = np.arange(num_windows)
width = 0.35

# Plot bars
bars1 = axes[3].bar(x_pos - width/2, train_accuracies, width, 
                    label='Validation Accuracy', color='blue', alpha=0.8, edgecolor='black')
bars2 = axes[3].bar(x_pos + width/2, test_accuracies, width, 
                    label='Test Accuracy', color='red', alpha=0.8, edgecolor='black')

axes[3].set_xlabel('Window', fontsize=20, fontweight='bold')
axes[3].set_ylabel('Accuracy', fontsize=20, fontweight='bold')
axes[3].set_xticks(x_pos)
axes[3].set_xticklabels([f'W{i+1}' for i in range(num_windows)], fontsize=16)
axes[3].legend(fontsize=16, loc='upper right')
axes[3].grid(True, linestyle='--', alpha=0.5, axis='y')
axes[3].set_ylim([0, 1.1])

# Add value labels on bars
def add_labels(bars, ax):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')

add_labels(bars1, axes[3])
add_labels(bars2, axes[3])

# Adjust layout to avoid overlapping
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()

print("\n✅ 4-subplot comprehensive analysis completed")

# ============================================================================
# VISUALIZATION: CV ACCURACY FOR EACH WINDOW (STACKED PLOTS)
# ============================================================================
print("\n" + "="*80)
print("CROSS-VALIDATION ACCURACY FOR EACH WINDOW (STACKED PLOTS)")
print("="*80 + "\n")

fig, axes = plt.subplots(num_windows, 1, figsize=(5, 3.5 * num_windows))

# If only one window, axes won't be an array
if num_windows == 1:
    axes = [axes]

for window_idx in range(num_windows):
    if cv_accuracy_history[window_idx]:  # Check if not empty
        fold_indices = np.arange(1, len(cv_accuracy_history[window_idx]) + 1)
        fold_accuracies = cv_accuracy_history[window_idx]
        mean_accuracy = np.mean(fold_accuracies)
        
        # Get color and marker for this window
        color = colors[window_idx % len(colors)]
        marker = markers[window_idx % len(markers)]
        
        # Plot scatter and line
        axes[window_idx].scatter(fold_indices, fold_accuracies, s=50, c=color, 
                                edgecolors='black', zorder=3, marker=marker)
        axes[window_idx].plot(fold_indices, fold_accuracies, 'o-', linewidth=1.5, 
                             markersize=0, color=color, alpha=0.5)
        axes[window_idx].axhline(y=mean_accuracy, color=color, linestyle='--', 
                                alpha=0.7, linewidth=1.5, 
                                label=f'Mean Accuracy: {mean_accuracy:.4f}')
        
        # Styling
        axes[window_idx].set_xlabel('CV Fold', fontsize=15)
        axes[window_idx].set_ylabel('Accuracy', fontsize=15)
        axes[window_idx].set_title(f'Window {window_idx + 1}: XGBoost CV Results ({len(fold_accuracies)} folds)', 
                                  fontsize=15)
        axes[window_idx].set_xticks(fold_indices)
        axes[window_idx].set_ylim([0, 1])
        axes[window_idx].grid(True, alpha=0.3)
        axes[window_idx].legend(loc='best')

plt.tight_layout()
plt.show()

# ============================================================================
# VISUALIZATION 6: Feature Importance Evolution Heatmap
# ============================================================================
print("\n" + "="*80)
print("FEATURE IMPORTANCE HEATMAP")
print("="*80 + "\n")

if window_feature_importance:
    # Collect all unique features across all windows
    all_features = set()
    for features in selected_features_per_window.values():
        all_features.update(features)
    all_features = sorted(list(all_features))
    
    # Create a matrix: rows = features, columns = windows
    importance_matrix = np.zeros((len(all_features), num_windows))
    
    for window_idx in range(num_windows):
        window_name = f'Window {window_idx + 1}'
        importance_dict = window_feature_importance.get(window_name, {})
        for feat_idx, feature in enumerate(all_features):
            if feature in importance_dict:
                importance_matrix[feat_idx, window_idx] = importance_dict[feature]
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 15))
    im = ax.imshow(importance_matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(num_windows))
    ax.set_yticks(np.arange(len(all_features)))
    ax.set_xticklabels([f'W{i+1}' for i in range(num_windows)], fontsize=20)
    ax.set_yticklabels(all_features, fontsize=15)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean |SHAP value|', fontsize=25, rotation=270, labelpad=30)
    
    # Add title and labels
    ax.set_title('Feature Importance Evolution Across \n all Expanding Windows',
                 fontsize=25, pad=20, fontweight='bold')
    ax.set_xlabel('Expanding Window', fontsize=20, fontweight='bold')
    ax.set_ylabel('Features', fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# VISUALIZATION 7: Top Features Evolution & Stability Analysis
# ============================================================================
print("\n" + "="*80)
print("TOP FEATURES EVOLUTION AND STABILITY ANALYSIS")
print("="*80 + "\n")

if window_feature_importance:
    # Define distinct colors and markers for 10 features
    colors = [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
    ]
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '>']
    
    # Prepare data
    overall_importance = {}
    for window_name, importance_dict in window_feature_importance.items():
        for feature, importance in importance_dict.items():
            if feature not in overall_importance:
                overall_importance[feature] = []
            overall_importance[feature].append(importance)
    
    mean_importance = {feat: np.mean(vals) for feat, vals in overall_importance.items()}
    top_10_features = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    top_10_feature_names = [feat[0] for feat in top_10_features]
    
    # Combine Visualization into subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 15))
    ax1, ax2 = axes
    
    # Subplot 1: Top 10 Features Evolution (Line Plot)
    for idx, feature in enumerate(top_10_feature_names):
        importance_values = []
        for window_idx in range(num_windows):
            window_name = f'Window {window_idx + 1}'
            if feature in window_feature_importance[window_name]:
                importance_values.append(window_feature_importance[window_name][feature])
            else:
                importance_values.append(0)
        
        ax1.plot(range(1, num_windows + 1), importance_values,
                 marker=markers[idx],
                 color=colors[idx],
                 label=feature,
                 linewidth=2,
                 markersize=8)
    
    ax1.set_xlabel('Expanding Window', fontsize=20, fontweight='bold')
    ax1.set_ylabel('Mean |SHAP value|', fontsize=20, fontweight='bold')
    ax1.set_title('Top 10 Features: Importance Evolution \n Across Windows',
                  fontsize=25, pad=20, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xticks(range(1, num_windows + 1))
    ax1.set_xticklabels([f'W{i}' for i in range(1, num_windows + 1)])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=3, fontsize=20, frameon=True)
    
    # Subplot 2: Feature Stability Analysis
    feature_appearance_in_top5 = {feat: 0 for feat in all_features}
    for window_name, importance_dict in window_feature_importance.items():
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_5 = [feat[0] for feat in sorted_features[:5]]
        for feat in top_5:
            feature_appearance_in_top5[feat] += 1
    
    stable_features = {k: v for k, v in feature_appearance_in_top5.items() if v > 0}
    stable_features = dict(sorted(stable_features.items(), key=lambda x: x[1], reverse=True)[:15])
    
    bars = ax2.barh(list(stable_features.keys()), list(stable_features.values()),
                    color='steelblue', edgecolor='black', linewidth=1.2)
    
    # Color bars based on stability
    for i, bar in enumerate(bars):
        val = list(stable_features.values())[i]
        if val >= 6:
            bar.set_color('darkgreen')
        elif val >= 4:
            bar.set_color('orange')
        else:
            bar.set_color('crimson')
    
    ax2.set_xlabel('Number of Times in Top 5', fontsize=20, fontweight='bold')
    ax2.set_ylabel('Features', fontsize=20, fontweight='bold')
    ax2.set_title('Feature Stability: Appearance in Top 5 \n Across Windows',
                  fontsize=25, pad=20, fontweight='bold')
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    ax2.set_xlim(0, num_windows + 0.5)
    
    # Add legend
    legend_elements = [
        Patch(facecolor='darkgreen', label='High Stability (6-7 windows)'),
        Patch(facecolor='orange', label='Medium Stability (4-5 windows)'),
        Patch(facecolor='crimson', label='Low Stability (1-3 windows)')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=20)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=1, bottom=0.1)
    plt.show()

# ============================================================================
# TRADING SIMULATION
# ============================================================================
print("\n" + "="*80)
print("TRADING SIMULATION")
print("="*80 + "\n")

initial_capital = 1000
sm1 = initial_capital
f = 0
ns = 0
nn = test_window
n = initial_window

open_prices = df['Open'].values
close_prices = df['Close'].values

modelval_XGB = []

for j in range(num_windows):
    for i in range(len(yptt[j])):
        price_idx = j * nn + i + n
        open_price = open_prices[price_idx]
        close_price = close_prices[price_idx]
        signal = yptt[j][i]
        
        if signal == 1 and f == 0:
            ns = (sm1 / (1 + cb)) / open_price
            sm1 = ns * close_price
            modelval_XGB.append(sm1)
            f = 1
        elif signal == 1 and f == 1:
            sm1 = ns * close_price
            modelval_XGB.append(sm1)
        elif signal == 0 and f == 0:
            price_change_pct = (close_price - open_price) / open_price
            sm1 = sm1 * (1 - price_change_pct)
            modelval_XGB.append(sm1)
        elif signal == 0 and f == 1:
            price_change_pct = (close_price - open_price) / open_price
            sm1 = ns * open_price * (1 - cs) * (1 - price_change_pct)
            modelval_XGB.append(sm1)
            f = 0
            ns = 0

total_return = ((modelval_XGB[-1] - initial_capital) / initial_capital) * 100

print(f"{'='*60}")
print(f"TRADING RESULTS")
print(f"{'='*60}")
print(f"Initial Investment:     ${initial_capital:.2f}")
print(f"Final Investment Value: ${modelval_XGB[-1]:.2f}")
print(f"Total Return:           {total_return:.2f}%")
print(f"Absolute Profit/Loss:   ${modelval_XGB[-1] - initial_capital:.2f}")
print(f"{'='*60}\n")

# Plot trading performance
plt.figure(figsize=(12, 6))
plt.plot(modelval_XGB, label='Investment Value', linewidth=2, color='darkgreen')
plt.axhline(y=initial_capital, color='r', linestyle='--', label='Initial Investment')
plt.xlabel('Trading Day')
plt.ylabel('Investment Value ($)')
plt.title('Trading Performance - XGBoost with Dynamic CV Tuning')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================
print("\n" + "="*80)
print("PERFORMANCE METRICS")
print("="*80 + "\n")

def calculate_performance_metrics(returns, riskfree_rate=0.02):
    """Calculate comprehensive performance metrics"""
    
    returns = pd.Series(returns)
    daily_returns = returns.pct_change().dropna()
    
    # Basic metrics
    total_return = (returns.iloc[-1] / returns.iloc[0] - 1) * 100
    annual_return = ((returns.iloc[-1] / returns.iloc[0]) ** (252/len(returns)) - 1) * 100
    
    # Maximum Drawdown
    cum_returns = (1 + daily_returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Volatility
    annual_volatility = daily_returns.std() * np.sqrt(252)
    
    # Sharpe Ratio
    excess_returns = daily_returns - riskfree_rate/252
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
    
    # Sortino Ratio
    downside_returns = daily_returns[daily_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_deviation if downside_deviation > 0 else 0
    
    # Calmar Ratio
    calmar_ratio = annual_return / abs(max_drawdown * 100) if max_drawdown != 0 else 0
    
    return {
        'Total Return (%)': total_return,
        'Annual Return (%)': annual_return,
        'Max Drawdown': max_drawdown,
        'Annual Volatility': annual_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio
    }

# Calculate metrics
metrics = calculate_performance_metrics(modelval_XGB, riskfree)

print("Performance Metrics Summary:")
print("-" * 40)
for metric, value in metrics.items():
    if 'Return' in metric or 'Drawdown' in metric:
        print(f"{metric}: {value:.2f}{'%' if '%' in metric else ''}")
    else:
        print(f"{metric}: {value:.4f}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY - XGBOOST WITH ALL VISUALIZATIONS")
print("="*80 + "\n")

print(f"Total Windows Processed: {num_windows}")
print(f"Total Predictions: {len(y_test_window_XGB)}")
print(f"Overall Accuracy: {acc:.4f}")
print(f"Overall F1-Score: {f1:.4f}")
print(f"Total Trading Return: {total_return:.2f}%")
print(f"Annual Return: {metrics['Annual Return (%)']:.2f}%")
print(f"Maximum Drawdown: {metrics['Max Drawdown']:.4f}")
print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.4f}")
print(f"Sortino Ratio: {metrics['Sortino Ratio']:.4f}")

print("\nVisualizations Generated:")
print("-" * 30)
print("1. Confusion Matrix")
print("2. Dynamic CV Folds Analysis")
print("3. Hyperparameter Tuning Summary")
print("4. CV Accuracy Trend")
print("5. 4-Subplot Comprehensive Analysis (SFS, ROC, PR, Overfitting)")
print("6. Feature Importance Heatmap")
print("7. Top Features Evolution & Stability Analysis")
print("8. Trading Performance Plot")
print("9. SHAP Visualizations for each window")

print("\nModel Insights:")
print("-" * 30)
# Get most common parameters
for param_name in ['learning_rate', 'max_depth', 'n_estimators']:
    values = [params.get(param_name) for params in best_params_history]
    if values:
        most_common = max(set(values), key=values.count)
        print(f"{param_name}: {most_common}")

# ============================================================================
# EXECUTION TIME
# ============================================================================
end = time.time()
execution_time = (end - start) / 60
print("\n" + "="*80)
print(f"Total Execution Time: {execution_time:.2f} minutes")
print("="*80)

