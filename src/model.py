import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Features we CANNOT use for training (IDs, raw text, raw time, and the raw/untransformed targets)
DROP_COLS = ['uid', 'mid', 'time', 'content', 
             'forward_count', 'comment_count', 'like_count',
             'forward_count_clipped', 'comment_count_clipped', 'like_count_clipped']

# The transformed targets we are actually trying to predict
TARGETS = ['forward_count_log', 'comment_count_log', 'like_count_log']

# =========================================================
# 1. Baseline Model (Rule-Based: User Historical Average)
# =========================================================
def baseline_model_predict(df_test):
    """
    Baseline: Simply predict the user's historical average.
    Since past_avg was calculated on raw counts, no log/exp transformation is needed here.
    """
    print("--- Running Baseline Model ---")
    predictions = pd.DataFrame({'uid': df_test['uid'], 'mid': df_test['mid']})
    
    for col in ['forward_count', 'comment_count', 'like_count']:
        # Predict past average, rounded to integer
        predictions[col] = np.round(df_test[f'user_past_avg_{col}']).astype(int)
        
    return predictions

# =========================================================
# 2. Setup Time-Based Validation Split
# =========================================================
def get_time_split(df_train):
    """
    Splits the training data chronologically.
    Train: Feb 1, 2015 - June 30, 2015
    Valid: July 1, 2015 - July 31, 2015
    """
    print("--- Creating Time-Based Split ---")
    # Define the split date (Last month of train set is for validation)
    split_date = '2015-07-01'
    
    trn_mask = df_train['time'] < split_date
    val_mask = df_train['time'] >= split_date
    
    train_set = df_train[trn_mask].copy()
    valid_set = df_train[val_mask].copy()
    
    print(f"Training set shape (Feb-Jun): {train_set.shape}")
    print(f"Validation set shape (Jul): {valid_set.shape}")
    
    return train_set, valid_set

# =========================================================
# 3. Train Separate LightGBM Models
# =========================================================
def train_lgbm_models(train_set, valid_set, test_set):
    """
    Trains 3 separate LightGBM models (Forwards, Comments, Likes)
    """
    print("\n--- Training LightGBM Models ---")
    
    # Define features to train on
    features = [c for c in train_set.columns if c not in DROP_COLS + TARGETS]
    print(f"Number of features used: {len(features)}")
    
    # Base Hyperparameters (These can be tuned later)
    lgb_params = {
        'objective': 'regression',
        'metric': 'mae',          # Mean Absolute Error is robust to outliers
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'n_jobs': -1,
        'random_state': 42
    }
    
    models = {}
    test_predictions = pd.DataFrame({'uid': test_set['uid'], 'mid': test_set['mid']})
    valid_predictions = pd.DataFrame({'uid': valid_set['uid'], 'mid': valid_set['mid']})
    
    for target in TARGETS:
        print(f"\n>>> Training for target: {target} <<<")
        
        # Prepare LightGBM Datasets
        dtrain = lgb.Dataset(train_set[features], label=train_set[target])
        dvalid = lgb.Dataset(valid_set[features], label=valid_set[target], reference=dtrain)
        
        # Train Model with Early Stopping
        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dtrain, dvalid],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
        )
        models[target] = model
        
        # Predict on Validation (for ensembling/checking later)
        val_pred_log = model.predict(valid_set[features], num_iteration=model.best_iteration)
        # Convert log predictions back to raw counts: exp(pred) - 1
        valid_predictions[target] = np.expm1(val_pred_log)
        
        # Predict on Test
        test_pred_log = model.predict(test_set[features], num_iteration=model.best_iteration)
        test_predictions[target] = np.expm1(test_pred_log)

    return models, valid_predictions, test_predictions

# =========================================================
# 4. Ensembling / Model Fusion (LGBM + XGBoost)
# =========================================================
def train_xgboost_and_ensemble(train_set, valid_set, test_set, lgb_test_preds):
    """
    Trains XGBoost models and fuses predictions with LightGBM.
    """
    print("\n--- Training XGBoost Models for Ensembling ---")
    features = [c for c in train_set.columns if c not in DROP_COLS + TARGETS]
    
    xgb_params = {
        'objective': 'reg:absoluteerror', # MAE
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    final_ensemble_preds = pd.DataFrame({'uid': test_set['uid'], 'mid': test_set['mid']})
    
    for target in TARGETS:
        print(f"Training XGBoost for {target}...")
        model = xgb.XGBRegressor(**xgb_params, n_estimators=500, early_stopping_rounds=50)
        
        model.fit(
            train_set[features], train_set[target],
            eval_set=[(valid_set[features], valid_set[target])],
            verbose=False
        )
        
        # Predict Test
        xgb_pred_log = model.predict(test_set[features])
        xgb_pred = np.expm1(xgb_pred_log)
        
        # --- FUSION / BLENDING ---
        # 60% LightGBM, 40% XGBoost (Weights can be tuned)
        lgb_pred = lgb_test_preds[target].values
        fused_pred = (0.6 * lgb_pred) + (0.4 * xgb_pred)
        
        # Ensure no negative predictions and round to integers
        final_ensemble_preds[target] = np.clip(np.round(fused_pred), 0, None).astype(int)
        
    return final_ensemble_preds

# ==========================================
# How to run the pipeline:
# ==========================================
# 1. Get baseline (just to see how simple rules look)
# df_baseline_preds = baseline_model_predict(df_test_final)

# 2. Split data chronologically
# train_set, valid_set = get_time_split(df_train_final)

# 3. Train LightGBM models
# lgbm_models, lgbm_val_preds, lgbm_test_preds = train_lgbm_models(train_set, valid_set, df_test_final)

# 4. Train XGBoost and fuse them together!
# final_test_predictions = train_xgboost_and_ensemble(train_set, valid_set, df_test_final, lgbm_test_preds)