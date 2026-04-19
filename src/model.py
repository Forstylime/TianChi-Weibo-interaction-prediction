import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from utils.calculate_score import calculate_weibo_score

# 训练中不可使用的列（ID类、原始文本、原始时间、以及未转换的标签）
DROP_COLS = ['uid', 'mid', 'time', 'content', 
             'forward_count', 'comment_count', 'like_count',
             'forward_count_clipped', 'comment_count_clipped', 'like_count_clipped']

# 预测的目标列（对数转换后的）
TARGETS = ['forward_count_log', 'comment_count_log', 'like_count_log']

# =========================================================
# 1. Baseline Model (基于用户历史平均值的规则模型)
# =========================================================
def baseline_model_predict(df_test):
    """
    Baseline: 直接预测用户历史平均水平。
    """
    print("--- Running Baseline Model Prediction ---")
    predictions = pd.DataFrame({'uid': df_test['uid'], 'mid': df_test['mid']})
    
    for col in ['forward_count', 'comment_count', 'like_count']:
        # 预测历史平均值，并四舍五入取整
        predictions[col] = np.round(df_test[f'user_past_avg_{col}']).astype(int)
        
    return predictions

# =========================================================
# 2. 时间切分验证集
# =========================================================
def get_time_split(df_train):
    """
    按时间顺序切分数据：
    训练集: 2015-02-01 至 2015-06-30
    验证集: 2015-07-01 至 2015-07-31
    """
    print("--- Creating Time-Based Split ---")
    split_date = '2015-07-01'
    
    trn_mask = df_train['time'] < split_date
    val_mask = df_train['time'] >= split_date
    
    train_set = df_train[trn_mask].copy()
    valid_set = df_train[val_mask].copy()
    
    print(f"Training set shape: {train_set.shape}")
    print(f"Validation set shape: {valid_set.shape}")
    
    return train_set, valid_set

# =========================================================
# 3. 训练 LightGBM 模型
# =========================================================
def train_lgbm_models(train_set, valid_set, test_set):
    """
    训练 3 个独立的 LightGBM 模型（转发、评论、点赞）
    """
    print("\n--- Training LightGBM Models ---")
    
    features = [c for c in train_set.columns if c not in DROP_COLS + TARGETS]
    print(f"Using {len(features)} features for training.")
    
    lgb_params = {
        'objective': 'regression',
        'metric': 'mae',
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
        print(f"Training LGBM for: {target}")
        
        dtrain = lgb.Dataset(train_set[features], label=train_set[target])
        dvalid = lgb.Dataset(valid_set[features], label=valid_set[target], reference=dtrain)
        
        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dtrain, dvalid],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
        )
        models[target] = model
        
        # 验证集预测并还原对数
        val_pred_log = model.predict(valid_set[features], num_iteration=model.best_iteration)
        valid_predictions[target] = np.expm1(val_pred_log)
        
        # 测试集预测并还原对数
        test_pred_log = model.predict(test_set[features], num_iteration=model.best_iteration)
        test_predictions[target] = np.expm1(test_pred_log)

    return models, valid_predictions, test_predictions

# =========================================================
# 4. 训练 XGBoost 并进行模型融合 (LGBM + XGBoost)
# =========================================================
def train_xgboost_and_ensemble(train_set, valid_set, test_set, lgb_test_preds):
    """
    训练 XGBoost 模型并与 LightGBM 的预测结果按比例融合。
    """
    print("\n--- Training XGBoost Models & Ensembling ---")
    features = [c for c in train_set.columns if c not in DROP_COLS + TARGETS]
    
    xgb_params = {
        'objective': 'reg:absoluteerror',
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
        
        # XGBoost 预测并还原
        xgb_pred_log = model.predict(test_set[features])
        xgb_pred = np.expm1(xgb_pred_log)
        
        # 模型融合：60% LGBM + 40% XGBoost
        lgb_pred = lgb_test_preds[target].values
        fused_pred = (0.6 * lgb_pred) + (0.4 * xgb_pred)
        
        # 确保预测值不为负数 (暂不取整，留给后处理)
        final_ensemble_preds[target] = np.clip(fused_pred, 0, None)
        
    return final_ensemble_preds

# =========================================================
# 5. 后处理与阈值优化
# =========================================================
def apply_post_processing(preds, t_f, t_c, t_l):
    """
    根据设定的阈值进行后处理：
    1. 连续值预测如果 < threshold，置为 0。
    2. 连续值预测如果 >= threshold 且 < 1，置为 1。
    3. 连续值预测如果 >= 1，使用向下取整 (math.floor)。
    4. 连带压制规则：如果点赞数为 0，则评论数和转发数强制置为 0。
    """
    res = preds.copy()
    
    thresholds = {'forward_count': t_f, 'comment_count': t_c, 'like_count': t_l}
    
    for col in ['forward_count', 'comment_count', 'like_count']:
        if col not in res.columns:
            target_col = col + '_log' if col + '_log' in res.columns else col
        else:
            target_col = col
            
        val = res[target_col].values.astype(float)
        new_val = np.floor(val)
        
        t = thresholds[col]
        # values between t and 1 become 1
        new_val[(val >= t) & (val < 1.0)] = 1.0
        # values below t become 0
        new_val[val < t] = 0.0
        
        res[target_col] = new_val.astype(int)
        
    # 连带压制规则
    like_col = 'like_count_log' if 'like_count_log' in res.columns else 'like_count'
    comment_col = 'comment_count_log' if 'comment_count_log' in res.columns else 'comment_count'
    forward_col = 'forward_count_log' if 'forward_count_log' in res.columns else 'forward_count'
    
    mask_like_0 = res[like_col] == 0
    res.loc[mask_like_0, comment_col] = 0
    res.loc[mask_like_0, forward_col] = 0
    
    return res

def optimize_thresholds(real_valid_data, valid_preds):
    """
    在验证集上搜索最佳截断阈值，最大化官方评分。
    """
    print("\n--- Optimizing Thresholds ---")
    best_score = -1.0
    best_thresholds = (0.5, 0.5, 0.5)
    
    # 定义搜索空间 (0.5 到 0.8，步长 0.05)
    search_space = np.arange(0.5, 0.85, 0.05)
    
    # 暂存用于评分的 DataFrame，因为评分函数期望特定的列名
    eval_preds = valid_preds.copy()
    if 'forward_count_log' in eval_preds.columns:
        eval_preds = eval_preds.rename(columns={
            'forward_count_log': 'forward_count',
            'comment_count_log': 'comment_count',
            'like_count_log': 'like_count'
        })
        
    for t_f in search_space:
        for t_c in search_space:
            for t_l in search_space:
                # 应用后处理
                processed_preds = apply_post_processing(eval_preds, t_f, t_c, t_l)
                
                # 计算得分 (verbose=False 避免刷屏)
                score = calculate_weibo_score(real_valid_data, processed_preds, verbose=False)
                
                if score > best_score:
                    best_score = score
                    best_thresholds = (t_f, t_c, t_l)
                    
    print(f"Best Thresholds Found - Forward: {best_thresholds[0]:.2f}, Comment: {best_thresholds[1]:.2f}, Like: {best_thresholds[2]:.2f}")
    print(f"Best Validation Score: {best_score*100:.4f}")
    
    return best_thresholds

# =========================================================
# 6. 集成 Pipeline 函数
# =========================================================
def run_pipeline(df_train_final, df_test_final):
    """
    一键运行：切分数据 -> Baseline -> LGBM -> XGBoost融合
    返回：验证集真实值, Baseline验证结果, LGBM验证结果, 最终融合测试集预测结果
    """
    # 1. 划分验证集
    train_set, valid_set = get_time_split(df_train_final)
    
    # 2. 计算 Baseline (用于本地验证)
    baseline_val_preds = baseline_model_predict(valid_set)
    
    # 3. 训练 LightGBM
    lgbm_models, lgbm_val_preds, lgbm_test_preds = train_lgbm_models(train_set, valid_set, df_test_final)
    
    # 4. 训练 XGBoost 并融合生成最终提交结果
    final_test_predictions = train_xgboost_and_ensemble(train_set, valid_set, df_test_final, lgbm_test_preds)
    
    return valid_set, baseline_val_preds, lgbm_val_preds, final_test_predictions