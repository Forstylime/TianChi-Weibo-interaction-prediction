import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import warnings
from calculate_score import calculate_weibo_score
warnings.filterwarnings('ignore')

DROP_COLS = ['uid', 'mid', 'time', 'content',
             'forward_count', 'comment_count', 'like_count',
             'forward_count_clipped', 'comment_count_clipped', 'like_count_clipped']
TARGETS = ['forward_count_log', 'comment_count_log', 'like_count_log']
TARGET_RAW = ['forward_count', 'comment_count', 'like_count']

# =========================================================
# 样本权重（直接对齐评分公式）
# =========================================================
def get_sample_weights(df):
    count_i = (df['forward_count'] + df['comment_count'] + df['like_count']).clip(upper=100)
    return (count_i + 1).values

# =========================================================
# 时间划分
# =========================================================
def get_time_split(df_train):
    split_date = '2015-07-01'
    train_set = df_train[df_train['time'] < split_date].copy()
    valid_set = df_train[df_train['time'] >= split_date].copy()
    print(f"Train: {train_set.shape}, Valid: {valid_set.shape}")
    return train_set, valid_set

# =========================================================
# LightGBM（加入样本权重）
# =========================================================
def train_lgbm_models(train_set, valid_set, test_set):
    print("\n--- Training LightGBM ---")
    features = [c for c in train_set.columns if c not in DROP_COLS + TARGETS]
    print(f"Features: {len(features)}")

    lgb_params = {
        'objective': 'regression_l1',  # MAE，对偏差类指标更好
        'metric': 'mae',
        'learning_rate': 0.05,
        'num_leaves': 63,           # 适当加大
        'max_depth': 7,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'lambda_l1': 0.1,
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    }

    # 样本权重
    train_weights = get_sample_weights(train_set)
    valid_weights = get_sample_weights(valid_set)

    models = {}
    val_preds_raw = pd.DataFrame({'uid': valid_set['uid'], 'mid': valid_set['mid']})
    test_preds_raw = pd.DataFrame({'uid': test_set['uid'], 'mid': test_set['mid']})

    for target in TARGETS:
        col = target.replace('_log', '')
        print(f"\n>>> {target} <<<")

        dtrain = lgb.Dataset(train_set[features], label=train_set[target], weight=train_weights)
        dvalid = lgb.Dataset(valid_set[features], label=valid_set[target], weight=valid_weights, reference=dtrain)

        model = lgb.train(
            lgb_params, dtrain,
            num_boost_round=2000,
            valid_sets=[dtrain, dvalid],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
        )
        models[target] = model

        val_preds_raw[col] = np.expm1(model.predict(valid_set[features])).clip(0).round().astype(int)
        test_preds_raw[col] = np.expm1(model.predict(test_set[features])).clip(0).round().astype(int)

    print("\n[LightGBM 验证集得分]")
    calculate_weibo_score(valid_set, val_preds_raw)

    return models, val_preds_raw, test_preds_raw

# =========================================================
# XGBoost
# =========================================================
def train_xgboost_models(train_set, valid_set, test_set):
    print("\n--- Training XGBoost ---")
    features = [c for c in train_set.columns if c not in DROP_COLS + TARGETS]
    train_weights = get_sample_weights(train_set)

    xgb_params = {
        'objective': 'reg:absoluteerror',
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'n_estimators': 2000,
        'early_stopping_rounds': 100
    }

    val_preds_raw = pd.DataFrame({'uid': valid_set['uid'], 'mid': valid_set['mid']})
    test_preds_raw = pd.DataFrame({'uid': test_set['uid'], 'mid': test_set['mid']})

    for target in TARGETS:
        col = target.replace('_log', '')
        print(f">>> {target} <<<")
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(
            train_set[features], train_set[target],
            sample_weight=train_weights,
            eval_set=[(valid_set[features], valid_set[target])],
            verbose=False
        )
        val_preds_raw[col] = np.expm1(model.predict(valid_set[features])).clip(0).round().astype(int)
        test_preds_raw[col] = np.expm1(model.predict(test_set[features])).clip(0).round().astype(int)

    print("\n[XGBoost 验证集得分]")
    calculate_weibo_score(valid_set, val_preds_raw)

    return val_preds_raw, test_preds_raw

# =========================================================
# 网格搜索最优融合权重
# =========================================================
def find_best_blend(valid_set, lgb_val, xgb_val):
    print("\n--- 搜索最优融合权重 ---")
    best_score, best_w = 0, 0.5

    for w in np.arange(0.3, 0.85, 0.05):
        blended = pd.DataFrame({'uid': lgb_val['uid'], 'mid': lgb_val['mid']})
        for col in TARGET_RAW:
            blended[col] = (w * lgb_val[col] + (1-w) * xgb_val[col]).clip(0).round().astype(int)
        score = calculate_weibo_score(valid_set, blended, verbose=False)
        print(f"  LGB权重={w:.2f}: {score*100:.4f}%")
        if score > best_score:
            best_score, best_w = score, w

    print(f"\n最优权重: LGB={best_w:.2f}, XGB={1-best_w:.2f}, 得分={best_score*100:.4f}%")
    return best_w

# =========================================================
# 主流程
# =========================================================
def run_pipeline(df_train, df_test):
    train_set, valid_set = get_time_split(df_train)

    # 训练
    lgb_models, lgb_val, lgb_test = train_lgbm_models(train_set, valid_set, df_test)
    xgb_val, xgb_test = train_xgboost_models(train_set, valid_set, df_test)

    # 搜索最优权重
    best_w = find_best_blend(valid_set, lgb_val, xgb_val)

    # 生成最终提交
    final_preds = pd.DataFrame({'uid': df_test['uid'], 'mid': df_test['mid']})
    for col in TARGET_RAW:
        final_preds[col] = (best_w * lgb_test[col] + (1-best_w) * xgb_test[col]).clip(0).round().astype(int)

    print("\n[最终融合验证集得分]")
    final_val = pd.DataFrame({'uid': valid_set['uid'], 'mid': valid_set['mid']})
    for col in TARGET_RAW:
        final_val[col] = (best_w * lgb_val[col] + (1-best_w) * xgb_val[col]).clip(0).round().astype(int)
    calculate_weibo_score(valid_set, final_val)

    return final_preds