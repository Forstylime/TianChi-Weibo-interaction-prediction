import os
import sys
import pickle
import pandas as pd
import numpy as np

# 将项目根目录添加到 Python 路径中，确保可以从 src 和 utils 文件夹导入模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 正确导入各个模块
from utils.data_process import preprocess_data
from src.feature_eng import create_features
from src.model import run_pipeline
from utils.calculate_score import calculate_weibo_score
from utils.submission import generate_submission

# ==================== 1. 配置参数 ====================
TRAIN_DATA_PATH = 'dataset/weibo_train_data.txt'
PREDICT_DATA_PATH = 'dataset/weibo_predict_data.txt'
TRAIN_PKL_PATH = 'dataset/weibo_train_features.pkl'
PREDICT_PKL_PATH = 'dataset/weibo_predict_features.pkl'
SUBMISSION_PATH = 'submission/submission_latest.txt'

def main():
    # ==================== 2. 数据加载与特征工程 (带缓存机制) ====================
    if os.path.exists(TRAIN_PKL_PATH) and os.path.exists(PREDICT_PKL_PATH):
        print(">>> 检测到缓存文件，正在加载预处理好的特征...")
        with open(TRAIN_PKL_PATH, 'rb') as f:
            df_train_final = pickle.load(f)
        with open(PREDICT_PKL_PATH, 'rb') as f:
            df_test_final = pickle.load(f)
    else:
        print(">>> 未检测到缓存，开始从原始数据生成特征...")
        print("正在读取原始 txt 数据...")
        df_train = pd.read_csv(TRAIN_DATA_PATH, sep='\t', header=None, 
                               names=['uid', 'mid', 'time', 'forward_count', 'comment_count', 'like_count', 'content'])
        df_test = pd.read_csv(PREDICT_DATA_PATH, sep='\t', header=None, 
                              names=['uid', 'mid', 'time', 'content'])
            
        print("正在进行数据清洗...")
        df_train_clean = preprocess_data(df_train, is_train=True)
        df_test_clean = preprocess_data(df_test, is_train=False)
            
        print("正在执行特征工程 (这可能需要一些时间)...")
        df_train_final, df_test_final = create_features(df_train_clean, df_test_clean)
            
        print(f"正在保存特征至: {TRAIN_PKL_PATH}")
        os.makedirs('dataset', exist_ok=True)
        with open(TRAIN_PKL_PATH, 'wb') as f:
            pickle.dump(df_train_final, f)
        with open(PREDICT_PKL_PATH, 'wb') as f:
            pickle.dump(df_test_final, f)

    print(f"训练集规模: {df_train_final.shape}")
    print(f"预测集规模: {df_test_final.shape}")

    # ==================== 3. 运行模型流水线 ====================
    # run_pipeline 包含了：时间切分、Baseline计算、LGBM训练、XGB融合
    print("\n>>> 进入模型训练与预测流水线...")
    valid_set, baseline_val_preds, lgbm_val_preds, final_test_predictions = run_pipeline(
        df_train_final, df_test_final
    )

    # ==================== 4. 本地验证评估 ====================
    print("\n=== 开始本地验证集评估 ===")
    
    # 提取真值
    real_valid_data = valid_set[['forward_count', 'comment_count', 'like_count']]

    # (1) 评估 Baseline 模型
    baseline_score = calculate_weibo_score(real_valid_data, baseline_val_preds)
    print(f"\n[Baseline] 基于历史平均值的得分: {baseline_score*100:.4f}")

    # (2) 评估 LightGBM 模型并寻找最佳阈值
    from src.model import optimize_thresholds, apply_post_processing
    
    # 在未经处理的连续值验证集预测上搜索最佳阈值
    best_thresholds = optimize_thresholds(real_valid_data, lgbm_val_preds)
    t_f, t_c, t_l = best_thresholds
    
    # 使用最佳阈值对 LGBM 验证集预测结果进行后处理
    lgbm_val_preds_processed = apply_post_processing(lgbm_val_preds, t_f, t_c, t_l)
    
    lgbm_val_preds_renamed = lgbm_val_preds_processed.rename(columns={
        'forward_count_log': 'forward_count',
        'comment_count_log': 'comment_count',
        'like_count_log': 'like_count'
    })
    
    lgbm_score = calculate_weibo_score(real_valid_data, lgbm_val_preds_renamed)
    print(f"\n[LightGBM] 机器学习模型 (最佳阈值后处理) 的得分: {lgbm_score*100:.4f}")

    # 计算提升
    improvement = (lgbm_score - baseline_score) * 100
    print(f"\n🚀 模型相对 Baseline 提升: {improvement:.4f}%")

    # ==================== 5. 生成提交文件 ====================
    print("\n>>> 正在生成最终提交文件...")
    
    # 使用刚找到的最佳阈值对最终的测试集预测结果(LGBM+XGBoost融合后)进行后处理
    final_test_predictions_processed = apply_post_processing(final_test_predictions, t_f, t_c, t_l)
    
    # 对齐提交要求的列名（去掉 _log）
    final_test_predictions_renamed = final_test_predictions_processed.rename(columns={
        'forward_count_log': 'forward_count',
        'comment_count_log': 'comment_count',
        'like_count_log': 'like_count'
    })

    # 确保保存目录存在
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    
    generate_submission(final_test_predictions_renamed, SUBMISSION_PATH)
    
    print("-" * 30)
    print(f"✅ 成功! 提交文件已保存在: {SUBMISSION_PATH}")
    print("-" * 30)

if __name__ == "__main__":
    main()