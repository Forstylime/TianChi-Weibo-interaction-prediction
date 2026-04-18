import numpy as np
import pandas as pd

def calculate_weibo_score(real_df, pred_df, verbose=True):
    """
    根据比赛官方公式计算得分
    real_df: 包含真实值的 DataFrame (必须有 forward_count, comment_count, like_count)
    pred_df: 包含预测值的 DataFrame (必须有 forward_count, comment_count, like_count)
    注：输入的数据必须是还原后的原始整数数值，不能是 log 转换后的值！
    """
    
    # 提取真实值和预测值
    fr, cr, lr = real_df['forward_count'].values, real_df['comment_count'].values, real_df['like_count'].values
    fp, cp, lp = pred_df['forward_count'].values, pred_df['comment_count'].values, pred_df['like_count'].values
    
    # 1. 计算各项偏差 (Deviation)
    dev_f = np.abs(fp - fr) / (fr + 5)
    dev_c = np.abs(cp - cr) / (cr + 3)
    dev_l = np.abs(lp - lr) / (lr + 3)
    
    # 2. 计算每篇博文的准确率 (Precision_i)
    precision_i = 1 - 0.5 * dev_f - 0.25 * dev_c - 0.25 * dev_l
    
    # 3. 计算改进的符号函数 sgn(precision_i - 0.8)
    # 准确率大于0.8记为1，否则记为0
    sgn_val = np.where(precision_i > 0.8, 1, 0)
    
    # 4. 计算博文的总互动数并截断 (count_i)
    count_i = fr + cr + lr
    # 官方规则：当 count_i > 100 时，取值为 100
    count_i_capped = np.clip(count_i, 0, 100)
    
    # 5. 计算最终的总分
    numerator = np.sum((count_i_capped + 1) * sgn_val)
    denominator = np.sum(count_i_capped + 1)
    
    final_score = numerator / denominator if denominator > 0 else 0

    if verbose:
        hit_rate = sgn_val.mean()
        high_eng = count_i > 20
        print(f"  总分: {final_score*100:.4f}%")
        print(f"  整体命中率(precision>0.8): {hit_rate*100:.2f}%")
        print(f"  高互动样本命中率: {sgn_val[high_eng].mean()*100:.2f}% ({high_eng.sum()}条)")
        print(f"  precision_i 均值: {precision_i.mean():.4f}, 中位数: {np.median(precision_i):.4f}")
    
    return final_score

# ==========================================
# 如何使用：
# ==========================================
# 假设你在 Step 4 跑完了代码，得到了 valid_set (包含真实值) 和 lgbm_val_preds (包含预测值)

# real_data = valid_set[['forward_count', 'comment_count', 'like_count']]
# pred_data = lgbm_val_preds[['forward_count', 'comment_count', 'like_count']]

# score = calculate_weibo_score(real_data, pred_data)