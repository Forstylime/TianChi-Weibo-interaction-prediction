# 🚀 Weibo Interaction Prediction: Technical Summary

This document provides a comprehensive overview of the current technical architecture and feature engineering strategies implemented to predict Weibo interactions (Forward, Comment, and Like counts).

---

## 🛠️ 1. Feature Engineering Architecture

The feature set has been significantly expanded to capture complex user behavioral patterns and content semantics.

### 🧩 A. High-Dimensional NLP Features
- **Semantic Extraction**: Uses `TfidfVectorizer` (char-level n-grams, 1-2) with a vocabulary of 1,000.
- **Dimensionality Reduction**: Compresses text features into **64 latent semantic components** using `TruncatedSVD`. This preserves significantly more semantic context than previous iterations (v1.0 used 5).

### 🕰️ B. Multi-Scale Temporal Features
- **Decomposition**: Extraction of `hour`, `day_of_week`, `month`, and `week_of_year`.
- **Behavioral Bins**: `is_weekend` flag and a 6-bin `time_segment` (Dawn, Morning, Afternoon, Evening, Peak, Night) to capture diurnal activity variations.

### 📝 C. Content & Meta Features
- **Structural Metrics**: `content_len`, `url_count`, `hashtag_count`, and `mention_count`.
- **Keyword Triggers**: Regex-based detection for:
  - `has_lucky_draw`: Keywords like "抽奖", "红包", "转发抽".
  - `has_title_bracket`: Formal content markers like "【】" or "《》".
  - `has_question_mark`: Interactive or rhetorical queries.

### 📊 D. Advanced User Historical Statistics
- **Historical Benchmarks**: `user_past_avg`, `user_past_q90`, and `user_past_median` calculated cumulatively (Expanding Window) to prevent data leakage.
- **Time-Window Dynamics**: Rolling statistics for **3d, 7d, 14d, and 30d** windows (Mean, Max, Median) to capture "recency" and "viral" bursts.
- **Personalized Context**: `user_recent_5_avg` and `time_diff_from_last` (seconds since previous post).
- **Cold Start Strategy**:
  - Test set users are initialized with their **final training profile** (latest state).
  - Truly new users utilize a **Global Median** fallback derived from the training set.

---

## 🤖 2. Modeling & Training Strategy

The pipeline shifted from a simple ensemble to a sophisticated **Target Chaining** approach with **Sample Weighting**.

### ⛓️ A. K-Fold Target Chaining
To capture the correlations between interaction types (e.g., people who Like are more likely to Comment), the model predicts targets in a specific order:
1. **Like** $\rightarrow$ 2. **Comment** $\rightarrow$ 3. **Forward**
- **Meta-Features**: Out-of-Fold (OOF) predictions from previous stages are fed as features into the subsequent models.
- **Cross-Validation**: 5-Fold K-Fold ensures robust OOF generation and prevents overfitting.

### ⚖️ B. Formula-Aligned Sample Weighting
Models are trained with custom weights to align with the official evaluation metric:
$$Weight = \log(1 + \text{clip}(\text{Total Interactions}, 0, 100)) + 1.0$$
- This strategy prioritizes accuracy on high-engagement posts (which carry more weight in the score) while using log-smoothing to prevent extreme outliers from destabilizing the gradient.

### 🎭 C. Model Fusion
The final prediction is a weighted ensemble:
- **XGBoost (90%)**: Primary regressor using `reg:absoluteerror`.
- **LightGBM (10%)**: Complementary regressor using `MAE` objective.
- **Target Transformation**: All counts are **99.9th percentile clipped** and **Log1p transformed** before training.

---

## 📐 3. Validation & Post-Processing

### 🔍 A. Local Validation
- **Time-Based Split**: Training on Feb–June data, Validating on July data.
- **Scoring**: Official competition formula integrated for local performance tracking.

### ✂️ B. Dynamic Threshold Optimization
Predictions are refined through a threshold-based post-processing layer:
- **Search**: The pipeline automatically searches for the best truncation thresholds ($t_f, t_c, t_l$) to maximize the final score.
- **Logic**:
  - If $Pred < Threshold \rightarrow 0$
  - If $Threshold \leq Pred < 1 \rightarrow 1$
  - If $Pred \geq 1 \rightarrow \lfloor Pred \rfloor$

---

## 📈 4. Score Evolution Record

| Version | Validation Score | Absolute Improvement | Submission File | Official Score | Key changes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline | 27.5265% | - | - | - | |
| V 1.0 | 30.9287% | 3.4022% | submission_0.0.txt | 0.2950 | 添加特征 |
| V 1.1 | 30.6816% | 3.1551% | submission_1.2.txt | 0.3020 | 添加特征|
| V 1.2 | 31.3058% | 3.7792% | submission_1.3.txt | 0.3036 | 增加时间窗口特征 |
| V 1.3 | 31.0624% | 3.5359% | submission_1.4.txt | 0.3032 | SVD维度5 -> 64 |
| V 1.4 | 31.3058% | 3.7792% | submission_1.5.txt | 0.3036 | - |
| V 2.0 | 31.3092% | 3.7827% | submission_2.0.txt | 0.3000 | 增加后处理 |
| V 2.1 | 31.5589% | 4.0324% | submission_2.1.txt | 0.3031 | - |
| V 2.2 | 31.3829% | 3.8563% | submission_2.2.txt | 0.3073 | 修改模型权重 |
| V 2.3 | 31.7122% | 4.1857% | submission_2.3.txt | 0.3116 | 引入样本加权 |
| V 2.4 | 32.5763% | 5.0498% | submission_2.4.txt | 0.3102 | 增加 Stacking |
| V 3.0 | 31.7122% | 4.1857% | submission_3.0.txt | 0.3195 | 只用Xgboost，全量训练 |
| 🏆 **V 3.1** | 31.7122% | 4.1857% | submission_3.1.txt | **0.3199** | 融合：10%LGBM+90%XGBoost |