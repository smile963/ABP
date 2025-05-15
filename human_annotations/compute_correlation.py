

import json
import numpy as np
from scipy.stats import kendalltau

# 读取 JSON 文件
with open('human_annotations.json') as f:
    data = json.load(f)

def compute_correlation(metric1, metric2):
    metric1_scores = []
    # num1 = 0
    # num22 = 0
    
    # for score in data:
    #     # 获取 metric1 的前两个值
    #     score1 = score.get(metric1, np.nan)[0]
    #     score2 = score.get(metric1, np.nan)[1]
    #     score3 = score.get(metric1, np.nan)[2]
    #     # 如果两个值的差异大于等于 2，则设置为 NaN，否则计算平均值
    #     if max(score1, score2, score3) - min(score1, score2, score3) >= 2:
    #         metric1_scores.append(np.nan)
    #         num1 += 1
    #     else:
    #         metric1_scores.append((score1 + score2 + score3) / 3.0)
    #         num22 += 1
    metric1_scores = [score.get(metric1, np.nan) for score in data]
    metric2_scores = [score.get(metric2, np.nan) for score in data]
    # 移除 NaN 值
    valid_indices = ~np.isnan(metric1_scores) & ~np.isnan(metric2_scores)
    valid_metric1_scores = np.array(metric1_scores)[valid_indices]
    valid_metric2_scores = np.array(metric2_scores)[valid_indices]
    
    # 计算 Spearman's 和 Kendall Tau 相关性
    print("Spearman's Correlation: ", np.corrcoef(valid_metric1_scores, valid_metric2_scores)[0, 1])
    print('Kendall Tau Score: ', kendalltau(valid_metric1_scores, valid_metric2_scores))
    # print(num1)
    # print(num22)

# 示例使用
metrics = ['clipScore', 'siglipScore','HPSv2', 'imagereward', 'PickScore', 'Science-T2I', 'avg_gpt4o']

for metric in metrics:
    print("evaluation_metric: ", metric)
    compute_correlation('average_score', metric)
    print("-"*50)

