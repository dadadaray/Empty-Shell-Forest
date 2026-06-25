"""
项目配置文件
============
DATA_DIR 自动指向仓库内的 data/ 文件夹。
所有脚本运行时自动 chdir 到此目录，无需手动修改路径。
"""

import os

# 自动解析为 <repo>/data/ 绝对路径，无需手动修改
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)

# 所有数据文件已就位 (data/ 目录，共 61 个文件，~614MB)。
# 如需从原始 GEE 目录同步新数据，修改下行路径即可。

# Yan方法参数（与 Yan et al. 2025 Nature Plants 对齐）
YAN_BASELINE_YEARS = 3        # 干扰前非干旱年数量
YAN_SPEI_THRESHOLD = -1       # SPEI-12 干旱阈值
YAN_RECOVERY_WINDOW = 8       # 恢复观测窗口（年）
YAN_DTW_K = 3                 # DTW聚类数

# 全球推演参数（Fig 4）
GLOBAL_RESOLUTION_KM = 5      # 像素分辨率
GLOBAL_CHUNK_ROWS = 200       # 分块处理行数（避免OOM）
ES_RISK_THRESHOLD = 0.5       # 空壳风险阈值

# 统计参数
ALPHA = 0.05                  # 显著性水平
N_BOOTSTRAP = 1000            # Bootstrap 重采样次数
