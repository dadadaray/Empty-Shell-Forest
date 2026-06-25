# 数据目录 — 论文必需文件

共 **39 个文件**，~528MB。所有脚本通过 `config.py` 自动解析到本目录。

## Yan 流水线产出（核心）

以下 4 个文件由 `scripts/02_dtw_clustering/07_yan_pipeline.py` 生成，所有结果与论文一致：

| 文件 | 内容 | 关键数字 |
|------|------|----------|
| `Yan_DTW_Labels_601.csv` | 601 站点 K=3 DTW 标签 + ES flag | ES=183 (30.4%) |
| `Yan_DTW_Tensor_601.csv` | 601 站点 x 4 指标 x 8 年轨迹张量 | — |
| `Yan_XGBoost_Predictions_1393.csv` | 1,393 站点 ES 概率预测 | Prob>0.5: 511 (36.7%) |
| `XGBoost_Global_20F.json` | 二分类 XGBoost 模型 (max_depth=2, 150 trees) | AUC=0.909, HR=1.60 |

---

## 流水线输入数据 (12 files)

| 文件 | 行数 | 大小 | 用途 |
|------|:---:|------|------|
| `GTM_Full_1613.csv` | 1,613 | 48KB | Hammond 2022 + Hartmann/Forzieri 2022 原始站点 |
| `GTM_1613_Ultimate_QA_Check.csv` | 1,613 | 27KB | ESA CCI land cover + WRI driver class 质检 |
| `GTM_Golden_1401.csv` | 1,401 | 27KB | ESA+WRI 过滤后的黄金站点 |
| `GTM_Golden_1401_NDVI_NDII_Aligned.csv` | 51,838 | 3MB | Landsat NDVI/NDII 时序 (对齐后) |
| `GTM_Golden_1401_GOSIF_2000_2024.csv` | 27,674 | 574KB | GOSIF SIF 年时序 |
| `GTM_Golden_1401_VODCA_CKXU_1987_2020.csv` | 47,635 | 1.5MB | VODCA C-band 年时序 |
| `GTM_Golden_1401_VODCA_LBAND_2010_2020.csv` | 15,412 | 581KB | VODCA L-band 年时序 |
| `GTM_Golden_1401_AGB_Biomass_sat_io.csv` | 14,011 | 309KB | ESA CCI 地上生物量 (AGB) |
| `GTM_Golden_1401_CanopyHeight_Official.csv` | 1,402 | 34KB | 冠层高度 |
| `GTM_Golden_1401_SPEI_12_24_36_1984_2022.csv` | 54,640 | 10MB | SPEI-12/24/36 干旱指数 |
| `GTM_Golden_1401_Landsat_NDVI&NDII_1984-2002.csv` | — | — | Landsat 前期 |
| `GTM_Golden_1401_Landsat_NDVI&NDII_2003-2020.csv` | — | — | Landsat 后期 |

## 主数据 (3 files)

| 文件 | 行数 | 大小 | 用途 |
|------|:---:|------|------|
| `GTM_Master_Panel_Data_Final.csv` | 54,640 | 4.3MB | 全量面板 (NDVI/NDII/SIF/VOD/AGB/Height + event_start) |
| `GTM_Master_Panel_Data_Spatial.csv` | 54,640 | 5.5MB | 含经纬度面板 |
| `GTM_Master_Dynamic_Cleaned.csv` | 108,671 | 10MB | 含生态命运分类 (Forest/Early Succession/State Shift) |

## 聚类数据 (5 files)

| 文件 | 用途 |
|------|------|
| `Core_Clustering_Tensor_Input.csv` | 601 站点 × 4 指标 × 8 年张量 |
| `Final_Perfect_Tensor_for_DTW.csv` | DTW 最终输入张量 |
| `GTM_Clustering_Results_618.csv` | K=3 DTW 聚类标签 + 指标 |
| `GEE_Input_Final_Pure.csv` | DTW 标签 (Cluster_3: 0=SlowBurn, 1=Synchronous, 2=EmptyShell) |
| `GEE_Input_Final_Pure_149_with_TimeSeries.csv` | 149 训练站点 8 年完整轨迹 |

## XGBoost 特征 (5 files)

| 文件 | 行数 | 用途 |
|------|:---:|------|
| `GTM_1393_Landsat_20D_Advanced.csv` | 1,393 | 全站点 20 特征矩阵 |
| `GTM_149_Landsat_20D_Advanced.csv` | 149 | 训练子集 20 特征 |
| `GTM_149_Full_20_Mechanisms_XGBoost.csv` | 149 | 完整 20 特征 + 机制标注 |
| `GTM_1401_Sites_Probabilities_3class_TEMP.csv` | 1,401 | 三分类预测概率 |
| `GTM_1401_Final_Strict_Metrics.csv` | 1,401 | 严格筛选版指标 |

## 死亡率数据 (4 files)

| 文件 | 用途 |
|------|------|
| `Mortality_16_25.json` | Hansen GFC 2003–2025 死亡事件 (JSON) |
| `Mortality_24_25.json` | 2024–2025 更新 |
| `Mortality_16_25_Golden.csv` | 1,401 站点死亡时间 (CSV) |
| `Mortality_24_25_Golden.csv` | 更新版 |

## 模型 (4 files, also in ../models/)

| 文件 | 大小 | 用途 |
|------|------|------|
| `XGBoost_Global_20F.json` | 113KB | Fig 3 全球推演模型 |
| `XGBoost_Model_3class_A1.json` | 111KB | 综合症 1 (Slow Burn) |
| `XGBoost_Model_3class_A2.json` | 113KB | 综合症 2 (Synchronous) |
| `XGBoost_Model_3class_A3.json` | 113KB | 综合症 3 (Empty Shell) |

## 空间数据 (1 file)

| 文件 | 大小 | 用途 |
|------|------|------|
| `ES_Global_20D_AGB_5km_Merged.tif` | 486MB | 全球 20 特征 + AGB 5km 栅格 (Fig 3 输入) |

---

## 数据来源

| 数据 | 来源 | 引用 |
|------|------|------|
| Tree mortality sites | Hammond et al. 2022 + Hartmann/Forzieri | *Nat. Commun.* 13, 1761; *Nature* 608, 534 |
| Landsat NDVI/NDII | Google Earth Engine, Collection 2 Tier 1 | Gorelick et al. 2017 |
| GOSIF SIF | globalecology.unh.edu/data/GOSIF.html | Li & Xiao 2019 |
| VODCA C-band | zenodo.org/record/2575599 | Moesinger et al. 2020 |
| ESA CCI AGB v6.0 | climate.esa.int/en/projects/biomass | Santoro et al. 2021 |
| TerraClimate | climatologylab.org/terraclimate.html | Abatzoglou et al. 2018 |
| SoilGrids 250m | soilgrids.org | Hengl et al. 2017 |
| SRTM 90m | www2.jpl.nasa.gov/srtm | Farr et al. 2007 |
| RESOLVE Ecoregions 2017 | ecoregions.appspot.com | Dinerstein et al. 2017 |
| Human Footprint | sedac.ciesin.columbia.edu | Kennedy et al. 2019 |
| Hansen GFC | earthenginepartners.appspot.com | Hansen et al. 2013 |
| SPEI | spei.csic.es | Vicente-Serrano et al. 2010 |

## 站点流水线

```
1,613 → ESA(排除40,50) → 1,545 → WRI(排除1,2,6) → 1,401 → 去水(80) → 1,393
  └─ 森林要求(10,95): -323 ─┐
  └─ 8年内无二次干扰 ────────┤
  └─ ≥5有效年观测/指标 ──────┘ → 601 DTW 训练站点
                                       ├─ Empty Shell:  183 (30.4%)
                                       ├─ Synchronous:  232 (38.6%)
                                       └─ Slow Burn:    188 (31.3%)

601 DTW labeled → XGBoost trained → predicted on all 1,393 → Prob_ES > 0.5 = 511 (36.7%)
```
