# 论文修改清单

代码 SHAP 排名已从 Recovery_Year 数据准备改为 Yan NDVI-minimum 对齐（601 站点, 183 ES），以下段落需同步修改。

---

## 2.2 Results

### 修改 1 — SHAP 特征排名
**旧**: Maximum Temperature, Biome Type, Precipitation, Sand Content, Maximum VPD
**新**: Maximum Temperature, Biome Type, Elevation, Burn Severity (dNBR), Human Footprint, Sand Content

### 修改 2 — SHAP 依赖图描述
**旧**: 四张依赖图 (tmmx → pr → sand → maxVPD)
**新**: 六张依赖图对应新 top-6 物理特征: tmmx, elevation, dNBR, Human_Footprint, Sand_Content, TWI

### 修改 3 — 嵌套 Cox 特征列表
**旧**: maximum temperature, precipitation, maximum VPD, burn severity, elevation
**新**: maximum temperature, elevation, dNBR, human footprint, sand content, TWI

### 修改 4 — Cox HR 值
运行 `pipeline/Cox_PH.py` 获取新的 6 个嵌套模型 HR/CI/p 值，替换文中数值

### 修改 5 — Schoenfeld 残差
**旧**: all P > 0.05
**新**: 如实报告——部分模型 P < 0.05

---

## 5.3 Methods

### 修改 6 — 消融实验表述
**旧**: Maximum Temperature, Precipitation, and Sand Content
**新**: Maximum Temperature, Elevation, and Sand Content

### 修改 7 — Discussion SHAP 值
**旧**: |SHAP| = 0.768, 0.471, 0.424
**新**: |SHAP| = 0.703, 0.284, 0.201

---

## 5.4 Methods

### 修改 8 — 嵌套 Cox 特征列表
同上修改 3

### 修改 9 — Fig 2 caption, Fig S2 caption, 摘要
同步更新特征名称和 SHAP 值
