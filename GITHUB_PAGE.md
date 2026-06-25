# Empty Shell Syndrome

**Rapid surface greening conceals arrested woody biomass recovery and elevates secondary mortality**

[![Nature Plants](https://img.shields.io/badge/Nature_Plants-2025-228B22)](https://doi.org/10.1038/s41477-025-01948-4)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-Research-green)](LICENSE)

---

## Abstract

Climate-driven disturbances trigger widespread forest mortality, yet post-disturbance monitoring relies on optical greenness (NDVI) that overestimates recovery. Tracking NDVI, NDII, VOD, and SIF across **1,393 global mortality sites**, we identify the *Empty Shell* syndrome — rapid surface greening masking long-term biomass stagnation. DTW clustering (K=3) decouples greenness from biomass, identifying 183 ES sites among 601 training sites. XGBoost + SHAP (AUC=0.909) reveals the syndrome is driven by maximum temperature, elevation, and burn severity acting through arrested succession. Survival analysis shows a **60% increase in secondary mortality risk** (HR=1.60, P=1.61×10⁻¹⁰). Global extrapolation under a standardized severe-disturbance scenario reveals a hidden carbon shortfall of **~2.65 Pg C**.

---

## Key Results

| Analysis | Method | Finding |
|----------|--------|---------|
| **DTW clustering** | K=3, 4 metrics × 8 years | ES = 183 / 601 (30.4%) |
| **XGBoost** | 20 features, max_depth=2 | AUC = 0.909 ± 0.026 |
| **SHAP drivers** | TreeExplainer | Max Temperature (0.70), Elevation (0.28), dNBR (0.21) |
| **Cox survival** | Nested models | HR = 1.60 [1.39, 1.85], P = 1.61×10⁻¹⁰ |
| **Mediation** | Sand Content control | HR drops to 1.10 (ns) |
| **Global projection** | 5km, 25.8M pixels | 22.2% forest at risk, ~2.65 Pg C deficit |

---

## Repository Structure

```
empty-shell-syndrome/
├── pipeline/       # Main experiments (9 scripts, Methods 5.1–5.5)
├── validation/     # Robustness checks (7 scripts)
├── figures/        # Paper figures (19 scripts)
├── gee/            # GEE data extraction (8 scripts)
├── data/           # Immutable input data (15 files)
└── temp/           # Runtime outputs
```

### Pipeline (`pipeline/`)

| Script | Methods | Description |
|--------|:------:|-------------|
| `clean_sites.py` | 5.1 | ESA+WRI+land-cover filter (1613→1393) |
| `assemble_panel.py` | 5.1 | Landsat alignment + 6-layer data fusion |
| **`pipeline.py`** | **5.2–5.4** | **Core**: Full Yan pipeline (DTW→XGBoost→Cox) |
| `XGBoost.py` | 5.3 | Binary XGBoost classifier |
| `SHAP.py` | 5.3 | SHAP TreeExplainer feature ranking |
| `Empty_Shell_probability.py` | 5.3 | Predict ES probability for 1,393 sites |
| `Cox_PH.py` | 5.4 | Nested Cox PH + Schoenfeld residuals + AIC |
| `survival.py` | 5.4 | Kaplan-Meier + log-rank test |
| `global_carbon_sink.py` | 5.5 | Global 5km projection + carbon deficit |

### Validation (`validation/`)

| Script | What it tests |
|--------|------|
| `ablation_single_metrics.py` | Drop one satellite metric at a time |
| `ablation_biome_type.py` | Remove Biome_Type, retrain (ΔAUC) |
| `window_3-10_stability.py` | Observation window sweep (W=3–10) |
| `validate_dtw_2-8__clusters.py` | DTW cluster quality (Silhouette + CH) |
| `spatial_thinning.py` | Grid thinning + criterion-based ES |
| `tertile_mortality_gradient.py` | ES tertile mortality gradient |
| `compute_all_pvalues.py` | All paper p-values in one place |

---

## Quick Start

```bash
git clone https://github.com/yourname/empty-shell-syndrome.git
cd empty-shell-syndrome
pip install -r requirements.txt

cd pipeline
python clean_sites.py
python assemble_panel.py
python pipeline.py          # Full pipeline: DTW→XGBoost→Cox
```

All scripts read from `data/`, write outputs to `temp/`.

---

## Four Satellite Metrics

| Metric | Sensor | Resolution | Measures |
|--------|--------|:---------:|----------|
| **NDVI** | Landsat 5–9 Collection 2 | 30 m | Canopy greenness |
| **NDII** | Landsat 5–9 Collection 2 | 30 m | Canopy water content |
| **VOD** | VODCA C-band | 0.25° | Woody biomass |
| **SIF** | GOSIF (OCO-2/MODIS) | 0.05° | Photosynthetic activity |

---

## Citation

If you use this code or data, please cite:

> Hammond et al. (2022) *Nature Communications*, 13, 1761.
> Yan et al. (2025) *Nature Plants*, 11, 731–742.
> Forzieri et al. (2022) *Nature*, 608, 534–539.

## License

Code is provided for academic research purposes. Data copyright belongs to original data providers.
