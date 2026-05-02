# HDB Resale Price Prediction — Team 8

Predict Singapore HDB resale flat prices using a 3-model gradient boosting ensemble.
**Evaluation metric:** Root Mean Squared Error (RMSE)

---

## Final Results

| Version | Models | OOF RMSE (SGD) | Submission File |
|---|---|---|---|
| V0.1 | LightGBM + XGBoost | 21,665 | Submission_Team8.V0.1.csv |
| **V0.2** | LightGBM + XGBoost + CatBoost | **21,457** | **Submission_Team8.V0.2.csv** |

### Per-Model Breakdown (V0.2)

| Model | OOF RMSE | Ensemble Weight |
|---|---|---|
| LightGBM | 22,133 | 33.0% |
| XGBoost | 21,881 | 33.4% |
| CatBoost | 21,690 | 33.6% |
| **Ensemble** | **21,457** | — |

---

## Project Structure

```
.
├── train.csv                     # Training data (150,634 rows, 77 columns)
├── test.csv                      # Test data (16,737 rows, 76 columns)
├── sample_sub_reg.csv            # Example submission format
├── predict.py                    # V0.1 pipeline: LightGBM + XGBoost
├── predict_v2.py                 # V0.2 pipeline: LightGBM + XGBoost + CatBoost
├── predict_v3.py                 # V0.3 pipeline: + Target Encoding + Ridge Stacking
├── Submission_Team8.V0.1.csv     # V0.1 predictions (16,735 rows)
├── Submission_Team8.V0.2.csv     # V0.2 predictions (16,735 rows) ← best
├── .env                          # Hyperparameter configuration
└── README.md
```

---

## Setup

```bash
pip install lightgbm xgboost catboost scikit-learn pandas numpy
```

## Usage

```bash
# Best submission (V0.2)
python predict_v2.py
```

Outputs `Submission_Team8.V0.2.csv` with columns `Id` and `Predicted`.

---

## Pipeline — V0.2

### 1. Feature Engineering (106 features total)

**Lease & Time**
| Feature | Description |
|---|---|
| `remaining_lease` | Years left on 99-year lease = 99 − hdb_age |
| `lease_left_pct` | remaining_lease / 99 |
| `time_index` | Linear month index from Jan 2012 — captures market price trend |

**Geospatial**
| Feature | Description |
|---|---|
| `dist_cbd` | Haversine distance (m) to Raffles Place |
| `dist_orchard` | Distance to Orchard Road |
| `dist_jurong` | Distance to Jurong East |
| `dist_woodlands` | Distance to Woodlands |
| `dist_changi` | Distance to Changi Airport |
| `dist_punggol` | Distance to Punggol |
| `log_dist_*` | Log-transformed distances (captures non-linear decay) |
| `is_mature_estate` | Flag for 19 mature HDB estates (historically higher prices) |

**Storey**
| Feature | Description |
|---|---|
| `storey_ratio` | mid_storey / max_floor_lvl — relative height in block |
| `is_high_floor` | Flag for floor ≥ 15 |
| `floors_to_top` | Floors above the unit to the top of the building |

**Interactions**
| Feature | Description |
|---|---|
| `area_x_storey` | floor_area × mid_storey (bigger + higher = premium) |
| `lease_x_area` | remaining_lease × floor_area |
| `cbd_x_storey` | CBD distance × floor (proximity compounds with height) |
| `lease_x_mrt` | remaining_lease × MRT distance |
| `area_x_cbd` | floor_area × CBD distance |
| `mrt_x_cbd` | MRT distance × CBD distance |

**Amenities**
| Feature | Description |
|---|---|
| `mall_score` | Weighted mall count: ×3 within 500m, ×2 within 1km, ×1 within 2km |
| `hawker_score` | Same weighting for hawker centres |
| `transport_access` | Sum of inverse MRT + bus distances (higher = better connected) |

**Block Composition**
| Feature | Description |
|---|---|
| `*room_sold_ratio` | Each room type / total units |
| `rental_ratio` | Rental units / total units |
| `dominant_room_type` | Most common room type in the block |

### 2. Categorical Encoding

Label-encoded via joint train+test factorization:

`town`, `flat_type`, `flat_model`, `planning_area`, `mrt_name`, `bus_stop_name`, `pri_sch_name`, `sec_sch_name`, `commercial`, `market_hawker`, `multistorey_carpark`, `precinct_pavilion`

LightGBM receives these as native `category` features. XGBoost uses integer codes. CatBoost handles categoricals internally using ordered target encoding.

### 3. Dropped Columns

| Column | Reason |
|---|---|
| `block`, `street_name` | Redundant — covered by lat/lon + town |
| `address` | 9,157 unique values; location captured by coordinates |
| `storey_range` | Covered by `lower`, `upper`, `mid_storey` |
| `Tranc_YearMonth` | Covered by `Tranc_Year` + `Tranc_Month` |
| `postal` | Location covered by `Latitude`, `Longitude` |
| `full_flat_type` | Redundant with `flat_type` + `flat_model` |
| `residential` | Constant — all rows are residential |

### 4. Models & Hyperparameters

**LightGBM**
```
num_leaves=255, learning_rate=0.05, feature_fraction=0.8
bagging_fraction=0.8, bagging_freq=5, min_child_samples=20
reg_alpha=0.1, reg_lambda=0.1, early_stopping=100
```

**XGBoost**
```
max_depth=6, learning_rate=0.05, subsample=0.8
colsample_bytree=0.8, min_child_weight=10
reg_alpha=0.1, reg_lambda=1.0, tree_method=hist
```

**CatBoost**
```
iterations=5000, learning_rate=0.03, depth=8
l2_leaf_reg=3, early_stopping=100
```

### 5. Ensemble

5-fold cross-validation (KFold, shuffle, seed 42) for all models.
Final prediction = inverse-RMSE weighted average:

```
weight_i = (1 / RMSE_i) / sum(1 / RMSE_j)
```

Weights were nearly equal (~33% each), indicating all three models contribute independently.

---

## Key Insights

- **Remaining lease** is the single strongest predictor — HDB flats with shorter leases sell significantly cheaper.
- **Distance to CBD** has a strong non-linear negative effect on price.
- **Storey level** interacts with area and location — high-floor large flats near CBD command the highest premiums.
- **CatBoost's native categorical handling** outperformed label-encoded LightGBM/XGBoost individually, making it the strongest single model.
- The 3-model ensemble reduced RMSE by ~208 points vs the best individual model.
