# HDB Resale Price Prediction

Predict Singapore HDB resale flat prices using gradient boosting ensemble.  
**Evaluation metric:** Root Mean Squared Error (RMSE)

## Results

| Model | OOF RMSE (SGD) |
|---|---|
| LightGBM (5-fold CV) | 22,282 |
| XGBoost (5-fold CV) | 21,932 |
| **Ensemble** | **21,665** |

## Project Structure

```
.
├── train.csv              # Training data (150,634 rows, 77 columns)
├── test.csv               # Test data (16,737 rows, 76 columns)
├── sample_sub_reg.csv     # Example submission format
├── predict.py             # Main pipeline: feature engineering + training + submission
├── submission.csv         # Generated predictions (output)
├── .env                   # Hyperparameter and path configuration
└── README.md
```

## Setup

```bash
pip install lightgbm xgboost scikit-learn pandas numpy
```

## Usage

```bash
python predict.py
```

Outputs `submission.csv` with columns `Id` and `Predicted`.

## Pipeline Overview

### Feature Engineering

| Feature | Description |
|---|---|
| `remaining_lease` | Years left on 99-year lease = 99 − hdb_age |
| `dist_to_cbd` | Haversine distance (m) to Raffles Place |
| `time_index` | Linear month index from Jan 2012 (captures price trend) |
| `storey_ratio` | mid_storey / max_floor_lvl — relative height in block |
| `*_ratio` | Room-type counts divided by total dwelling units |
| `total_rental` | Sum of all rental unit types |
| `rental_ratio` | Rental units / total dwelling units |
| Mall/Hawker `Within_*` NaN | Filled with 0 — no amenity within that radius |

### Categorical Encoding

Label-encoded via joint train+test factorization, then passed as LightGBM `category` features:
`town`, `flat_type`, `flat_model`, `planning_area`, `mrt_name`, `bus_stop_name`, `pri_sch_name`, `sec_sch_name`

### Dropped Columns

| Column | Reason |
|---|---|
| `block`, `street_name` | Redundant — captured by `address` and lat/lon |
| `address` | 9,157 unique values; location covered by lat/lon + town |
| `storey_range` | Covered by `lower`, `upper`, `mid_storey` |
| `Tranc_YearMonth` | Covered by `Tranc_Year` + `Tranc_Month` |
| `postal` | Location covered by `Latitude`, `Longitude` |
| `full_flat_type` | Redundant with `flat_type` + `flat_model` |
| `residential` | Constant — all rows are residential |

### Model

- **5-fold cross-validation** (KFold, shuffle, seed 42)
- **LightGBM** — `num_leaves=255`, `learning_rate=0.05`, early stopping at 100 rounds
- **XGBoost** — `max_depth=6`, `learning_rate=0.05`, `tree_method=hist`
- **Ensemble** — inverse-RMSE weighted average of OOF predictions (~50/50 split)

## Data Dictionary (Key Features)

| Column | Description |
|---|---|
| `resale_price` | Target — sale price in SGD |
| `town` | HDB township (26 unique) |
| `flat_type` | Room type (1–5 ROOM, EXECUTIVE, MULTI-GENERATION) |
| `floor_area_sqm` | Floor area in square metres |
| `lease_commence_date` | Year the 99-year lease began |
| `hdb_age` | Years from lease start to transaction year |
| `mid_storey` | Median floor of the storey range |
| `mrt_nearest_distance` | Distance (m) to nearest MRT |
| `Mall_Nearest_Distance` | Distance (m) to nearest mall |
| `pri_sch_nearest_distance` | Distance (m) to nearest primary school |
| `Latitude` / `Longitude` | GPS coordinates of the flat block |
