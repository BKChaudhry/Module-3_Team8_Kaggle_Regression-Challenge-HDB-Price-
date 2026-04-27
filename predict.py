import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ── Load ─────────────────────────────────────────────────────────────────────
train = pd.read_csv('train.csv', low_memory=False)
test  = pd.read_csv('test.csv',  low_memory=False)

test_ids = test['id'].copy()
y        = train['resale_price'].copy()

# ── Feature engineering ───────────────────────────────────────────────────────
CBD_LAT, CBD_LON = 1.2830, 103.8513   # Raffles Place

def haversine(lat1, lon1, lat2=CBD_LAT, lon2=CBD_LON):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def engineer(df):
    df = df.copy()

    # Remaining lease (HDB lease is 99 years)
    df['remaining_lease'] = 99 - df['hdb_age']

    # Transaction year already in Tranc_Year; add a linear time index
    df['time_index'] = (df['Tranc_Year'] - 2012) * 12 + df['Tranc_Month']

    # Distance to CBD
    df['dist_to_cbd'] = haversine(df['Latitude'], df['Longitude'])

    # Mall/Hawker within-radius NaN → 0 (no amenity within that radius)
    for col in ['Mall_Within_500m','Mall_Within_1km','Mall_Within_2km',
                'Hawker_Within_500m','Hawker_Within_1km','Hawker_Within_2km']:
        df[col] = df[col].fillna(0)

    # Nearest distances: fill with a large sentinel (no nearby = far)
    df['Mall_Nearest_Distance']   = df['Mall_Nearest_Distance'].fillna(df['Mall_Nearest_Distance'].median())
    df['Hawker_Nearest_Distance'] = df['Hawker_Nearest_Distance'].fillna(df['Hawker_Nearest_Distance'].median())

    # Floor density: how high relative to building height
    df['storey_ratio'] = df['mid_storey'] / df['max_floor_lvl'].replace(0, 1)

    # Proportion of each room type in the block
    total = df['total_dwelling_units'].replace(0, 1)
    for room in ['1room_sold','2room_sold','3room_sold','4room_sold',
                 '5room_sold','exec_sold','multigen_sold','studio_apartment_sold']:
        df[f'{room}_ratio'] = df[room] / total

    # Rental proportion
    rental_cols = ['1room_rental','2room_rental','3room_rental','other_room_rental']
    df['total_rental'] = df[rental_cols].sum(axis=1)
    df['rental_ratio'] = df['total_rental'] / total

    return df

train = engineer(train)
test  = engineer(test)

# ── Drop low-value / leaky columns ───────────────────────────────────────────
DROP = [
    'id', 'resale_price',
    'Tranc_YearMonth',          # covered by Tranc_Year + Tranc_Month
    'block', 'street_name',     # address covers both; too high cardinality
    'address',                  # 9157 unique; lat/lon + town capture location
    'storey_range',             # covered by lower/upper/mid/mid_storey
    'postal',                   # lat/lon captures location
    'full_flat_type',           # redundant (flat_type + flat_model)
    'residential',              # constant column
]
DROP_TRAIN = [c for c in DROP if c in train.columns]
DROP_TEST  = [c for c in DROP if c in test.columns and c != 'resale_price']

X      = train.drop(columns=DROP_TRAIN)
X_test = test.drop(columns=[c for c in DROP_TEST])

# ── Encode categoricals ───────────────────────────────────────────────────────
CAT_COLS = [
    'town', 'flat_type', 'flat_model', 'planning_area',
    'mrt_name', 'bus_stop_name', 'pri_sch_name', 'sec_sch_name',
    'commercial', 'market_hawker', 'multistorey_carpark', 'precinct_pavilion',
]

for col in CAT_COLS:
    if col in X.columns:
        combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
        codes, _ = pd.factorize(combined)
        X[col]      = pd.Categorical(codes[:len(X)])
        X_test[col] = pd.Categorical(codes[len(X):])

# ── Cross-validated LightGBM ─────────────────────────────────────────────────
LGB_PARAMS = dict(
    objective        = 'regression',
    metric           = 'rmse',
    num_leaves       = 255,
    learning_rate    = 0.05,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq     = 5,
    min_child_samples= 20,
    reg_alpha        = 0.1,
    reg_lambda       = 0.1,
    verbose          = -1,
    n_jobs           = -1,
)

N_FOLDS   = 5
kf        = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof_lgb   = np.zeros(len(X))
pred_lgb  = np.zeros(len(X_test))

print("Training LightGBM …")
for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CAT_COLS)
    dval   = lgb.Dataset(X_va, label=y_va, reference=dtrain, categorical_feature=CAT_COLS)

    model = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round      = 3000,
        valid_sets           = [dval],
        callbacks            = [lgb.early_stopping(100, verbose=False),
                                 lgb.log_evaluation(500)],
    )
    oof_lgb[va_idx]  = model.predict(X_va)
    pred_lgb        += model.predict(X_test) / N_FOLDS

    fold_rmse = np.sqrt(mean_squared_error(y_va, oof_lgb[va_idx]))
    print(f"  Fold {fold} RMSE: {fold_rmse:,.0f}")

lgb_cv = np.sqrt(mean_squared_error(y, oof_lgb))
print(f"LightGBM OOF RMSE: {lgb_cv:,.0f}\n")

# ── Cross-validated XGBoost ──────────────────────────────────────────────────
# XGBoost needs numeric-only; convert category → int code
X_xgb      = X.copy()
X_test_xgb = X_test.copy()
for col in CAT_COLS:
    if col in X_xgb.columns:
        X_xgb[col]      = X_xgb[col].cat.codes
        X_test_xgb[col] = X_test_xgb[col].cat.codes

XGB_PARAMS = dict(
    objective         = 'reg:squarederror',
    eval_metric       = 'rmse',
    max_depth         = 6,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_weight  = 10,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    tree_method       = 'hist',
    n_jobs            = -1,
    random_state      = 42,
)

oof_xgb  = np.zeros(len(X_xgb))
pred_xgb = np.zeros(len(X_test_xgb))

print("Training XGBoost …")
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_xgb), 1):
    X_tr, X_va = X_xgb.iloc[tr_idx], X_xgb.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx],      y.iloc[va_idx]

    model = xgb.XGBRegressor(**XGB_PARAMS, n_estimators=3000, early_stopping_rounds=100)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=500)

    oof_xgb[va_idx]  = model.predict(X_va)
    pred_xgb        += model.predict(X_test_xgb) / N_FOLDS

    fold_rmse = np.sqrt(mean_squared_error(y_va, oof_xgb[va_idx]))
    print(f"  Fold {fold} RMSE: {fold_rmse:,.0f}")

xgb_cv = np.sqrt(mean_squared_error(y, oof_xgb))
print(f"XGBoost OOF RMSE: {xgb_cv:,.0f}\n")

# ── Weighted ensemble (weight by inverse OOF RMSE) ───────────────────────────
w_lgb = 1 / lgb_cv
w_xgb = 1 / xgb_cv
w_sum = w_lgb + w_xgb
w_lgb /= w_sum
w_xgb /= w_sum

oof_ens  = w_lgb * oof_lgb  + w_xgb * oof_xgb
pred_ens = w_lgb * pred_lgb + w_xgb * pred_xgb

ens_cv = np.sqrt(mean_squared_error(y, oof_ens))
print(f"Ensemble OOF RMSE: {ens_cv:,.0f}")
print(f"LGB weight: {w_lgb:.3f}  |  XGB weight: {w_xgb:.3f}")

# ── Write submission ──────────────────────────────────────────────────────────
sub = pd.DataFrame({'Id': test_ids, 'Predicted': pred_ens})
sub.to_csv('submission.csv', index=False)
print("\nSubmission saved -> submission.csv")
print(sub.head())
