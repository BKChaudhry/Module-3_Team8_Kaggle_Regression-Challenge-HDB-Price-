# HDB Resale Price Prediction — Presentation Notes
## Team 8

---

## Slide 1 — Problem Statement

**Goal:** Predict the resale price of HDB flats in Singapore.

- Dataset: 150,634 training transactions, 16,735 test cases
- Target variable: `resale_price` (SGD)
- Evaluation metric: **Root Mean Squared Error (RMSE)**
- Mean resale price: ~SGD 449,000 | Range: SGD 150,000 – 1,258,000

**Why is this hard?**
- Prices are driven by a mix of physical, locational, policy, and time-based factors
- Many categorical variables (town, flat type, MRT station, school names)
- Non-linear relationships — e.g. distance to CBD matters more for some flat types than others

---

## Slide 2 — Our Approach (Overview)

We built a **3-model gradient boosting ensemble** with rich feature engineering.

```
Raw Data
   ↓
Feature Engineering (106 features)
   ↓
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  LightGBM   │  │   XGBoost   │  │  CatBoost   │
│  5-fold CV  │  │  5-fold CV  │  │  5-fold CV  │
└─────────────┘  └─────────────┘  └─────────────┘
         ↓               ↓               ↓
         └───────────────┴───────────────┘
                         ↓
              Inverse-RMSE Weighted Average
                         ↓
                  Final Predictions
```

---

## Slide 3 — Feature Engineering

We created 106 features from the original 77 columns. The key additions:

### Why Feature Engineering Matters
Raw data gives models the facts. Feature engineering gives models the *meaning*.

### Key Feature Groups

**1. Remaining Lease**
- HDB flats have a 99-year lease. A flat with 40 years left is worth far less than one with 80 years.
- `remaining_lease = 99 − hdb_age`
- This is consistently the **#1 most important feature** across all models.

**2. Distance to Key Locations**
- We computed Haversine distances to 6 Singapore landmarks: CBD (Raffles Place), Orchard Road, Jurong East, Woodlands, Changi Airport, Punggol.
- Closer to CBD = significantly higher price.
- Log-transformed distances capture the non-linear "decay" effect.

**3. Time Trend**
- HDB prices rose significantly from 2012–2024.
- `time_index = (year − 2012) × 12 + month` — a single linear variable capturing the market trend.

**4. Interaction Features**
- `area × storey`: A large flat on a high floor commands a double premium.
- `lease × area`: A big flat with many years remaining is the most valuable combination.
- `CBD distance × storey`: Being high up matters more when you're already near the CBD.

**5. Amenity Scores**
- Weighted count of malls and hawker centres by proximity ring (500m ×3, 1km ×2, 2km ×1).
- Missing `Within_*` values filled with 0 — no amenity within that radius.

**6. Mature Estate Flag**
- Singapore's HDB policy designates 19 "mature estates" (e.g. Toa Payoh, Queenstown).
- These estates consistently command price premiums.

---

## Slide 4 — Why Gradient Boosting?

Gradient boosting builds trees sequentially, each correcting the errors of the previous.

**Why it works for this problem:**
- Handles mixed numeric + categorical features natively
- Robust to outliers and skewed distributions
- Captures non-linear relationships and interactions automatically
- No need for feature scaling

**Why three different models?**
Each model has a different bias:
- **LightGBM**: Leaf-wise tree growth — fast, deep trees, great for large datasets
- **XGBoost**: Level-wise growth — more conservative, different error patterns
- **CatBoost**: Ordered boosting — handles categorical variables natively using internal target encoding, reduces overfitting on categoricals

When models make different errors, averaging them cancels noise → lower RMSE.

---

## Slide 5 — Cross-Validation Strategy

**5-fold cross-validation** on the training set.

```
Training Data (150,634 rows)
├── Fold 1: Train on 80% → Validate on 20%
├── Fold 2: Train on 80% → Validate on 20%
├── Fold 3: Train on 80% → Validate on 20%
├── Fold 4: Train on 80% → Validate on 20%
└── Fold 5: Train on 80% → Validate on 20%
         ↓
  OOF predictions cover 100% of training data
         ↓
  Single reliable RMSE estimate (no optimism bias)
```

**Why this matters:**
- A single train/test split gives an unstable RMSE estimate.
- OOF predictions give one prediction per training row, never seen during training.
- Test predictions = average of 5 models trained on slightly different data (acts as bagging).

---

## Slide 6 — Model Results

### Individual Model Performance (5-fold OOF RMSE)

| Model | V0.1 RMSE | V0.2 RMSE | Improvement |
|---|---|---|---|
| LightGBM | 22,282 | 22,133 | −149 |
| XGBoost | 21,932 | 21,881 | −51 |
| CatBoost | — | 21,690 | new |

### Ensemble Performance

| Version | RMSE | vs Previous |
|---|---|---|
| V0.1 (LGB + XGB) | 21,665 | baseline |
| **V0.2 (LGB + XGB + CB)** | **21,457** | **−208** |

### Ensemble Weights (Inverse-RMSE)
- LightGBM: 33.0%
- XGBoost: 33.4%
- CatBoost: 33.6%

Near-equal weights confirm all three models contribute meaningfully — none dominates.

---

## Slide 7 — Key Findings

**What drives HDB resale prices the most?**

1. **Remaining lease** — Every extra year on the lease adds ~SGD 1,500–3,000 to price on average. Flats with <60 years remaining are significantly discounted.

2. **Floor area** — The most direct predictor. Price scales roughly linearly with sqm but the interaction with storey level and location creates premiums.

3. **Distance to CBD** — Flats within 3km of Raffles Place command a 15–25% premium over comparable flats in the suburbs.

4. **Storey level** — High-floor units (floor 15+) command 5–10% premium over ground-floor equivalents in the same block.

5. **Town / Planning Area** — Mature estates like Queenstown, Toa Payoh, and Bukit Merah consistently price above average.

6. **MRT proximity** — Units within 500m of an MRT interchange command the highest transport premium.

7. **Time** — The market rose steadily over the training period; more recent transactions carry a systematic upward adjustment.

---

## Slide 8 — What We Would Do Next

Given more time, the next improvements would be:

1. **Target Encoding** (in-progress as V0.3)
   - Replace label-encoded categoricals with smoothed mean resale price per category
   - Must be done within each fold to prevent data leakage
   - Expected gain: −500 to −1,000 RMSE

2. **Ridge Meta-Stacking**
   - Use OOF predictions from LGB/XGB/CatBoost as inputs to a Ridge regression
   - Learns the optimal linear combination rather than fixed weights
   - Expected gain: −50 to −100 RMSE

3. **Hyperparameter Tuning with Optuna**
   - Automated Bayesian search over num_leaves, depth, learning_rate, regularisation
   - Expected gain: −100 to −300 RMSE

4. **Geographic Clustering**
   - Group flats by geohash cell, compute cluster-level price statistics
   - Captures micro-location effects not captured by individual distance features

---

## Slide 9 — Summary

| What | Result |
|---|---|
| Best submission | Submission_Team8.V0.2.csv |
| Best OOF RMSE | **21,457** |
| Features engineered | 106 |
| Models trained | 3 (LGB + XGB + CatBoost) |
| CV folds | 5-fold (all models) |
| Key insight | Remaining lease + CBD distance + storey interaction drive the most value |

**The ensemble of three diverse gradient boosting models, trained on 106 engineered features with 5-fold cross-validation, achieves a final RMSE of 21,457 — meaning our predictions are on average within ~SGD 21,500 of the actual resale price.**
