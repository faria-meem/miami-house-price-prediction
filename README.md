# Miami House Price Prediction

Regression project predicting single-family home **sale prices** in Miami-Dade County from property, location and structure features, using Python (pandas, scikit-learn).

## Dataset
- 13,932 home sales, 15 predictive features, no missing values or duplicate rows.
- Target: `SALE_PRC` (USD). Median price about $310K, mean about $400K, right-skewed (skewness 3.2).
- Features: floor area, land area, special-feature value, structure quality and age, distances to ocean, water, rail, highway, city center and subcenter, airport-noise flag, month sold, latitude/longitude.
- File: `data/miami-housing.csv`. Please cite the original source of the dataset if you publish this project.

## Approach
1. Data inspection and exploratory analysis (distributions, correlations, key relationships, price map)
2. Preprocessing: dropped the `PARCELNO` identifier, 80/20 train-test split (`random_state=42`), scaling inside a pipeline for linear models to avoid leakage
3. Model comparison with 5-fold cross-validation on the training set: Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting
4. Grid search for Random Forest (3-fold CV)
5. One-time evaluation on the held-out test set
6. Permutation feature importance and residual analysis

## Results (test set)

| Model | RMSE (USD) | MAE (USD) | R2 |
|---|---|---|---|
| **Gradient Boosting** | **89,841** | 49,294 | **0.918** |
| Random Forest (tuned) | 98,960 | 48,644 | 0.901 |
| Decision Tree | 128,623 | 67,079 | 0.832 |
| Ridge | 174,799 | 117,312 | 0.690 |
| Linear Regression | 174,805 | 117,328 | 0.690 |

- Gradient Boosting explains about 92% of price variance, with a mean absolute percentage error of 12.3%.
- Tree ensembles beat linear models by a wide margin, which points to non-linear effects and interactions.
- Most important features (permutation importance, drop in R2): floor area (0.31), distance to ocean (0.14), longitude (0.11), distance to city center (0.10), distance to subcenter (0.07).
- Tuning Random Forest gave little benefit, and Gradient Boosting was left with reasonable fixed settings and not tuned.

## Limitations and next steps
- The random split can place nearby houses, and a few repeated parcels, in both train and test, so scores may be optimistic. A spatial or parcel-grouped split would be stricter.
- Prices are right-skewed and errors grow for expensive homes (see the residual plot). Modeling `log(price)` is a natural next step.
- Try XGBoost/LightGBM, tune Gradient Boosting, and add SHAP explanations.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook miami_house_price_prediction.ipynb
```

## Structure
```
data/miami-housing.csv
images/                      # saved plots
miami_house_price_prediction.ipynb
requirements.txt
```
