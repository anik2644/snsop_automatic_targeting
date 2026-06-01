# Model files

Put your exported bundle here:

**`support_type_model_assets.joblib`**

It should be the dictionary you save from Colab / training:

- `model` — trained estimator (e.g. `xgboost.XGBClassifier`)
- `feature_names` — same list as `FEATURE_COLS` used at training time (column order matters)
- `label_encoder` — fitted `sklearn.preprocessing.LabelEncoder`

The API loads this file automatically if it exists (see `resolve_assets_path()` in `main.py`). You can also set **`ASSETS_PATH`** to any full path.

Large `.joblib` files are gitignored by default so they are not committed by mistake; use your own storage or CI artifacts for deployment.
