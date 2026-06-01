# Support type API (FastAPI)

## Where to put `support_type_model_assets.joblib`

Your Colab export:

```python
model_assets = {
    "model": model_support_type,
    "feature_names": FEATURE_COLS,
    "label_encoder": le,
}
joblib.dump(model_assets, "support_type_model_assets.joblib")
```

**Recommended (repo default):** copy the file to:

**`backend/models/support_type_model_assets.joblib`**

See also [`models/README.md`](models/README.md).

**Alternative:** same filename next to the app: **`backend/support_type_model_assets.joblib`**

**Any path:** set environment variable **`ASSETS_PATH`** to the full path of your `.joblib` file (this wins over the default search).

Search order when `ASSETS_PATH` is **not** set:

1. `backend/models/support_type_model_assets.joblib`
2. `backend/support_type_model_assets.joblib`

If neither exists, a **dummy** `RandomForestClassifier` is trained for local development.

Large `.joblib` files are **gitignored** by default (see repo `.gitignore`); keep them locally or in artifact storage / your deployment pipeline.

## Contract

**POST** `/predict_support_type`

Request body:

```json
{
  "household": {
    "male_0_2_normal": 0,
    "male_0_2_disabled": 0,
    "male_0_2_chronically_ill": 0,
    "male_0_2_both": 0,
    "...": 0,
    "female_65_plus_both": 0
  }
}
```

Missing keys are treated as `0`. Column names and order must match **`feature_names`** saved inside your joblib (your `FEATURE_COLS`).

**Preprocessing:** `pd.DataFrame([household])[feature_names]` then **`clip(0, SUBGROUP_CLIP_MAX)`** (default cap **10**).

Response:

```json
{
  "predicted_support_type": "DIS",
  "top_features": [
    {"feature": "female_18_35_normal", "shap_value": 2.12},
    {"feature": "female_36_64_normal", "shap_value": 1.68}
  ]
}
```

**GET** `/health` — includes `assets_source`: path to the loaded joblib, or `"dummy"`.

**POST** `/get_support_type`

Request body:

```json
{
  "household": {
    "male_0_2_normal": 0,
    "male_0_2_disabled": 0,
    "male_0_2_chronically_ill": 0,
    "male_0_2_both": 0,
    "male_3_5_normal": 0,
    "male_3_5_disabled": 0,
    "male_3_5_chronically_ill": 0,
    "male_3_5_both": 0,
    "male_6_17_normal": 1,
    "male_6_17_disabled": 0,
    "male_6_17_chronically_ill": 0,
    "male_6_17_both": 0,
    "male_18_35_normal": 1,
    "male_18_35_disabled": 0,
    "male_18_35_chronically_ill": 0,
    "male_18_35_both": 0,
    "male_36_64_normal": 0,
    "male_36_64_disabled": 0,
    "male_36_64_chronically_ill": 0,
    "male_36_64_both": 0,
    "male_65_plus_normal": 0,
    "male_65_plus_disabled": 0,
    "male_65_plus_chronically_ill": 0,
    "male_65_plus_both": 0,
    "female_0_2_normal": 0,
    "female_0_2_disabled": 0,
    "female_0_2_chronically_ill": 0,
    "female_0_2_both": 0,
    "female_3_5_normal": 1,
    "female_3_5_disabled": 0,
    "female_3_5_chronically_ill": 0,
    "female_3_5_both": 0,
    "female_6_17_normal": 0,
    "female_6_17_disabled": 0,
    "female_6_17_chronically_ill": 0,
    "female_6_17_both": 0,
    "female_18_35_normal": 1,
    "female_18_35_disabled": 0,
    "female_18_35_chronically_ill": 0,
    "female_18_35_both": 0,
    "female_36_64_normal": 0,
    "female_36_64_disabled": 0,
    "female_36_64_chronically_ill": 0,
    "female_36_64_both": 0,
    "female_65_plus_normal": 0,
    "female_65_plus_disabled": 0,
    "female_65_plus_chronically_ill": 0,
    "female_65_plus_both": 0
  }
}
```

Response:

```json
{
  "support_type": "DIS"
}
```

## Run with Docker

From the `backend` directory:

```bash
docker compose up --build
```

The API will be available at:

```text
http://localhost:8040/health
```

The Compose service uses `restart: unless-stopped`, so Docker will keep the container alive and restart it if it exits. Stop it from another terminal with:

```bash
docker compose down
```

## Run locally

### macOS/Linux

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

Open:

```text
http://localhost:5000/health
```

### Windows PowerShell

```powershell
cd backend
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

Open:

```text
http://localhost:5000/health
```

Use **Python 3.11 or 3.12** for the venv when you depend on XGBoost (see troubleshooting below). The `backend/.python-version` file is a hint for pyenv-style tools only; Windows does not install Python from that file.

### Windows: “No suitable Python runtime found” (py launcher)

That message means **`py -3.12`** (or similar) has **no matching install registered**. Fix by installing Python 3.12, then recreate the venv.

1. **See what the launcher knows** (optional):

   ```powershell
   py --list
   ```

2. **Install Python 3.12** (pick one):

   - **Installer:** [python.org — Windows downloads](https://www.python.org/downloads/windows/) → **Python 3.12.x** → run installer and enable **“Add python.exe to PATH”** (or note the install path, e.g. `C:\Users\You\AppData\Local\Programs\Python\Python312\`).
   - **winget:** `winget install Python.Python.3.12`

3. **New terminal**, then create the venv with the **3.12** you just installed:

   ```powershell
   cd E:\Automated Targeting\automated-targeting\backend
   py -3.12 -m venv .venv
   ```

   If `py -3.12` still fails, use the **full path** to `python.exe` from the installer (example — yours may differ):

   ```powershell
   & "C:\Users\YOURNAME\AppData\Local\Programs\Python\Python312\python.exe" -m venv .venv
   ```

4. **Activate and install deps:**

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
   .\.venv\Scripts\Activate.ps1
   python -V
   pip install -r requirements.txt
   ```

You can activate from the repo root with:

`& "E:\Automated Targeting\automated-targeting\backend\.venv\Scripts\Activate.ps1"`

After activation, `python` and `pip` refer to that venv.

### `ModuleNotFoundError: No module named 'xgboost'`

That happens when **`joblib.load`** unpickles an **XGBoost** model but **`xgboost` is not installed** in the same venv you use for `uvicorn`.

```bash
cd backend
.\.venv\Scripts\activate
pip install xgboost
```

If **`pip install xgboost`** fails with “no matching distribution”, your interpreter is probably **too new** (e.g. **Python 3.14**). XGBoost wheels often lag behind the latest Python. **Create the venv with Python 3.11 or 3.12**, then:

```bash
pip install -r requirements.txt
```

## Joblib contents (required)

| Key | Description |
|-----|----------------|
| `model` | Trained tree model (e.g. `xgboost.XGBClassifier`). `shap.TreeExplainer` is created at load if `explainer` is omitted. |
| `feature_names` | `list[str]` — same order as training (`FEATURE_COLS`). |
| `label_encoder` | Fitted `sklearn.preprocessing.LabelEncoder`. |

Optional: `explainer` — if you pre-built SHAP explainer and want to reuse it (otherwise built with `TreeExplainer(model)`).

Legacy: if only `FEATURE_COLS` is present, it is copied to `feature_names`.

## Environment

| Variable | Purpose |
|----------|---------|
| `ASSETS_PATH` | Full path to `support_type_model_assets.joblib` (overrides default search) |
| `SUBGROUP_CLIP_MAX` | Per-column clip upper bound (default `10`) |
| `CORS_ORIGINS` | Comma-separated allowed origins (default `*`, allows any origin) |
| `OPENAPI_SERVER_URL` | Server URL used by Swagger Try it out (default `http://localhost:8040`) |
# snsop_automatic_targeting
