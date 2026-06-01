"""
FastAPI service: household member counts -> support type + top SHAP features.

Aligns with training-time preprocessing:
  - Column order: FEATURE_COLS / RAW_MEMBER_COLS (gender × age × status).
  - Counts clipped to [0, THRESHOLD_PER_SUBGROUP] per column (default 10).

**Real model bundle** (from your Colab / notebook):

    joblib.dump(
        {
            "model": model_support_type,      # e.g. XGBClassifier
            "feature_names": FEATURE_COLS,   # list, same order as training
            "label_encoder": le,
        },
        "support_type_model_assets.joblib",
    )

Place the file under `backend/models/` (recommended) or next to `main.py`.
Override path with env **`ASSETS_PATH`**. If no file is found, a dummy
`RandomForestClassifier` is trained for local development.

Requires **`xgboost`** installed when loading an XGBoost model (see `requirements.txt`).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- Same layout as your training / assembly code ----------------------------

GENDERS = ["male", "female"]
AGE_GROUPS = ["0_2", "3_5", "6_17", "18_35", "36_64", "65_plus"]
STATUSES = ["normal", "disabled", "chronically_ill", "both"]
WORKING_AGES = ["18_35", "36_64"]


def member_col(gender: str, age_group: str, status: str) -> str:
    return f"{gender}_{age_group}_{status}"


RAW_MEMBER_COLS: list[str] = [
    member_col(g, a, s) for g in GENDERS for a in AGE_GROUPS for s in STATUSES
]

# Alias used in many notebooks / joblib bundles
FEATURE_COLS: list[str] = RAW_MEMBER_COLS

# Default matches assembled_households[RAW_MEMBER_COLS].clip(0, THRESHOLD_PER_SUBGROUP)
THRESHOLD_PER_SUBGROUP = 10


def _clip_upper() -> int:
    return int(os.environ.get("SUBGROUP_CLIP_MAX", str(THRESHOLD_PER_SUBGROUP)))


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float


class SupportTypeResponse(BaseModel):
    predicted_support_type: str
    top_features: list[FeatureContribution]


class HouseholdCounts(BaseModel):
    """Household member counts grouped by gender, age group, and condition."""

    model_config = ConfigDict(extra="ignore")

    male_0_2_normal: int = Field(0, ge=0, description="Male children age 0-2 with no disability or chronic illness.")
    male_0_2_disabled: int = Field(0, ge=0, description="Male children age 0-2 with a disability.")
    male_0_2_chronically_ill: int = Field(0, ge=0, description="Male children age 0-2 with a chronic illness.")
    male_0_2_both: int = Field(0, ge=0, description="Male children age 0-2 with both disability and chronic illness.")
    male_3_5_normal: int = Field(0, ge=0, description="Male children age 3-5 with no disability or chronic illness.")
    male_3_5_disabled: int = Field(0, ge=0, description="Male children age 3-5 with a disability.")
    male_3_5_chronically_ill: int = Field(0, ge=0, description="Male children age 3-5 with a chronic illness.")
    male_3_5_both: int = Field(0, ge=0, description="Male children age 3-5 with both disability and chronic illness.")
    male_6_17_normal: int = Field(0, ge=0, description="Male children age 6-17 with no disability or chronic illness.")
    male_6_17_disabled: int = Field(0, ge=0, description="Male children age 6-17 with a disability.")
    male_6_17_chronically_ill: int = Field(0, ge=0, description="Male children age 6-17 with a chronic illness.")
    male_6_17_both: int = Field(0, ge=0, description="Male children age 6-17 with both disability and chronic illness.")
    male_18_35_normal: int = Field(0, ge=0, description="Male adults age 18-35 with no disability or chronic illness.")
    male_18_35_disabled: int = Field(0, ge=0, description="Male adults age 18-35 with a disability.")
    male_18_35_chronically_ill: int = Field(0, ge=0, description="Male adults age 18-35 with a chronic illness.")
    male_18_35_both: int = Field(0, ge=0, description="Male adults age 18-35 with both disability and chronic illness.")
    male_36_64_normal: int = Field(0, ge=0, description="Male adults age 36-64 with no disability or chronic illness.")
    male_36_64_disabled: int = Field(0, ge=0, description="Male adults age 36-64 with a disability.")
    male_36_64_chronically_ill: int = Field(0, ge=0, description="Male adults age 36-64 with a chronic illness.")
    male_36_64_both: int = Field(0, ge=0, description="Male adults age 36-64 with both disability and chronic illness.")
    male_65_plus_normal: int = Field(0, ge=0, description="Male adults age 65+ with no disability or chronic illness.")
    male_65_plus_disabled: int = Field(0, ge=0, description="Male adults age 65+ with a disability.")
    male_65_plus_chronically_ill: int = Field(0, ge=0, description="Male adults age 65+ with a chronic illness.")
    male_65_plus_both: int = Field(0, ge=0, description="Male adults age 65+ with both disability and chronic illness.")
    female_0_2_normal: int = Field(0, ge=0, description="Female children age 0-2 with no disability or chronic illness.")
    female_0_2_disabled: int = Field(0, ge=0, description="Female children age 0-2 with a disability.")
    female_0_2_chronically_ill: int = Field(0, ge=0, description="Female children age 0-2 with a chronic illness.")
    female_0_2_both: int = Field(0, ge=0, description="Female children age 0-2 with both disability and chronic illness.")
    female_3_5_normal: int = Field(0, ge=0, description="Female children age 3-5 with no disability or chronic illness.")
    female_3_5_disabled: int = Field(0, ge=0, description="Female children age 3-5 with a disability.")
    female_3_5_chronically_ill: int = Field(0, ge=0, description="Female children age 3-5 with a chronic illness.")
    female_3_5_both: int = Field(0, ge=0, description="Female children age 3-5 with both disability and chronic illness.")
    female_6_17_normal: int = Field(0, ge=0, description="Female children age 6-17 with no disability or chronic illness.")
    female_6_17_disabled: int = Field(0, ge=0, description="Female children age 6-17 with a disability.")
    female_6_17_chronically_ill: int = Field(0, ge=0, description="Female children age 6-17 with a chronic illness.")
    female_6_17_both: int = Field(0, ge=0, description="Female children age 6-17 with both disability and chronic illness.")
    female_18_35_normal: int = Field(0, ge=0, description="Female adults age 18-35 with no disability or chronic illness.")
    female_18_35_disabled: int = Field(0, ge=0, description="Female adults age 18-35 with a disability.")
    female_18_35_chronically_ill: int = Field(0, ge=0, description="Female adults age 18-35 with a chronic illness.")
    female_18_35_both: int = Field(0, ge=0, description="Female adults age 18-35 with both disability and chronic illness.")
    female_36_64_normal: int = Field(0, ge=0, description="Female adults age 36-64 with no disability or chronic illness.")
    female_36_64_disabled: int = Field(0, ge=0, description="Female adults age 36-64 with a disability.")
    female_36_64_chronically_ill: int = Field(0, ge=0, description="Female adults age 36-64 with a chronic illness.")
    female_36_64_both: int = Field(0, ge=0, description="Female adults age 36-64 with both disability and chronic illness.")
    female_65_plus_normal: int = Field(0, ge=0, description="Female adults age 65+ with no disability or chronic illness.")
    female_65_plus_disabled: int = Field(0, ge=0, description="Female adults age 65+ with a disability.")
    female_65_plus_chronically_ill: int = Field(0, ge=0, description="Female adults age 65+ with a chronic illness.")
    female_65_plus_both: int = Field(0, ge=0, description="Female adults age 65+ with both disability and chronic illness.")


class HouseholdRequest(BaseModel):
    """Flat household counters; missing keys default to 0."""

    household: dict[str, int] = Field(default_factory=dict)


class GetSupportTypeRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
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
                        "female_65_plus_both": 0,
                    }
                }
            ]
        }
    )

    household: HouseholdCounts = Field(
        default_factory=HouseholdCounts,
        description="Counts of household members by gender, age group, and condition. Leave unknown groups as 0.",
    )


class GetSupportTypeResponse(BaseModel):
    support_type: str = Field(..., description="Predicted support type label.")


def _col_index_map(cols: list[str]) -> dict[str, int]:
    return {c: i for i, c in enumerate(cols)}


def _disabled_working_age_sum(row: np.ndarray, idx: dict[str, int]) -> float:
    s = 0.0
    for g in GENDERS:
        for a in WORKING_AGES:
            c = member_col(g, a, "disabled")
            s += float(row[idx[c]])
    return s


def _both_status_sum(row: np.ndarray, idx: dict[str, int]) -> float:
    s = 0.0
    for c in FEATURE_COLS:
        if c.endswith("_both"):
            s += float(row[idx[c]])
    return s


def _synthetic_labels_from_counts(X: np.ndarray, cols: list[str], rng: np.random.Generator) -> np.ndarray:
    """
    Rule-based labels correlated with working-age disabled + both counts,
    so SHAP tends to highlight plausible columns (similar qualitative behaviour
    to a real support-type model without your proprietary weights).
    """
    idx = _col_index_map(cols)
    scores = np.empty(len(X), dtype=np.float64)
    for i in range(len(X)):
        scores[i] = _disabled_working_age_sum(X[i], idx) + 0.45 * _both_status_sum(X[i], idx)
    scores = scores + rng.normal(0, 0.35, size=len(X))
    return np.where(scores >= 3.0, "DIS", np.where(scores >= 1.25, "FSV", "HEV"))


def _candidate_asset_paths() -> list[str]:
    """Default search order when ASSETS_PATH is unset."""
    base = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(base, "models", "support_type_model_assets.joblib"),
        os.path.join(base, "support_type_model_assets.joblib"),
    ]


def resolve_assets_path() -> str | None:
    """
    Resolve path to support_type_model_assets.joblib.

    1. If ASSETS_PATH is set and that file exists, use it.
    2. Else first existing file among _candidate_asset_paths().
    3. Else None (dummy model).
    """
    env = os.environ.get("ASSETS_PATH", "").strip()
    if env and os.path.isfile(env):
        return env
    for p in _candidate_asset_paths():
        if os.path.isfile(p):
            return p
    return None


def _validate_model_assets(data: dict[str, Any], path: str) -> dict[str, Any]:
    """Ensure joblib dict matches the Colab export shape."""
    required = ("model", "feature_names", "label_encoder")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            f"Joblib at {path!r} is missing keys {missing}. "
            "Save with: model, feature_names, label_encoder "
            "(see Colab snippet / backend/README.md)."
        )
    fn = data["feature_names"]
    if not isinstance(fn, (list, tuple)) or not fn:
        raise ValueError("feature_names must be a non-empty list (your FEATURE_COLS).")
    data["feature_names"] = list(fn)
    # Back-compat if an older bundle used FEATURE_COLS only (already copied earlier)
    return data


def _build_dummy_assets() -> dict[str, Any]:
    rng = np.random.default_rng(42)
    cols = list(FEATURE_COLS)
    n = len(cols)
    cap = _clip_upper()

    # Sparse-ish integer counts in [0, cap] (similar scale to clipped rollups)
    p_hit = 0.11
    X = rng.binomial(cap, p_hit, size=(900, n)).astype(np.float64)
    for _ in range(120):
        i = rng.integers(0, len(X))
        j = rng.integers(0, n)
        X[i, j] = float(rng.integers(1, cap + 1))

    labels = _synthetic_labels_from_counts(X, cols, rng)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    model = RandomForestClassifier(
        n_estimators=96,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    # Fit with named columns so predict(DataFrame) matches your real pipeline.
    X_df = pd.DataFrame(X, columns=cols)
    model.fit(X_df, y)
    explainer = shap.TreeExplainer(model)
    return {
        "model": model,
        "feature_names": cols,
        "label_encoder": le,
        "explainer": explainer,
        "_loaded_from": "dummy",
    }


def load_assets() -> dict[str, Any]:
    path = resolve_assets_path()
    if path is None:
        return _build_dummy_assets()

    try:
        data: dict[str, Any] = joblib.load(path)
    except ModuleNotFoundError as e:
        mod = getattr(e, "name", None) or ""
        hint = (
            f"Could not load {path!r}: Python tried to import {mod!r} while unpickling your model.\n"
            "  • If this is an XGBoost model, install it in **this** virtualenv:\n"
            "        pip install xgboost\n"
            "    (or: pip install -r requirements.txt from the backend folder.)\n"
            "  • If `pip install xgboost` fails with “no matching distribution”, your Python is likely "
            "too new for published wheels (common on **Python 3.14**). Recreate the venv with "
            "**Python 3.11 or 3.12**, then reinstall dependencies."
        )
        raise RuntimeError(hint) from e

    if "feature_names" not in data and "FEATURE_COLS" in data:
        data["feature_names"] = list(data["FEATURE_COLS"])
    data = _validate_model_assets(data, path)
    data["_loaded_from"] = path

    if "explainer" not in data:
        data["explainer"] = shap.TreeExplainer(data["model"])

    return data


assets: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global assets
    assets = load_assets()
    yield


app = FastAPI(
    title="Support type prediction",
    lifespan=lifespan,
    servers=[{"url": os.environ.get("OPENAPI_SERVER_URL", "http://localhost:8040")}],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


def _household_payload(req: HouseholdRequest | GetSupportTypeRequest) -> dict[str, int]:
    household = req.household or {}
    if isinstance(household, BaseModel):
        return household.model_dump()
    return household


def _household_dataframe(req: HouseholdRequest | GetSupportTypeRequest) -> pd.DataFrame:
    """
    Step 1–3 from your notebook: one dict -> DataFrame -> columns in model order -> clip.
    """
    h = _household_payload(req)
    names: list[str] = list(assets["feature_names"])
    row = {k: int(h.get(k, 0) or 0) for k in names}
    new_df = pd.DataFrame([row])
    new_df = new_df[names]
    cap = _clip_upper()
    new_df[names] = new_df[names].clip(lower=0, upper=cap)
    return new_df


def _shap_vals_for_predicted_class(shap_values: Any, pred_class: int, n_features: int) -> np.ndarray:
    """
    Match your SHAP handling:
      - Older SHAP -> list of arrays: shap_values[pred_class][0]
      - Newer SHAP -> ndarray: shap_values[0, :, pred_class]
    Plus small fallbacks for 2-D single-output layouts.
    """
    pc = int(pred_class)

    if isinstance(shap_values, list):
        shap_vals_for_class = np.asarray(shap_values[pc], dtype=np.float64)[0]
    else:
        arr = np.asarray(shap_values, dtype=np.float64)
        if arr.ndim == 3:
            shap_vals_for_class = arr[0, :, pc]
        elif arr.ndim == 2:
            shap_vals_for_class = arr[0]
        elif arr.ndim == 1:
            shap_vals_for_class = arr
        else:
            shap_vals_for_class = np.ravel(arr)[:n_features]

    vec = np.ravel(shap_vals_for_class).astype(np.float64)
    if vec.size != n_features:
        if vec.size > n_features:
            vec = vec[:n_features]
        else:
            vec = np.pad(vec, (0, n_features - vec.size))
    return vec


def _predict_support_type_label(req: HouseholdRequest | GetSupportTypeRequest) -> str:
    model = assets["model"]
    le: LabelEncoder = assets["label_encoder"]
    df = _household_dataframe(req)
    pred_class = int(model.predict(df)[0])
    return str(le.inverse_transform([pred_class])[0])


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "features": len(assets.get("feature_names", [])),
        "subgroup_clip_max": _clip_upper(),
        "assets_source": assets.get("_loaded_from"),
    }


@app.post("/predict_support_type", response_model=SupportTypeResponse)
def predict_support_type(req: HouseholdRequest) -> SupportTypeResponse:
    model = assets["model"]
    feature_names: list[str] = list(assets["feature_names"])
    le: LabelEncoder = assets["label_encoder"]
    explainer: shap.TreeExplainer = assets["explainer"]

    df = _household_dataframe(req)

    pred_class = int(model.predict(df)[0])
    predicted_label = str(le.inverse_transform([pred_class])[0])

    shap_values = explainer.shap_values(df)
    n_features = len(feature_names)
    shap_row = _shap_vals_for_predicted_class(shap_values, pred_class, n_features)

    feature_contrib = pd.DataFrame({"feature": feature_names, "shap_value": shap_row})
    feature_contrib["abs_shap"] = feature_contrib["shap_value"].abs()
    top = feature_contrib.sort_values(by="abs_shap", ascending=False).head(5)

    top_features = [
        FeatureContribution(feature=str(r["feature"]), shap_value=float(r["shap_value"]))
        for _, r in top.iterrows()
    ]

    return SupportTypeResponse(
        predicted_support_type=predicted_label,
        top_features=top_features,
    )


@app.post("/get_support_type", response_model=GetSupportTypeResponse)
def get_support_type(req: GetSupportTypeRequest) -> GetSupportTypeResponse:
    return GetSupportTypeResponse(support_type=_predict_support_type_label(req))
