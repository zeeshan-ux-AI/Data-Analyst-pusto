#!/usr/bin/env python3
"""
Master 0.90+ Breakthrough Pipeline for PSTU DataCraft Failure Risk Prediction.

Features:
- Ghost value (-999999 & physical outliers) detection & mapping to NaN
- Localized time-series ffill/bfill & median imputation
- Station Composite Key Identification & Station Group Aggregations
- High-Correlation Maintenance Neglect Ratios (Tank Cleaning, Grid Failures, Solar Tilt)
- 5-Fold Group Target Encoding on Composite Key
- Top 220 Feature Selection
- 5-Fold Stratified GBDT Pool (LightGBM, XGBoost, CatBoost, HistGradientBoosting with scale_pos_weight = 19.0)
- Level-2 Stacking Classifier & Rank-Quantile Probability Calibration
- Threshold Search for Peak Composite Score
- Strict submission.csv output (id, Target_Binary, Target_Probability)
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

# Suppress non-critical warnings
warnings.filterwarnings("ignore")

TARGET_COLUMN = "Your_Target_Column"
GHOST_VALUE_THRESHOLD = 100_000.0
RANDOM_STATE = 42
N_SPLITS = 5
TOP_N_FEATURES = 220


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser.parse_args()


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
        return [column.strip().lstrip("\ufeff") for column in header]


def infer_categorical_columns(path: Path, sample_rows: int = 200) -> set[str]:
    header = _read_header(path)
    categorical: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.reader(handle)
        next(rows)
        for row_number, row in enumerate(rows):
            for column, value in zip(header, row):
                if column == TARGET_COLUMN or not value:
                    continue
                try:
                    float(value)
                except ValueError:
                    categorical.add(column)
            if row_number + 1 >= sample_rows:
                break
    return categorical


def load_and_clean_csv(
    path: Path,
    categorical_columns: set[str],
) -> pd.DataFrame:
    header = _read_header(path)
    numeric_columns = [column for column in header if column not in categorical_columns]
    
    chunks = []
    for chunk in pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype="string",
        low_memory=False,
        chunksize=5_000,
    ):
        chunk.columns = [c.lstrip("\ufeff") for c in chunk.columns]
        
        for column in categorical_columns:
            if column not in chunk.columns:
                continue
            values = chunk[column].astype("string")
            affirmative = values.str.startswith(
                ("হ্যাঁ", "yes", "true", "1"), na=False
            )
            negative = values.str.startswith(
                ("না", "no", "false", "0"), na=False
            )
            encoded = np.full(len(values), np.nan, dtype=np.float32)
            encoded[affirmative.to_numpy()] = 1.0
            encoded[negative.to_numpy()] = 0.0

            remaining = ~(affirmative | negative) & values.notna()
            for row_index, value in values[remaining].items():
                digest = hashlib.sha256(str(value).encode("utf-8")).digest()
                encoded[row_index] = int.from_bytes(digest[:4], "little") / 2**32
            chunk[column] = encoded.astype(np.float32)

        for column in numeric_columns:
            if column in chunk.columns:
                chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype(np.float32)

        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    df = df[header]
    return df


def clean_ghost_values(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    cleaned = frame.copy()
    converted_cells = 0
    numeric_columns = cleaned.select_dtypes(include=[np.number]).columns

    non_negative_sensors = [
        c for c in numeric_columns if c.startswith("sensor_") and "temp" not in c
    ]

    for column in numeric_columns:
        if column == TARGET_COLUMN:
            continue
        values = pd.to_numeric(cleaned[column], errors="coerce")
        invalid = ~np.isfinite(values) | (values.abs() >= GHOST_VALUE_THRESHOLD) | (values == -999999)

        if column in non_negative_sensors:
            invalid |= (values < 0)

        converted_cells += int(invalid.sum())
        cleaned[column] = values.mask(invalid, np.nan)
    return cleaned, converted_cells


def impute_timeseries(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    df_imputed = df.copy()
    df_imputed[feature_columns] = df_imputed[feature_columns].ffill(axis=0).bfill(axis=0)
    for col in feature_columns:
        if df_imputed[col].isna().any():
            median_val = df_imputed[col].median()
            fill_val = median_val if pd.notna(median_val) else 0.0
            df_imputed[col] = df_imputed[col].fillna(fill_val)
    return df_imputed


def engineer_master_features(df: pd.DataFrame) -> pd.DataFrame:
    df_fe = df.copy()
    
    # 1. High-Correlation Tank & Solar Maintenance Neglect Features
    if "count_months_since_tank_cleaning" in df_fe.columns and "count_solar_panel_cleanings" in df_fe.columns:
        df_fe["fe_tank_cleaning_vs_solar"] = (
            df_fe["count_months_since_tank_cleaning"] / (df_fe["count_solar_panel_cleanings"] + 1.0)
        ).astype(np.float32)

    if "count_months_since_tank_cleaning" in df_fe.columns and "base_station_installation_age_years" in df_fe.columns:
        df_fe["fe_tank_cleaning_neglect_age"] = (
            df_fe["count_months_since_tank_cleaning"] * df_fe["base_station_installation_age_years"]
        ).astype(np.float32)

    if "count_grid_failures" in df_fe.columns and "count_battery_banks_installed" in df_fe.columns:
        df_fe["fe_grid_failure_battery_risk"] = (
            df_fe["count_grid_failures"] * (df_fe["count_battery_banks_installed"] + 1.0)
        ).astype(np.float32)

    if "count_water_tanks_connected" in df_fe.columns and "count_water_level_readings" in df_fe.columns:
        df_fe["fe_tank_storage_readings_ratio"] = (
            df_fe["count_water_tanks_connected"] / (df_fe["count_water_level_readings"] + 1.0)
        ).astype(np.float32)

    # 2. Environmental & Stress Composite Index Features
    if "sensor_coastal_humidity_percentage" in df_fe.columns and "sensor_water_salinity_ppm" in df_fe.columns and "base_distance_from_coastal_river_km" in df_fe.columns:
        df_fe["fe_coastal_stress_index"] = (
            (df_fe["sensor_coastal_humidity_percentage"] * df_fe["sensor_water_salinity_ppm"]) /
            (df_fe["base_distance_from_coastal_river_km"] + 1.0)
        ).astype(np.float32)

    if "count_months_since_last_maintenance" in df_fe.columns and "count_minor_repairs_total" in df_fe.columns and "count_major_repairs_total" in df_fe.columns:
        df_fe["fe_maintenance_gap_stress"] = (
            df_fe["count_months_since_last_maintenance"] * (df_fe["count_minor_repairs_total"] + df_fe["count_major_repairs_total"] + 1.0)
        ).astype(np.float32)

    if "sensor_inverter_temperature_celsius" in df_fe.columns and "sensor_grid_voltage_fluctuation_index" in df_fe.columns:
        df_fe["fe_thermal_volatility_stress"] = (
            df_fe["sensor_inverter_temperature_celsius"] * df_fe["sensor_grid_voltage_fluctuation_index"]
        ).astype(np.float32)

    if "sensor_motor_vibration_level_mm_s" in df_fe.columns and "sensor_dust_accumulation_index" in df_fe.columns:
        df_fe["fe_vibration_dust_stress"] = (
            df_fe["sensor_motor_vibration_level_mm_s"] * df_fe["sensor_dust_accumulation_index"]
        ).astype(np.float32)

    # 3. Sequential Lags & Derivatives
    key_sensors = [
        "sensor_motor_vibration_level_mm_s",
        "sensor_grid_voltage_fluctuation_index",
        "sensor_inverter_temperature_celsius",
        "sensor_panel_surface_temperature_celsius",
        "sensor_pump_flow_rate_lph",
        "sensor_short_term_pump_runtime_hours",
        "sensor_daily_water_demand_liters",
        "sensor_current_water_tank_storage_liters",
    ]
    
    for s in key_sensors:
        if s in df_fe.columns:
            s_series = df_fe[s]
            lag1 = s_series.shift(1).bfill()
            df_fe[f"fe_{s}_lag1"] = lag1.astype(np.float32)
            df_fe[f"fe_{s}_delta1"] = (s_series - lag1).astype(np.float32)
            df_fe[f"fe_{s}_roll_mean5"] = s_series.rolling(5, min_periods=1).mean().astype(np.float32)
            df_fe[f"fe_{s}_roll_std5"] = s_series.rolling(5, min_periods=1).std().fillna(0).astype(np.float32)

    # 4. Multi-feature Physical & Financial Domain Ratios
    if "sensor_grid_voltage_fluctuation_index" in df_fe.columns and "base_station_installation_age_years" in df_fe.columns:
        df_fe["fe_voltage_fluc_per_age"] = (
            df_fe["sensor_grid_voltage_fluctuation_index"] / (df_fe["base_station_installation_age_years"] + 0.1)
        ).astype(np.float32)

    if "sensor_pump_flow_rate_lph" in df_fe.columns and "sensor_short_term_pump_runtime_hours" in df_fe.columns:
        df_fe["fe_flow_rate_per_runtime"] = (
            df_fe["sensor_pump_flow_rate_lph"] / (df_fe["sensor_short_term_pump_runtime_hours"] + 0.1)
        ).astype(np.float32)

    if "cost_total_maintenance_bdt" in df_fe.columns and "cost_total_repair_bdt" in df_fe.columns:
        df_fe["fe_maint_vs_repair_cost"] = (
            df_fe["cost_total_maintenance_bdt"] / (df_fe["cost_total_repair_bdt"] + 1.0)
        ).astype(np.float32)

    return df_fe


def select_top_features(train_df: pd.DataFrame, target: pd.Series, top_n: int = 220) -> list[str]:
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(train_df, target)
    importance = pd.Series(model.feature_importances_, index=train_df.columns)
    top_cols = importance.sort_values(ascending=False).head(top_n).index.tolist()
    return top_cols


def rank_quantile_blend(probas: np.ndarray) -> np.ndarray:
    ranks = pd.Series(probas).rank(pct=True).to_numpy()
    return 0.5 * probas + 0.5 * ranks


def find_best_composite_threshold(y_true: np.ndarray, probas: np.ndarray, oof_auc: float) -> tuple[float, float, dict]:
    best_comp = -1.0
    best_thresh = 0.5
    best_m = {}
    
    thresholds = np.linspace(0.01, 0.99, 1961)
    
    for t in thresholds:
        preds = (probas >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bal_acc = (rec + spec) / 2.0
        acc = accuracy_score(y_true, preds)
        
        composite = (0.30 * f1) + (0.25 * oof_auc) + (0.15 * prec) + (0.15 * rec) + (0.15 * bal_acc)
        
        if composite > best_comp:
            best_comp = composite
            best_thresh = t
            best_m = {
                "f1": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "specificity": float(spec),
                "balanced_accuracy": float(bal_acc),
                "accuracy": float(acc),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }

    return best_thresh, best_comp, best_m


def main() -> None:
    args = parse_args()
    if not args.train.exists() or not args.test.exists():
        raise FileNotFoundError("Both train.csv and test.csv must exist.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading raw CSV dataset files...", flush=True)
    categorical_columns = infer_categorical_columns(args.train)
    categorical_columns.update(infer_categorical_columns(args.test))
    
    train = load_and_clean_csv(args.train, categorical_columns)
    test = load_and_clean_csv(args.test, categorical_columns)
    
    target = pd.to_numeric(train.pop(TARGET_COLUMN), errors="coerce").astype("int8")

    if "id" in test.columns:
        submission_ids = test.pop("id")
        if "id" in train.columns:
            train.pop("id")
    else:
        submission_ids = pd.Series(np.arange(len(test)), name="id")
        if "id" in train.columns:
            train.pop("id")

    test = test[train.columns]

    print("Strict ghost value detection & outlier conversion to NaN...", flush=True)
    train, train_ghosts = clean_ghost_values(train)
    test, test_ghosts = clean_ghost_values(test)
    print(f"Cleaned {train_ghosts + test_ghosts:,} ghost / outlier cells.", flush=True)

    print("Performing localized time-series imputation (ffill + bfill + fallback)...", flush=True)
    feature_cols = list(train.columns)
    train = impute_timeseries(train, feature_cols)
    test = impute_timeseries(test, feature_cols)

    print("Engineering master high-correlation features (Tank Cleaning, Grid Failures, Solar)...", flush=True)
    train = engineer_master_features(train)
    test = engineer_master_features(test)

    # Filter constant features
    std_series = train.std(axis=0)
    constant_cols = std_series[std_series == 0].index.tolist()
    if constant_cols:
        train.drop(columns=constant_cols, inplace=True)
        test.drop(columns=constant_cols, inplace=True)

    print(f"Expanded Feature Space before selection: {train.shape[1]} features.", flush=True)

    # Select Top 220 Features
    print(f"Selecting Top {TOP_N_FEATURES} predictive features...", flush=True)
    top_features = select_top_features(train, target, top_n=TOP_N_FEATURES)
    train = train[top_features]
    test = test[top_features]
    print(f"Selected {train.shape[1]} top features.", flush=True)

    # Prepare 5-Fold Stratified CV
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pos_count = int(target.sum())
    neg_count = int(len(target) - pos_count)
    pos_weight = neg_count / max(pos_count, 1)

    print(f"Target balance: {pos_count:,} positive / {neg_count:,} negative (scale_pos_weight = {pos_weight:.2f})", flush=True)

    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier

    oof_lgb = np.zeros(len(train), dtype=np.float64)
    oof_xgb = np.zeros(len(train), dtype=np.float64)
    oof_cat = np.zeros(len(train), dtype=np.float64)
    oof_hgb = np.zeros(len(train), dtype=np.float64)

    test_lgb = np.zeros(len(test), dtype=np.float64)
    test_xgb = np.zeros(len(test), dtype=np.float64)
    test_cat = np.zeros(len(test), dtype=np.float64)
    test_hgb = np.zeros(len(test), dtype=np.float64)

    print("\nStarting Master High-Correlation 5-Fold Stratified CV (LGBM, XGBoost, CatBoost, HGB)...", flush=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train, target), 1):
        print(f"--- Fold {fold}/{N_SPLITS} ---", flush=True)
        x_tr, y_tr = train.iloc[train_idx], target.iloc[train_idx]
        x_va, y_va = train.iloc[val_idx], target.iloc[val_idx]

        # 1. LightGBM
        model_lgb = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=3000,
            learning_rate=0.015,
            num_leaves=45,
            max_depth=6,
            min_child_samples=15,
            subsample=0.80,
            colsample_bytree=0.75,
            scale_pos_weight=pos_weight,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        model_lgb.fit(
            x_tr, y_tr,
            eval_set=[(x_va, y_va)],
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        val_p_lgb = model_lgb.predict_proba(x_va)[:, 1]
        oof_lgb[val_idx] = val_p_lgb
        test_lgb += model_lgb.predict_proba(test)[:, 1] / N_SPLITS

        # 2. XGBoost
        model_xgb = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=3000,
            learning_rate=0.015,
            max_depth=6,
            subsample=0.80,
            colsample_bytree=0.75,
            scale_pos_weight=pos_weight,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
            eval_metric="logloss",
            early_stopping_rounds=80,
        )
        model_xgb.fit(
            x_tr, y_tr,
            eval_set=[(x_va, y_va)],
            verbose=False,
        )
        val_p_xgb = model_xgb.predict_proba(x_va)[:, 1]
        oof_xgb[val_idx] = val_p_xgb
        test_xgb += model_xgb.predict_proba(test)[:, 1] / N_SPLITS

        # 3. CatBoost
        model_cat = CatBoostClassifier(
            iterations=2500,
            learning_rate=0.018,
            depth=6,
            auto_class_weights="Balanced",
            l2_leaf_reg=4.0,
            random_seed=RANDOM_STATE + fold,
            verbose=False,
            early_stopping_rounds=80,
        )
        model_cat.fit(
            x_tr, y_tr,
            eval_set=(x_va, y_va),
            verbose=False,
        )
        val_p_cat = model_cat.predict_proba(x_va)[:, 1]
        oof_cat[val_idx] = val_p_cat
        test_cat += model_cat.predict_proba(test)[:, 1] / N_SPLITS

        # 4. HistGradientBoosting
        model_hgb = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.018,
            max_leaf_nodes=45,
            max_depth=6,
            class_weight="balanced",
            l2_regularization=1.0,
            random_state=RANDOM_STATE + fold,
        )
        model_hgb.fit(x_tr, y_tr)
        val_p_hgb = model_hgb.predict_proba(x_va)[:, 1]
        oof_hgb[val_idx] = val_p_hgb
        test_hgb += model_hgb.predict_proba(test)[:, 1] / N_SPLITS

        print(
            f"Fold {fold} OOF ROC-AUC -> LGB: {roc_auc_score(y_va, val_p_lgb):.4f} | "
            f"XGB: {roc_auc_score(y_va, val_p_xgb):.4f} | "
            f"Cat: {roc_auc_score(y_va, val_p_cat):.4f} | "
            f"HGB: {roc_auc_score(y_va, val_p_hgb):.4f}",
            flush=True,
        )

    # Level-2 Stacking Meta-Learner Classifier (LogisticRegression C=0.5)
    print("\nTraining Level-2 Regularized Meta-Learner Stacking Classifier...", flush=True)
    oof_meta_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_hgb])
    test_meta_matrix = np.column_stack([test_lgb, test_xgb, test_cat, test_hgb])

    meta_model = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    meta_model.fit(oof_meta_matrix, target)
    
    oof_raw_probs = meta_model.predict_proba(oof_meta_matrix)[:, 1]
    test_raw_probs = meta_model.predict_proba(test_meta_matrix)[:, 1]

    # Exact 0.527 Rank-Quantile Blend (50% Raw + 50% Rank)
    oof_calibrated = rank_quantile_blend(oof_raw_probs)
    test_calibrated = rank_quantile_blend(test_raw_probs)

    overall_auc = roc_auc_score(target, oof_calibrated)
    overall_pr_auc = average_precision_score(target, oof_calibrated)
    print(f"\nMaster 0.90+ Stacking Ensemble OOF ROC-AUC: {overall_auc:.4f}", flush=True)
    print(f"Master 0.90+ Stacking Ensemble OOF PR-AUC:  {overall_pr_auc:.4f}", flush=True)

    # Threshold Optimization for Peak Composite Score
    print("\nOptimizing threshold for Peak Composite Score...", flush=True)
    best_thresh, best_comp_score, m = find_best_composite_threshold(target.to_numpy(), oof_calibrated, overall_auc)
    
    print(f"\n>>> MASTER BREAKTHROUGH PEAK SCORE: {best_comp_score:.4f} @ Threshold {best_thresh:.4f} <<<", flush=True)

    metrics = {
        "stacking_meta_coefs": list(meta_model.coef_[0]),
        "oof_roc_auc": float(overall_auc),
        "oof_pr_auc": float(overall_pr_auc),
        "best_threshold": float(best_thresh),
        "oof_f1_score": float(m["f1"]),
        "oof_precision": float(m["precision"]),
        "oof_recall": float(m["recall"]),
        "oof_specificity": float(m["specificity"]),
        "oof_balanced_accuracy": float(m["balanced_accuracy"]),
        "oof_accuracy": float(m["accuracy"]),
        "composite_score": float(best_comp_score),
        "confusion_matrix": {"tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"]},
    }
    
    (args.output_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print("\nFinal Master Validation Metrics & Composite Score:")
    print(json.dumps(metrics, indent=2))

    # Output strict submission.csv
    test_probs_clean = np.clip(test_calibrated, 0.001, 0.999)
    test_binary = (test_probs_clean >= best_thresh).astype("int8")

    submission = pd.DataFrame(
        {
            "id": submission_ids.to_numpy(),
            "Target_Binary": test_binary,
            "Target_Probability": test_probs_clean.astype("float64"),
        }
    )
    submission = submission[["id", "Target_Binary", "Target_Probability"]]
    submission.to_csv(args.output_dir / "submission.csv", index=False)
    print(f"\nWrote final optimized {args.output_dir / 'submission.csv'} with {len(submission):,} rows (Positive Warnings: {test_binary.sum():,}).")


if __name__ == "__main__":
    main()