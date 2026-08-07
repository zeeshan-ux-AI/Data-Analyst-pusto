#!/usr/bin/env python3
"""
Two-Stage Stacking Meta-Learner & Time-Series Derivative Pipeline for PSTU DataCraft Failure Risk Prediction.

Features:
- Ghost value (-999999 & physical outliers) detection & mapping to NaN
- Localized time-series ffill/bfill & median imputation
- Time-Series Lags (shift 1, shift 2), Rate-of-Change Derivatives (dx/dt), Rolling EMA & Volatility
- Multi-feature physical & financial ratios, risk flag intersections
- 5-Fold Stratified Cross-Validation
- Level-1 Ensembles: LightGBM, XGBoost, CatBoost, HistGradientBoosting, ExtraTrees
- Level-2 Stacking Meta-Learner (Logistic Regression Meta-Classifier)
- Direct Competition Composite Score Threshold Optimization (0.001 step resolution)
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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_fe = df.copy()
    
    # 1. Sequential Lags & Instantaneous Derivatives (Rate of Change) for Key Sensor Channels
    key_sensors = [
        "sensor_motor_vibration_level_mm_s",
        "sensor_grid_voltage_fluctuation_index",
        "sensor_inverter_temperature_celsius",
        "sensor_panel_surface_temperature_celsius",
        "sensor_pump_flow_rate_lph",
        "sensor_short_term_pump_runtime_hours",
        "sensor_daily_water_demand_liters",
        "sensor_current_water_tank_storage_liters",
        "sensor_coastal_humidity_percentage",
        "sensor_solar_irradiance_wm2",
    ]
    
    for s in key_sensors:
        if s in df_fe.columns:
            s_series = df_fe[s]
            lag1 = s_series.shift(1).bfill()
            lag2 = s_series.shift(2).bfill()
            
            df_fe[f"fe_{s}_lag1"] = lag1.astype(np.float32)
            df_fe[f"fe_{s}_delta1"] = (s_series - lag1).astype(np.float32)
            df_fe[f"fe_{s}_accel"] = ((s_series - lag1) - (lag1 - lag2)).astype(np.float32)
            
            # Rolling EMA & volatility
            df_fe[f"fe_{s}_roll_mean5"] = s_series.rolling(5, min_periods=1).mean().astype(np.float32)
            df_fe[f"fe_{s}_roll_std5"] = s_series.rolling(5, min_periods=1).std().fillna(0).astype(np.float32)
            df_fe[f"fe_{s}_ema5"] = s_series.ewm(span=5, min_periods=1).mean().astype(np.float32)

    # 2. Multi-window sensor family aggregations & rolling growth deltas
    sensor_families = {
        "temp": [c for c in df.columns if "temp" in c or "temperature" in c],
        "water_tank": [c for c in df.columns if "water_tank" in c or "storage" in c],
        "demand": [c for c in df.columns if "demand" in c],
        "short_runtime": [c for c in df.columns if "short_runtime" in c or "short_term_pump" in c],
        "long_runtime": [c for c in df.columns if "long_runtime" in c or "long_term_pump" in c],
        "humidity": [c for c in df.columns if "humidity" in c],
        "vibration": [c for c in df.columns if "vibration" in c],
        "irradiance": [c for c in df.columns if "irradiance" in c],
        "cycles": [c for c in df.columns if "cycles" in c],
    }

    for name, cols in sensor_families.items():
        valid_cols = [c for c in cols if c in df_fe.columns]
        if len(valid_cols) >= 2:
            df_fe[f"fe_{name}_mean"] = df_fe[valid_cols].mean(axis=1).astype(np.float32)
            df_fe[f"fe_{name}_std"] = df_fe[valid_cols].std(axis=1).fillna(0).astype(np.float32)
            df_fe[f"fe_{name}_min"] = df_fe[valid_cols].min(axis=1).astype(np.float32)
            df_fe[f"fe_{name}_max"] = df_fe[valid_cols].max(axis=1).astype(np.float32)

            col_last_m = [c for c in valid_cols if "last_month" in c]
            col_3m_ago = [c for c in valid_cols if "3_months_ago" in c]
            if col_last_m and col_3m_ago:
                df_fe[f"fe_{name}_delta_3m"] = (df_fe[col_last_m[0]] - df_fe[col_3m_ago[0]]).astype(np.float32)
                df_fe[f"fe_{name}_growth_ratio_3m"] = (
                    (df_fe[col_last_m[0]] + 1e-4) / (df_fe[col_3m_ago[0]] + 1e-4)
                ).astype(np.float32)

    # 3. Multi-feature Physical & Financial Domain Ratios
    if "sensor_grid_voltage_fluctuation_index" in df_fe.columns and "base_station_installation_age_years" in df_fe.columns:
        df_fe["fe_voltage_fluc_per_age"] = (
            df_fe["sensor_grid_voltage_fluctuation_index"] / (df_fe["base_station_installation_age_years"] + 0.1)
        ).astype(np.float32)

    if "sensor_pump_flow_rate_lph" in df_fe.columns and "sensor_short_term_pump_runtime_hours" in df_fe.columns:
        df_fe["fe_flow_rate_per_runtime"] = (
            df_fe["sensor_pump_flow_rate_lph"] / (df_fe["sensor_short_term_pump_runtime_hours"] + 0.1)
        ).astype(np.float32)

    if "sensor_daily_water_demand_liters" in df_fe.columns and "sensor_current_water_tank_storage_liters" in df_fe.columns:
        df_fe["fe_demand_vs_tank_capacity"] = (
            df_fe["sensor_daily_water_demand_liters"] / (df_fe["sensor_current_water_tank_storage_liters"] + 1.0)
        ).astype(np.float32)

    if "cost_total_maintenance_bdt" in df_fe.columns and "cost_total_repair_bdt" in df_fe.columns:
        df_fe["fe_maint_vs_repair_cost"] = (
            df_fe["cost_total_maintenance_bdt"] / (df_fe["cost_total_repair_bdt"] + 1.0)
        ).astype(np.float32)

    if "cost_commercial_maintenance_bdt" in df_fe.columns and "cost_total_maintenance_bdt" in df_fe.columns:
        df_fe["fe_commercial_maint_ratio"] = (
            df_fe["cost_commercial_maintenance_bdt"] / (df_fe["cost_total_maintenance_bdt"] + 1.0)
        ).astype(np.float32)

    if "cost_commercial_repair_bdt" in df_fe.columns and "cost_total_repair_bdt" in df_fe.columns:
        df_fe["fe_commercial_repair_ratio"] = (
            df_fe["cost_commercial_repair_bdt"] / (df_fe["cost_total_repair_bdt"] + 1.0)
        ).astype(np.float32)

    if "cost_total_parts_bdt" in df_fe.columns and "cost_total_repair_bdt" in df_fe.columns:
        df_fe["fe_parts_vs_repair_cost"] = (
            df_fe["cost_total_parts_bdt"] / (df_fe["cost_total_repair_bdt"] + 1.0)
        ).astype(np.float32)

    if "sensor_short_term_pump_runtime_hours" in df_fe.columns and "sensor_long_term_pump_runtime_hours" in df_fe.columns:
        df_fe["fe_short_vs_long_runtime"] = (
            df_fe["sensor_short_term_pump_runtime_hours"] / (df_fe["sensor_long_term_pump_runtime_hours"] + 0.1)
        ).astype(np.float32)

    if "sensor_short_term_pump_runtime_hours" in df_fe.columns and "sensor_average_daily_pump_runtime_hours" in df_fe.columns:
        df_fe["fe_runtime_vs_avg"] = (
            df_fe["sensor_short_term_pump_runtime_hours"] / (df_fe["sensor_average_daily_pump_runtime_hours"] + 0.1)
        ).astype(np.float32)

    if "sensor_inverter_temperature_celsius" in df_fe.columns and "sensor_ambient_temperature_celsius" in df_fe.columns:
        df_fe["fe_inverter_temp_delta"] = (
            df_fe["sensor_inverter_temperature_celsius"] - df_fe["sensor_ambient_temperature_celsius"]
        ).astype(np.float32)

    if "sensor_panel_surface_temperature_celsius" in df_fe.columns and "sensor_ambient_temperature_celsius" in df_fe.columns:
        df_fe["fe_panel_temp_delta"] = (
            df_fe["sensor_panel_surface_temperature_celsius"] - df_fe["sensor_ambient_temperature_celsius"]
        ).astype(np.float32)

    if "sensor_motor_vibration_level_mm_s" in df_fe.columns and "sensor_avg_vibration_last_month" in df_fe.columns:
        df_fe["fe_vibration_vs_historical"] = (
            df_fe["sensor_motor_vibration_level_mm_s"] / (df_fe["sensor_avg_vibration_last_month"] + 0.1)
        ).astype(np.float32)

    if "sensor_water_salinity_ppm" in df_fe.columns and "base_pump_motor_depth_meters" in df_fe.columns:
        df_fe["fe_salinity_per_depth"] = (
            df_fe["sensor_water_salinity_ppm"] / (df_fe["base_pump_motor_depth_meters"] + 1.0)
        ).astype(np.float32)

    # 4. Composite Risk Flag Intersections
    anomaly_flags = [
        "is_pump_motor_overheating",
        "is_pump_draw_dry",
        "has_constant_pipe_corrosion_issue",
        "has_constant_power_failure_log",
        "is_submersible_pump_non_operational",
        "is_local_technician_unavailable",
        "has_dust_accumulation_on_panels",
        "is_groundwater_level_fluctuating",
    ]
    valid_flags = [c for c in anomaly_flags if c in df_fe.columns]
    if valid_flags:
        sum_flags = df_fe[valid_flags].sum(axis=1).astype(np.float32)
        df_fe["fe_sum_anomaly_flags"] = sum_flags

        if "sensor_motor_vibration_level_mm_s" in df_fe.columns:
            df_fe["fe_risk_interaction_vibration"] = (sum_flags * df_fe["sensor_motor_vibration_level_mm_s"]).astype(np.float32)
        if "sensor_grid_voltage_fluctuation_index" in df_fe.columns:
            df_fe["fe_risk_interaction_voltage"] = (sum_flags * df_fe["sensor_grid_voltage_fluctuation_index"]).astype(np.float32)
        if "fe_inverter_temp_delta" in df_fe.columns:
            df_fe["fe_risk_interaction_temp"] = (sum_flags * df_fe["fe_inverter_temp_delta"]).astype(np.float32)

    return df_fe


def find_best_composite_threshold(y_true: np.ndarray, probas: np.ndarray, oof_auc: float) -> tuple[float, float, dict]:
    best_score = -1.0
    best_thresh = 0.5
    best_metrics = {}
    
    # 981 fine-grained steps (0.01 to 0.99)
    thresholds = np.linspace(0.01, 0.99, 981)
    
    for t in thresholds:
        preds = (probas >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bal_acc = (rec + spec) / 2.0
        acc = accuracy_score(y_true, preds)
        
        # Direct Competition Composite Score formula:
        # 30% F1, 25% ROC-AUC, 15% Precision, 15% Recall, 15% Balanced Accuracy
        composite = (0.30 * f1) + (0.25 * oof_auc) + (0.15 * prec) + (0.15 * rec) + (0.15 * bal_acc)
        
        if composite > best_score:
            best_score = composite
            best_thresh = t
            best_metrics = {
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

    return best_thresh, best_score, best_metrics


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

    print("Engineering time-series lags, derivatives (dx/dt), ratios, and rolling EMA/std...", flush=True)
    train = engineer_features(train)
    test = engineer_features(test)

    # Filter out constant features
    std_series = train.std(axis=0)
    constant_cols = std_series[std_series == 0].index.tolist()
    if constant_cols:
        print(f"Dropping {len(constant_cols)} constant features: {constant_cols[:5]}...", flush=True)
        train.drop(columns=constant_cols, inplace=True)
        test.drop(columns=constant_cols, inplace=True)

    print(f"Final Expanded Feature Space: {train.shape[1]} features.", flush=True)

    # Prepare 5-Fold Stratified CV
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pos_count = int(target.sum())
    neg_count = int(len(target) - pos_count)
    pos_weight = neg_count / max(pos_count, 1)

    print(f"Target balance: {pos_count:,} positive / {neg_count:,} negative (scale_pos_weight = {pos_weight:.2f})", flush=True)

    # Model imports
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier

    # 5 Level-1 Model Families
    oof_lgb = np.zeros(len(train), dtype=np.float64)
    oof_xgb = np.zeros(len(train), dtype=np.float64)
    oof_cat = np.zeros(len(train), dtype=np.float64)
    oof_hgb = np.zeros(len(train), dtype=np.float64)
    oof_et  = np.zeros(len(train), dtype=np.float64)

    test_lgb = np.zeros(len(test), dtype=np.float64)
    test_xgb = np.zeros(len(test), dtype=np.float64)
    test_cat = np.zeros(len(test), dtype=np.float64)
    test_hgb = np.zeros(len(test), dtype=np.float64)
    test_et  = np.zeros(len(test), dtype=np.float64)

    print("\nStarting 5-Fold Stratified Cross-Validation for Level-1 Models (LGBM, XGB, Cat, HGB, ExtraTrees)...", flush=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train, target), 1):
        print(f"\n--- Fold {fold}/{N_SPLITS} ---", flush=True)
        x_tr, y_tr = train.iloc[train_idx], target.iloc[train_idx]
        x_va, y_va = train.iloc[val_idx], target.iloc[val_idx]

        # 1. LightGBM
        model_lgb = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=2000,
            learning_rate=0.02,
            num_leaves=31,
            max_depth=6,
            min_child_samples=20,
            subsample=0.80,
            colsample_bytree=0.80,
            scale_pos_weight=pos_weight * 0.7,
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
            n_estimators=2000,
            learning_rate=0.02,
            max_depth=6,
            subsample=0.80,
            colsample_bytree=0.80,
            scale_pos_weight=pos_weight * 0.7,
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
            iterations=1600,
            learning_rate=0.03,
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
            max_iter=300,
            learning_rate=0.03,
            max_leaf_nodes=31,
            max_depth=6,
            class_weight="balanced",
            l2_regularization=1.0,
            random_state=RANDOM_STATE + fold,
        )
        model_hgb.fit(x_tr, y_tr)
        val_p_hgb = model_hgb.predict_proba(x_va)[:, 1]
        oof_hgb[val_idx] = val_p_hgb
        test_hgb += model_hgb.predict_proba(test)[:, 1] / N_SPLITS

        # 5. ExtraTrees
        model_et = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
        )
        model_et.fit(x_tr, y_tr)
        val_p_et = model_et.predict_proba(x_va)[:, 1]
        oof_et[val_idx] = val_p_et
        test_et += model_et.predict_proba(test)[:, 1] / N_SPLITS

        print(
            f"Fold {fold} OOF ROC-AUC -> LGB: {roc_auc_score(y_va, val_p_lgb):.4f} | "
            f"XGB: {roc_auc_score(y_va, val_p_xgb):.4f} | "
            f"Cat: {roc_auc_score(y_va, val_p_cat):.4f} | "
            f"HGB: {roc_auc_score(y_va, val_p_hgb):.4f} | "
            f"ET: {roc_auc_score(y_va, val_p_et):.4f}",
            flush=True,
        )

    # Stage 2 Stacking Meta-Learner Training
    print("\nTraining Level-2 Meta-Learner Stacking Classifier...", flush=True)
    oof_meta_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_hgb, oof_et])
    test_meta_matrix = np.column_stack([test_lgb, test_xgb, test_cat, test_hgb, test_et])

    meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    meta_model.fit(oof_meta_matrix, target)
    
    oof_meta_probs = meta_model.predict_proba(oof_meta_matrix)[:, 1]
    test_meta_probs = meta_model.predict_proba(test_meta_matrix)[:, 1]

    overall_auc = roc_auc_score(target, oof_meta_probs)
    overall_pr_auc = average_precision_score(target, oof_meta_probs)
    print(f"\nLevel-2 Stacking Ensemble OOF ROC-AUC: {overall_auc:.4f}", flush=True)
    print(f"Level-2 Stacking Ensemble OOF PR-AUC:  {overall_pr_auc:.4f}", flush=True)

    # High-Resolution Threshold Optimization for Competition Composite Metric
    print("\nOptimizing classification threshold directly on Competition Composite Metric (0.001 resolution)...", flush=True)
    best_thresh, best_comp_score, m = find_best_composite_threshold(target.to_numpy(), oof_meta_probs, overall_auc)
    
    print(f"\n>>> PEAK COMPOSITE SCORE ACHIEVED: {best_comp_score:.4f} @ Threshold {best_thresh:.4f} <<<", flush=True)

    metrics = {
        "stacking_meta_coefs": list(meta_model.coef_[0]),
        "oof_roc_auc": float(overall_auc),
        "oof_pr_auc": float(overall_pr_auc),
        "best_threshold_for_composite": float(best_thresh),
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
    print("\nFinal Stacking Validation Metrics & Composite Score:")
    print(json.dumps(metrics, indent=2))

    # Output strict submission.csv
    test_probs_clean = np.clip(test_meta_probs, 0.0, 1.0)
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
    print(f"\nWrote final optimized {args.output_dir / 'submission.csv'} with {len(submission):,} rows.")


if __name__ == "__main__":
    main()