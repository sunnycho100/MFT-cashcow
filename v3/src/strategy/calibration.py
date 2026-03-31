"""Probability calibration helpers for the v3 hybrid stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from v2.src.features.pipeline import get_feature_names
from v2.src.models.lgbm_model import LGBMModel


@dataclass(slots=True)
class CalibrationBundle:
    method: str
    calibrator: Any | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


def _safe_auc(y_true: np.ndarray, probs: np.ndarray) -> float | None:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, probs))


def fit_probability_calibrator(
    raw_scores: np.ndarray,
    targets: np.ndarray,
    method: str = "raw",
) -> CalibrationBundle:
    raw_scores = np.asarray(raw_scores, dtype=float)
    targets = np.asarray(targets, dtype=int)
    normalized_method = str(method or "raw").lower()

    raw_metrics = {
        "raw_brier": round(float(brier_score_loss(targets, raw_scores)), 6) if len(targets) else None,
        "raw_auc": round(_safe_auc(targets, raw_scores), 6) if _safe_auc(targets, raw_scores) is not None else None,
    }

    if normalized_method in {"raw", "none"} or len(raw_scores) == 0 or len(np.unique(targets)) < 2:
        return CalibrationBundle(
            method="raw",
            calibrator=None,
            metrics={**raw_metrics, "calibrated_brier": raw_metrics["raw_brier"], "calibrated_auc": raw_metrics["raw_auc"]},
        )

    if normalized_method == "isotonic":
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(raw_scores, targets)
        calibrated = calibrator.predict(raw_scores)
    elif normalized_method == "sigmoid":
        calibrator = LogisticRegression(max_iter=1000)
        calibrator.fit(raw_scores.reshape(-1, 1), targets)
        calibrated = calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f"Unsupported calibration method: {method}")

    return CalibrationBundle(
        method=normalized_method,
        calibrator=calibrator,
        metrics={
            **raw_metrics,
            "calibrated_brier": round(float(brier_score_loss(targets, calibrated)), 6),
            "calibrated_auc": round(_safe_auc(targets, calibrated), 6) if _safe_auc(targets, calibrated) is not None else None,
        },
    )


def apply_probability_calibration(raw_scores: np.ndarray, calibration: CalibrationBundle | None) -> np.ndarray:
    probs = np.asarray(raw_scores, dtype=float)
    if calibration is None or calibration.calibrator is None or calibration.method == "raw":
        return probs
    if calibration.method == "isotonic":
        return np.asarray(calibration.calibrator.predict(probs), dtype=float)
    if calibration.method == "sigmoid":
        return np.asarray(calibration.calibrator.predict_proba(probs.reshape(-1, 1))[:, 1], dtype=float)
    raise ValueError(f"Unsupported calibration method: {calibration.method}")


def predict_raw_probabilities(
    model: LGBMModel,
    frame: pl.DataFrame,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    names = feature_names or model.feature_names or get_feature_names(frame)
    features = frame.select(names).to_pandas()
    return np.asarray(model.model.predict_proba(features)[:, 1], dtype=float)


def predict_frame_with_calibration(
    model: LGBMModel,
    frame: pl.DataFrame,
    calibration: CalibrationBundle | None = None,
    feature_names: list[str] | None = None,
) -> pl.DataFrame:
    raw_scores = predict_raw_probabilities(model, frame, feature_names=feature_names)
    calibrated = apply_probability_calibration(raw_scores, calibration)

    pred_class = np.full(len(calibrated), 1, dtype=np.int32)
    pred_class[calibrated >= 0.5] = 2
    pred_class[calibrated <= 0.30] = 0

    return frame.with_columns(
        pl.Series("pred_class", pred_class),
        pl.Series("pred_raw_long_prob", raw_scores),
        pl.Series("pred_long_prob", calibrated),
        pl.Series("pred_prob_up", calibrated),
        pl.Series("pred_prob_down", 1.0 - calibrated),
        pl.Series("pred_prob_flat", np.zeros(len(calibrated))),
    )


def train_model_with_tail_calibration(
    config: dict,
    pair_frames: dict[str, pl.DataFrame],
    calibration_method: str = "raw",
    calibration_fraction: float = 0.2,
) -> tuple[LGBMModel, list[str], CalibrationBundle, dict[str, Any]]:
    names = get_feature_names(next(iter(pair_frames.values())))
    model = LGBMModel(config)

    train_parts = []
    calib_parts = []
    for frame in pair_frames.values():
        labeled = model.create_labels(frame)
        if len(labeled) < 10:
            continue
        split_idx = int(len(labeled) * (1.0 - calibration_fraction))
        split_idx = min(max(split_idx, 1), len(labeled) - 1)
        train_parts.append(labeled[:split_idx])
        calib_parts.append(labeled[split_idx:])

    train_df = pl.concat(train_parts)
    calib_df = pl.concat(calib_parts)
    metrics = model.train(train_df, calib_df, names)

    raw_calib = predict_raw_probabilities(model, calib_df, feature_names=names)
    calibration = fit_probability_calibrator(raw_calib, calib_df["target"].to_numpy(), method=calibration_method)
    metrics["calibration"] = calibration.metrics
    metrics["calibration_method"] = calibration.method
    return model, names, calibration, metrics
