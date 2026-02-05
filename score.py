\
import json
import os
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd

_model = None
_label_map = None


def init() -> None:
    """Loads artifacts once per node."""
    global _model, _label_map

    model_dir = os.environ.get("AZUREML_MODEL_DIR", ".")
    model_path = os.path.join(model_dir, "model.joblib")
    label_map_path = os.path.join(model_dir, "label_map.json")

    _model = joblib.load(model_path)

    if os.path.exists(label_map_path):
        with open(label_map_path, "r", encoding="utf-8") as f:
            _label_map = json.load(f)
    else:
        _label_map = None


def _read_inputs(mini_batch: Union[List[str], pd.DataFrame]) -> pd.DataFrame:
    """Supports JSONL/JSON, CSV, Parquet or direct DataFrame inputs."""
    if isinstance(mini_batch, pd.DataFrame):
        return mini_batch

    rows: List[Dict[str, Any]] = []
    for path in mini_batch:
        lower = path.lower()
        if lower.endswith(".jsonl") or lower.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
        elif lower.endswith(".csv"):
            df = pd.read_csv(path)
            rows.extend(df.to_dict(orient="records"))
        elif lower.endswith(".parquet"):
            df = pd.read_parquet(path)
            rows.extend(df.to_dict(orient="records"))
        else:
            raise ValueError(f"Unsupported file type: {path}")

    return pd.DataFrame.from_records(rows)


def run(mini_batch: Union[List[str], pd.DataFrame]) -> List[Dict[str, Any]]:
    """
    Example expects input column:
      - text: str
    Optional:
      - id: identifier
    Adapt feature extraction to your model.
    """
    if _model is None:
        raise RuntimeError("Model not initialized; init() not called.")

    df = _read_inputs(mini_batch)

    if "text" not in df.columns:
        raise ValueError("Input must contain a 'text' column.")

    texts = df["text"].astype(str).tolist()

    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(texts)
        preds = np.argmax(proba, axis=1)
        conf = np.max(proba, axis=1)
    else:
        preds = _model.predict(texts)
        conf = np.full(shape=(len(texts),), fill_value=np.nan)

    outputs: List[Dict[str, Any]] = []
    for i, pred in enumerate(preds):
        label = str(pred)
        if _label_map is not None:
            label = _label_map.get(str(pred), label)

        record_id = df["id"].iloc[i] if "id" in df.columns else i
        outputs.append(
            {
                "id": record_id,
                "label": label,
                "confidence": float(conf[i]) if conf is not None and not np.isnan(conf[i]) else None,
            }
        )

    return outputs
