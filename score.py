import json
import os
import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Replace these globals with your real model objects ---
_loaded = False

def init():
    global _loaded
    # In real code: load artifacts from AZUREML_MODEL_DIR
    model_dir = os.environ.get("AZUREML_MODEL_DIR")
    logger.info(f"AZUREML_MODEL_DIR={model_dir}")
    _loaded = True

def _predict_one(rec: Dict[str, Any]) -> Dict[str, Any]:
    # Keep input schema identical to your online endpoint
    if "document" not in rec or "num_preds" not in rec:
        raise ValueError("Expected keys: 'document' and 'num_preds'")
    # TODO: call your existing inference(document, num_preds)
    return {"ok": True, "num_preds": int(rec["num_preds"]), "document": rec["document"]}

def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    if content.startswith("{"):
        return [json.loads(content)]
    out = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out

def run(mini_batch: Union[List[str], "pandas.DataFrame"]):
    if not _loaded:
        # Safety: init should run first, but don't fail silently
        init()

    results = []

    # Typical case: list of file paths
    if isinstance(mini_batch, list):
        for path in mini_batch:
            for rec in _read_json_or_jsonl(path):
                results.append(_predict_one(rec))
        return results

    # DataFrame case (if you configure tabular input)
    try:
        if hasattr(mini_batch, "to_dict"):
            rows = mini_batch.to_dict(orient="records")
            for rec in rows:
                if isinstance(rec.get("document"), str):
                    rec["document"] = json.loads(rec["document"])
                results.append(_predict_one(rec))
            return results
    except Exception:
        pass

    raise ValueError(f"Unsupported mini_batch type: {type(mini_batch)}")
