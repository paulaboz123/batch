# Azure ML v2 — Batch endpoint: 1 endpoint, 3 deployments (DE/EN/PL)

This repo contains:
- **Python (SDK v2)** code to create/update a *single* Batch Endpoint with **3 deployments**.
- **YAML** definitions for endpoint and deployments.
- **Azure DevOps** pipeline example.
- Example **score.py** for a *model-based batch deployment*.

## What you need to set
### Azure resources
- `SUBSCRIPTION_ID`, `RESOURCE_GROUP`, `WORKSPACE_NAME`
- 3 computes (each deployment uses its own):
  - `COMPUTE_DE`, `COMPUTE_EN`, `COMPUTE_PL`
  - Recommended: AmlCompute clusters with `max_instances=1` each (your “3 VMs”).

### Models
You can reference models by **name** (latest) via env vars:
- `MODEL_NAME_DE`, `MODEL_NAME_EN`, `MODEL_NAME_PL`

(You can also pin versions in YAML if you prefer.)

## Folder layout
- `azureml/endpoint.yml`
- `azureml/deployments/{de,en,pl}.yml`
- `src/score.py`
- `src/conda.yml`
- `scripts/create_or_update.py`
- `scripts/smoke_submit.py`
- `notebooks/batch_endpoint_setup.ipynb`
- `devops/azure-pipelines.yml`

## Quick start (local)
```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

export SUBSCRIPTION_ID="..."
export RESOURCE_GROUP="..."
export WORKSPACE_NAME="..."
export ENDPOINT_NAME="my-batch-ep"
export LOCATION="westeurope"

export COMPUTE_DE="cpu-de"
export COMPUTE_EN="cpu-en"
export COMPUTE_PL="cpu-pl"

export MODEL_NAME_DE="clf-de"
export MODEL_NAME_EN="clf-en"
export MODEL_NAME_PL="clf-pl"

python scripts/create_or_update.py
```

## Submit a smoke test job
```bash
python scripts/smoke_submit.py --endpoint-name my-batch-ep --deployment-name en --input-path azureml://datastores/workspaceblobstore/paths/samples/input.jsonl
```

## Notes
- Batch deployments **do not route traffic** like online endpoints. You choose the deployment when you submit the job.
- `score.py` is invoked for each mini-batch. Keep it pure, deterministic, and avoid network calls.
