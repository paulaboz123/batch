# Azure ML Batch Endpoint (AAD) – E2E repo

## What you get
- Batch endpoint YAML (auth_mode=aad_token)
- 3 batch deployments (de/pl/en) YAML
- Batch `score.py` adapter (same JSON schema as your online endpoint, different transport)
- Azure DevOps pipeline: deploy endpoint + deployments + upload sample input + invoke smoke test
- Python SDK v2 script + notebook

## Required changes (IMPORTANT)
1) Edit `deployment/*-model.yml`
- `compute: azureml:<your_aml_compute_cluster>` (AmlCompute)
- `environment: azureml:<env_name>:<version>` (pin a version; avoid @latest)
- `model: azureml:<model_name>:<version>` (pin in prod)

2) Edit `azure-pipelines.yml` parameters
- `resourceGroup`, `workspace`, `serviceConnection`

3) Replace the placeholder scoring logic in `training/score.py` with your real `inference(document, num_preds)`.

## Invoke
Pipeline already submits a smoke-test invoke using the `albot-sample-input:1` data asset.
Outputs go to workspaceblobstore under `batch_outputs/smoke/...`.
