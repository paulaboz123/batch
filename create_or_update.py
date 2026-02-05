\
import os
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import BatchEndpoint, BatchDeployment
from azure.identity import DefaultAzureCredential

ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name, default)
    if not v:
        raise ValueError(f"Missing required environment variable: {name}")
    return v


def main() -> None:
    subscription_id = _env("SUBSCRIPTION_ID")
    resource_group = _env("RESOURCE_GROUP")
    workspace_name = _env("WORKSPACE_NAME")

    endpoint_name = _env("ENDPOINT_NAME")
    location = os.environ.get("LOCATION", "westeurope")

    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

    endpoint = BatchEndpoint(
        name=endpoint_name,
        description="Batch endpoint with 3 deployments (de/en/pl)",
        auth_mode="aad_token",
        location=location,
    )
    ml_client.batch_endpoints.begin_create_or_update(endpoint).result()

    deploy_paths = [
        ROOT / "azureml" / "deployments" / "de.yml",
        ROOT / "azureml" / "deployments" / "en.yml",
        ROOT / "azureml" / "deployments" / "pl.yml",
    ]

    for p in deploy_paths:
        deployment = BatchDeployment.load(p)
        ml_client.batch_deployments.begin_create_or_update(deployment).result()

    print(f"✅ Batch endpoint '{endpoint_name}' upserted with deployments: de, en, pl")


if __name__ == "__main__":
    main()
