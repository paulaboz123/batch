import argparse
from azure.ai.ml import MLClient
from azure.ai.ml.entities import BatchEndpoint, BatchDeployment
from azure.identity import DefaultAzureCredential

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subscription", required=True)
    p.add_argument("--resource-group", required=True)
    p.add_argument("--workspace", required=True)
    p.add_argument("--endpoint-yml", default="deployment/endpoint-dev.yml")
    p.add_argument("--deployments", nargs="+", default=[
        "deployment/de-model.yml",
        "deployment/pl-model.yml",
        "deployment/en-model.yml",
    ])
    args = p.parse_args()

    ml_client = MLClient(DefaultAzureCredential(), args.subscription, args.resource_group, args.workspace)

    endpoint = BatchEndpoint.load(args.endpoint_yml)
    ml_client.batch_endpoints.begin_create_or_update(endpoint).result()

    for dep_yml in args.deployments:
        dep = BatchDeployment.load(dep_yml)
        ml_client.batch_deployments.begin_create_or_update(dep).result()

if __name__ == "__main__":
    main()
