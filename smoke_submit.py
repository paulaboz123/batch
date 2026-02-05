\
import argparse
import os

from azure.ai.ml import MLClient, Input
from azure.identity import DefaultAzureCredential


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--deployment-name", required=True, choices=["de", "en", "pl"])
    parser.add_argument("--input-path", required=True, help="azureml://..., https://..., or local path")
    args = parser.parse_args()

    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)

    ml_client = MLClient(
        credential,
        os.environ["SUBSCRIPTION_ID"],
        os.environ["RESOURCE_GROUP"],
        os.environ["WORKSPACE_NAME"],
    )

    input_type = "uri_folder" if args.input_path.endswith("/") else "uri_file"
    job = ml_client.batch_endpoints.invoke(
        endpoint_name=args.endpoint_name,
        deployment_name=args.deployment_name,
        input=Input(type=input_type, path=args.input_path),
    )

    print(f"✅ Submitted batch job: {job.name}")


if __name__ == "__main__":
    main()
