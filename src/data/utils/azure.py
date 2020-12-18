# Azure keyvault dependencies
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

# Required for Azure Data Lake Storage Gen1 filesystem management
from azure.datalake.store import core, lib

import logging

# Set the logging level for all azure-* libraries
azure_logger = logging.getLogger('azure')
azure_logger.setLevel(logging.WARNING)


def secret_client(key_vault_url: str) -> SecretClient:
    # Get credentials
    credentials = DefaultAzureCredential()

    # Create a secret client
    secret_client = SecretClient(
        key_vault_url,  # Your KeyVault URL
        credentials
    )

    return secret_client


def adls_client(key_vault_url: str, store_name: str) -> core.AzureDLFileSystem:

    sc = secret_client(key_vault_url)

    adlCreds = lib.auth(
        tenant_id=sc.get_secret("tenantid").value,
        client_id=sc.get_secret("spclientid").value,
        client_secret=sc.get_secret("spclientsecret").value,
        resource="https://datalake.azure.net/"
    )

    # Create a filesystem client object
    adlsFileSystemClient = core.AzureDLFileSystem(
        adlCreds, store_name=store_name)

    return adlsFileSystemClient
