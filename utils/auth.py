import base64
import json

import ee
import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account


def get_credentials():
    service_account_info = json.loads(
        base64.b64decode(st.secrets["GOOGLE_SERVICE_ACCOUNT_KEY"])
    )
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    return credentials


def initialize_ee():
    """Initialize Earth Engine with service account credentials"""
    try:
        # Check if secrets are available
        if (
            "GOOGLE_SERVICE_ACCOUNT_KEY" in st.secrets
            and "GOOGLE_CLOUD_PROJECT_NAME" in st.secrets
        ):
            # Try to decode the key
            try:
                decoded = base64.b64decode(st.secrets["GOOGLE_SERVICE_ACCOUNT_KEY"])
                service_account_info = json.loads(decoded)

                # Create credentials
                credentials = service_account.Credentials.from_service_account_info(
                    service_account_info,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )

                # Initialize Earth Engine
                ee.Initialize(
                    credentials=credentials,
                    project=st.secrets["GOOGLE_CLOUD_PROJECT_NAME"],
                )
            except Exception:
                # Fallback initialization
                ee.Initialize(project="ee-ciut")
        else:
            # Fallback initialization
            ee.Initialize(project="ee-ciut")

    except Exception as e:
        st.error(f"Earth Engine initialization failed: {str(e)}")
        raise


def get_storage_client(credentials):
    return storage.Client(credentials=credentials)


def get_bucket_name():
    return st.secrets["GOOGLE_CLOUD_BUCKET_NAME"]
