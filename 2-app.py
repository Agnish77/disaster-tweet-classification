## ML Model Deployment at Streamlit Server

import streamlit as st
import os
import torch
from transformers import pipeline
import boto3

# -----------------------------
# AWS S3 SETTINGS
# -----------------------------
bucket_name = "agnishpaul"

AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = st.secrets["AWS_DEFAULT_REGION"]

# Local folder where model will be downloaded
local_path = "tinybert-disaster-tweet"

# Folder in S3 bucket
s3_prefix = "ml-models/tinybert-disaster-tweet/"

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION,
)

# -----------------------------
# FUNCTION TO DOWNLOAD FROM S3
# -----------------------------
def download_dir(local_path, s3_prefix):
    os.makedirs(local_path, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if "Contents" in result:
            for obj in result["Contents"]:
                s3_key = obj["Key"]

                # Remove prefix and build local file path
                relative_path = os.path.relpath(s3_key, s3_prefix)
                local_file_path = os.path.join(local_path, relative_path)

                # Create directories if needed
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download file
                s3.download_file(bucket_name, s3_key, local_file_path)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Disaster Tweet Classification 🚨")

if st.button("Download Model"):
    with st.spinner("Downloading model from S3..."):
        download_dir(local_path, s3_prefix)
    st.success("Download complete!")

text = st.text_area("Enter Tweet", "Type here...")

if st.button("Predict"):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # ******** THE FIX IS HERE *********
    classifier = pipeline(
        "text-classification",
        model="./tinybert-disaster-tweet",   # <--- Correct local folder
        device=device
    )

    with st.spinner("Predicting..."):
        output = classifier(text)
        st.write(output)
