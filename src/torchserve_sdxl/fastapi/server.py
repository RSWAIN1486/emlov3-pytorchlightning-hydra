from typing import Annotated

import io
import numpy as np
import requests
import json
import uuid

import boto3
from botocore.client import Config

from PIL import Image
from fastapi import FastAPI, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

app = FastAPI(
    title="Stable Diffusion XL - 1.0",
    description="Text to Image Service",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

s3_client = boto3.client(
    "s3",
    config=Config(
        region_name="ap-south-1",
        signature_version="s3v4",
        s3={"addressing_style": "path"},
    ),
)
bucket_name = "torchserve-sdxl"
objects_prefix = "sdxl-outputs"

results_map = {}

# # fill results_map with data in S3
# s3_results = s3_client.list_objects(Bucket=bucket_name, Prefix=objects_prefix)[
#     "Contents"
# ]
# for res in s3_results:
#     job_id = res["Key"].split("/")[1]
#     results_map[job_id] = {"status": "SUCCESS", "result": res["Key"]}

# List objects in the S3 bucket
response = s3_client.list_objects(Bucket=bucket_name, Prefix=objects_prefix)

# Check if 'Contents' key exists in the response
if 'Contents' in response:
    # 'Contents' key exists, so populate results_map
    s3_results = response['Contents']
    # Fill results_map with data from S3
    for res in s3_results:
        job_id = res["Key"].split("/")[1]
        results_map[job_id] = {"status": "SUCCESS", "result": res["Key"]}

else:
    # 'Contents' key does not exist, handle the case where there are no matching objects
    print("No objects found in S3 bucket with the specified prefix.")

def submit_inference(uid: str, text: str):
    results_map[uid] = {"status": "PENDING"}
    try:
        
        response = requests.post("http://localhost:8080/predictions/sdxl", data=text)
        # Contruct image from response
        image = Image.fromarray(np.array(json.loads(response.text), dtype="uint8"))
        # image = Image.open("out.jpg")

        img_bytes = io.BytesIO()

        image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        # upload image and text to s3
        filename = f"{objects_prefix}/{uid}/result.jpeg"
        s3_client.upload_fileobj(img_bytes, bucket_name, filename)

        # save the s3 uri here
        results_map[uid] = {"status": "SUCCESS", "result": filename}
    except Exception as e:
        print(f"ERROR :: {e}")
        results_map[uid] = {"status": "ERROR"}

@app.post("/text-to-image")
async def text_to_image(text: str, background_tasks: BackgroundTasks):
    uid = str(uuid.uuid4())
    results_map[uid] = {"status": "PENDING"}
    background_tasks.add_task(submit_inference, uid, text)

    return {"job-id": uid, "message": "job submitted successfully"}

@app.get("/results")
async def results(uid: str):
    # print(results_map)
    if uid not in results_map:
        return {"message": f"job-id={uid} is invalid", "status": "ERROR"}

    if results_map[uid]["status"] == "SUCCESS":
        obj_prefix = results_map[uid]["result"]

        print(f"{obj_prefix=}, {uid=}, {bucket_name=}")

        presigned_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": obj_prefix},
            ExpiresIn=3600 * 36,  # 36 hours
        )

        return {"url": presigned_url, "status": "SUCCESS"}

    return {
        "message": f"job status={results_map[uid]}",
        "status": results_map[uid]["status"],
    }

@app.get("/health")
async def health():
    return {"message": "ok"}

@app.get("/")
async def root():
    return {"message": "Welcome to Stable Diffusion XL Endpoint V1"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9080)