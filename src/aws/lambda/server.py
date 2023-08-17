from typing import Annotated

import io
import numpy as np
import onnxruntime as ort

from PIL import Image
from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from mangum import Mangum

app = FastAPI(
    title="AWS + FastAPI",
    description="AWS API Gateway, Lambdas and FastAPI (oh my)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("loading model...")
ort_session = ort.InferenceSession("resnetv2_50.onnx")
ort_session.run(['output'], {'input': np.random.randn(
    1, 3, 224, 224).astype(np.float32)})
print("model loaded ...")

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

print("getting classnames")
with open("imagenet_classes.txt", "r") as f:
    classes_response = f.read()
classes_list = [line.strip() for line in classes_response.split('\n')]

@app.post("/infer")
async def infer(image: Annotated[bytes, File()]):
    img = Image.open(io.BytesIO(image))
    img = img.convert("RGB")
    img = img.resize((224, 224))

    img_np = np.array(img)

    img_np = img_np / 255.0
    img_np = (img_np - mean) / std
    img_np = img_np.transpose(2, 0, 1)

    ort_outputs = ort_session.run(
        ['output'], {'input': img_np[None, ...].astype(np.float32)})

    pred_class_idx = np.argmax(ort_outputs[0])
    predicted_class = classes_list[pred_class_idx]

    return {
        "predicted": predicted_class,
    }

@app.get("/health")
async def health():
    return {
        "message": "ok"
    }

@app.get("/")
async def root():
    return {
        "message": "Welcome to Classification Endpoint V1"
    }

handler = Mangum(app=app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)